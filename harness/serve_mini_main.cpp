// Minimal OpenAI-compatible HTTP server for the NInfer->HIP build.
//
// The probe harness (mini_main.cpp) already drives the engine, but it only speaks the command line.
// This entry keeps the same Engine setup and exposes it over HTTP so tools that expect OpenAI's API
// can talk to it: /health, /v1/models and /v1/chat/completions (text and image inputs).
//
// One request runs at a time: the engine owns a single set of KV arenas and the migration target has
// not been audited for concurrent requests yet, so the mutex is a correctness guard, not a throttle.

#include "ninfer/engine.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using nlohmann::json;
using namespace ninfer;

namespace {

std::mutex g_engine_mutex;
bool g_vision_available = true;

std::string base64_decode(const std::string& encoded) {
    static const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::array<int, 256> reverse{};
    reverse.fill(-1);
    for (std::size_t index = 0; index < alphabet.size(); ++index) {
        reverse[static_cast<unsigned char>(alphabet[index])] = static_cast<int>(index);
    }
    std::string out;
    out.reserve(encoded.size() * 3 / 4 + 3);
    int accumulator = 0;
    int bits        = 0;
    for (const char character : encoded) {
        const int value = reverse[static_cast<unsigned char>(character)];
        if (value < 0) { continue; }
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((accumulator >> bits) & 0xFF));
        }
    }
    return out;
}

// Accepts either a data URL (base64 payload after the comma) or a bare base64 string.
bool decode_image(const std::string& url, std::string* media_type, std::vector<std::uint8_t>* bytes) {
    std::string payload = url;
    *media_type         = "image/png";
    if (url.rfind("data:", 0) == 0) {
        const std::size_t semicolon = url.find(';');
        const std::size_t comma     = url.find(',');
        if (semicolon == std::string::npos || comma == std::string::npos || comma < semicolon) {
            return false;
        }
        *media_type = url.substr(5, semicolon - 5);
        payload     = url.substr(comma + 1);
    }
    const std::string raw = base64_decode(payload);
    if (raw.empty()) { return false; }
    bytes->assign(raw.begin(), raw.end());
    return true;
}

ChatRole parse_role(const std::string& role) {
    if (role == "system") { return ChatRole::System; }
    if (role == "assistant") { return ChatRole::Assistant; }
    if (role == "tool") { return ChatRole::Tool; }
    return ChatRole::User;
}

std::string finish_reason_name(FinishReason reason) {
    switch (reason) {
        case FinishReason::OutputLimit: return "length";
        case FinishReason::ContextCapacity: return "length";
        case FinishReason::StopToken: return "stop";
        case FinishReason::StopString: return "stop";
        case FinishReason::Cancelled: return "stop";
        case FinishReason::None: return "stop";
    }
    return "stop";
}

std::string error_body(const std::string& message, const char* type) {
    json body;
    body["error"] = json{{"message", message}, {"type", type}, {"code", nullptr}};
    return body.dump();
}

// Chunk stream shared between the generation worker and the HTTP writer.
class ChunkQueue {
public:
    void push(std::string text) {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            items_.push_back(std::move(text));
        }
        ready_.notify_one();
    }

    void close() {
        {
            std::lock_guard<std::mutex> guard(mutex_);
            closed_ = true;
        }
        ready_.notify_all();
    }

    bool pop(std::string* out) {
        std::unique_lock<std::mutex> guard(mutex_);
        ready_.wait(guard, [this] { return closed_ || !items_.empty(); });
        if (items_.empty()) { return false; }
        *out = std::move(items_.front());
        items_.pop_front();
        return true;
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::string> items_;
    bool closed_ = false;
};

std::string sse_chunk(const std::string& id, const std::string& model, const char* field,
                      const std::string& text, bool first) {
    json payload;
    payload["id"]      = id;
    payload["object"]  = "chat.completion.chunk";
    payload["created"] = 0;
    payload["model"]   = model;
    json delta         = json::object();
    if (first) { delta["role"] = "assistant"; }
    delta[field] = text;
    payload["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}});
    return "data: " + payload.dump() + "\n\n";
}

// Forwards engine output deltas to the HTTP chunk queue as they are produced.
class SseSink : public OutputSink {
public:
    SseSink(std::shared_ptr<ChunkQueue> queue, std::string id, std::string model)
        : queue_(std::move(queue)), id_(std::move(id)), model_(std::move(model)) {}

    void start(GenerationStart) override {
        queue_->push(sse_chunk(id_, model_, "content", "", true));
    }
    void progress(PromptProgress) override {}
    void timing(GenerationTimingObservation) override {}

    void publish(OutputDelta delta) override {
        const char* field =
            delta.channel == OutputChannel::Reasoning ? "reasoning_content" : "content";
        queue_->push(sse_chunk(id_, model_, field, delta.text, false));
    }

private:
    std::shared_ptr<ChunkQueue> queue_;
    std::string id_;
    std::string model_;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <artifact> [--ctx N] [--kv rk4v4-e8] [--port 8080] [--host 127.0.0.1]\n"
                     "       [--kv-retrieval 1] [--resident-tokens N] [--archive-mib N] [--api-key KEY]\n",
                     argv[0]);
        return 2;
    }

    EngineOptions options;
    options.artifact_path = argv[1];

    std::string host   = "127.0.0.1";
    int port           = 8080;
    std::string api_key;
    std::string model_name = "ninfer-qwen3";

    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next                  = [&]() -> std::string {
            return (index + 1 < argc) ? argv[++index] : std::string{};
        };
        if (argument == "--ctx") {
            const std::uint32_t context = static_cast<std::uint32_t>(std::stoul(next()));
            options.max_context         = context;
            options.kv_capacity         = KvCapacityPolicy::explicit_capacity(context);
        } else if (argument == "--kv") {
            const std::string mode = next();
            _putenv_s("NINFER_KV", mode.c_str());
        } else if (argument == "--resident-tokens") {
            _putenv_s("NINFER_RESIDENT_TOKENS", next().c_str());
        } else if (argument == "--sink-tokens") {
            _putenv_s("NINFER_SINK_TOKENS", next().c_str());
        } else if (argument == "--archive-mib") {
            const std::string value = next();
            _putenv_s("NINFER_KV_ARCHIVE_MIB", value.c_str());
            _putenv_s("NINFER_KV_ARCHIVE", "1");
        } else if (argument == "--kv-retrieval") {
            _putenv_s("NINFER_KV_RETRIEVE", next().c_str());
        } else if (argument == "--kv-recall-pages") {
            _putenv_s("NINFER_KV_RECALL_PAGES", next().c_str());
        } else if (argument == "--spec") {
            _putenv_s("NINFER_SPEC", next().c_str());
        } else if (argument == "--draft-tokens") {
            _putenv_s("NINFER_DRAFT_TOKENS", next().c_str());
        } else if (argument == "--port") {
            port = std::stoi(next());
        } else if (argument == "--host") {
            host = next();
        } else if (argument == "--api-key") {
            api_key = next();
        } else if (argument == "--model-name") {
            model_name = next();
        } else {
            std::fprintf(stderr, "[serve] ignoring unknown argument %s\n", argument.c_str());
        }
    }

    std::fprintf(stderr, "[serve] loading %s\n", options.artifact_path.c_str());
    options.enable_vision = true;
    std::unique_ptr<Engine> engine;
    try {
        engine = std::make_unique<Engine>(std::move(options));
    } catch (const std::exception& error) {
        // Not every artifact ships a vision tower; fall back to text-only instead of refusing to run.
        std::fprintf(stderr, "[serve] vision unavailable (%s); retrying text-only\n", error.what());
        EngineOptions text_only;
        text_only.artifact_path = argv[1];
        text_only.enable_vision = false;
        engine                  = std::make_unique<Engine>(std::move(text_only));
        g_vision_available      = false;
    }
    std::fprintf(stderr, "[serve] engine ready (vision=%s)\n", g_vision_available ? "on" : "off");

    httplib::Server server;

    auto authorized = [&api_key](const httplib::Request& request) {
        if (api_key.empty()) { return true; }
        const auto header = request.get_header_value("Authorization");
        return header == ("Bearer " + api_key);
    };

    server.Get("/health", [](const httplib::Request&, httplib::Response& response) {
        response.set_content("{\"status\":\"ok\"}", "application/json");
    });

    server.Get("/v1/models", [&model_name](const httplib::Request&, httplib::Response& response) {
        json models;
        models["object"] = "list";
        models["data"]   = json::array({{{"id", model_name},
                                         {"object", "model"},
                                         {"owned_by", "ninfer"},
                                         {"created", 0}}});
        response.set_content(models.dump(), "application/json");
    });

    server.Post("/v1/chat/completions", [&](const httplib::Request& request,
                                            httplib::Response& response) {
        if (!authorized(request)) {
            response.status = 401;
            response.set_content(error_body("invalid api key", "invalid_request_error"),
                                 "application/json");
            return;
        }

        json body;
        try {
            body = json::parse(request.body);
        } catch (const std::exception& error) {
            response.status = 400;
            response.set_content(error_body(std::string("invalid JSON: ") + error.what(),
                                            "invalid_request_error"),
                                 "application/json");
            return;
        }

        PromptInput input;
        bool think = body.value("enable_thinking", false);
        input.options.enable_thinking = think;

        const auto messages = body.value("messages", json::array());
        for (const auto& message : messages) {
            ChatMessage out;
            out.role = parse_role(message.value("role", "user"));
            const auto& content = message.contains("content") ? message.at("content") : json();
            if (content.is_string()) {
                out.parts.push_back(MessagePart{MessagePartKind::Text, content.get<std::string>(), {}});
            } else if (content.is_array()) {
                for (const auto& part : content) {
                    const std::string type = part.value("type", "");
                    if (type == "text") {
                        out.parts.push_back(MessagePart{MessagePartKind::Text,
                                                        part.value("text", std::string{}), {}});
                    } else if (type == "image_url" && part.contains("image_url")) {
                        const std::string url =
                            part.at("image_url").value("url", std::string{});
                        OwnedMedia media;
                        media.kind = MediaKind::Image;
                        if (!decode_image(url, &media.media_type, &media.bytes)) {
                            response.status = 400;
                            response.set_content(
                                error_body("unsupported image payload; send a base64 data URL",
                                           "invalid_request_error"),
                                "application/json");
                            return;
                        }
                        out.parts.push_back(MessagePart{MessagePartKind::Media, {}, std::move(media)});
                    }
                }
            }
            input.messages.push_back(std::move(out));
        }

        if (input.messages.empty()) {
            response.status = 400;
            response.set_content(error_body("messages must not be empty", "invalid_request_error"),
                                 "application/json");
            return;
        }

        ninfer::RequestOptions req_opts;
        const std::uint32_t max_tokens =
            body.contains("max_tokens") ? body.value("max_tokens", 0U) : body.value("max_completion_tokens", 0U);
        req_opts.execution.requested_output_tokens = max_tokens;
    req_opts.execution.allow_prefix_reuse = false;
        if (body.contains("temperature") && !body.at("temperature").is_null()) {
            req_opts.execution.sampling.temperature = body.value("temperature", 0.0F);
        }
        if (body.contains("top_p") && !body.at("top_p").is_null()) {
            req_opts.execution.sampling.top_p = body.value("top_p", 1.0F);
        }
        if (body.contains("seed") && !body.at("seed").is_null()) {
            req_opts.execution.sampling.seed =
                static_cast<std::uint64_t>(body.value("seed", 0));
        }
        if (body.contains("stop")) {
            const auto& stop = body.at("stop");
            if (stop.is_string()) {
                req_opts.stop.strings.push_back(StopString{stop.get<std::string>()});
            } else if (stop.is_array()) {
                for (const auto& entry : stop) {
                    if (entry.is_string()) {
                        req_opts.stop.strings.push_back(StopString{entry.get<std::string>()});
                    }
                }
            }
        }

        const std::string request_id = "chatcmpl-local";
        const bool streaming         = body.value("stream", false);

        if (streaming) {
            auto queue = std::make_shared<ChunkQueue>();
            auto sink  = std::make_shared<SseSink>(queue, request_id, model_name);
            std::thread worker([&engine, &model_name, queue, sink, opts = req_opts,
                                in = std::move(input), id = request_id]() mutable {
                try {
                    std::lock_guard<std::mutex> guard(g_engine_mutex);
                    PreparedPrompt prepared = engine->prepare(std::move(in));
                    const GenerationResult result =
                        engine->generate(std::move(prepared), opts, sink.get());
                    json tail;
                    tail["id"]      = id;
                    tail["object"]  = "chat.completion.chunk";
                    tail["created"] = 0;
                    tail["model"]   = model_name;
                    tail["choices"] = json::array({{{"index", 0},
                                                    {"delta", json::object()},
                                                    {"finish_reason",
                                                     finish_reason_name(result.finish_reason)}}});
                    queue->push("data: " + tail.dump() + "\n\n");
                } catch (const std::exception& error) {
                    json failure;
                    failure["error"] = json{{"message", error.what()}, {"type", "internal_error"}};
                    queue->push("data: " + failure.dump() + "\n\n");
                }
                queue->push("data: [DONE]\n\n");
                queue->close();
            });
            worker.detach();

            response.set_chunked_content_provider(
                "text/event-stream", [queue](std::size_t, httplib::DataSink& data_sink) {
                    std::string chunk;
                    if (!queue->pop(&chunk)) { return false; }
                    return data_sink.write(chunk.data(), chunk.size());
                });
            response.set_header("Cache-Control", "no-cache");
            response.set_header("X-Accel-Buffering", "no");
            return;
        }

        try {
            std::lock_guard<std::mutex> guard(g_engine_mutex);
            PreparedPrompt prepared = engine->prepare(std::move(input));
            const GenerationResult result = engine->generate(std::move(prepared), req_opts);

            if (streaming) {
                // One delta carrying the finished text: OpenAI clients accept it, and per-token
                // streaming needs a sink that is not worth wiring before the API is validated.
                std::string out;
                auto chunk = [&](const std::string& text, const char* finish, bool first) {
                    json delta;
                    delta["id"]      = request_id;
                    delta["object"]  = "chat.completion.chunk";
                    delta["created"] = 0;
                    delta["model"]   = model_name;
                    delta["choices"] = json::array({{{"index", 0},
                                                     {"delta", first ? json{{"role", "assistant"},
                                                                            {"content", text}}
                                                                      : json{{"content", text}}},
                                                     {"finish_reason", finish}}});
                    out += "data: " + delta.dump() + "\n\n";
                };
                chunk(result.content, nullptr, true);
                chunk("", finish_reason_name(result.finish_reason).c_str(), false);
                out += "data: [DONE]\n\n";
                response.set_content(out, "text/event-stream");
                response.set_header("Cache-Control", "no-cache");
                return;
            }

            json answer;
            answer["id"]      = request_id;
            answer["object"]  = "chat.completion";
            answer["created"] = 0;
            answer["model"]   = model_name;
            answer["choices"] = json::array({{{"index", 0},
                                              {"message", json{{"role", "assistant"},
                                                               {"content", result.content}}},
                                              {"finish_reason", finish_reason_name(result.finish_reason)}}});
            answer["usage"] = json{{"prompt_tokens", result.prompt.prompt_tokens},
                                   {"completion_tokens", result.generated_token_ids.size()},
                                   {"total_tokens", result.prompt.prompt_tokens +
                                                        result.generated_token_ids.size()}};
            response.set_content(answer.dump(), "application/json");
        } catch (const std::exception& error) {
            response.status = 500;
            response.set_content(error_body(error.what(), "internal_error"), "application/json");
        }
    });

    std::fprintf(stderr, "[serve] listening on http://%s:%d\n", host.c_str(), port);
    if (!server.listen(host.c_str(), port)) {
        std::fprintf(stderr, "[serve] failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
