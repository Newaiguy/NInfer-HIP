// NInfer->HIP Phase 1.1: minimal engine entry (bypasses product/ + spdlog deps).
// Usage: ninfer_mini.exe <artifact> [prompt] [--ctx N] [--max-new N] [--kv bf16|fp8|nvfp4|k8v4]
//                        [--temp F] [--top-p F] [--top-k N] [--think] [--seed N]
//                        [--spec mtp] [--draft-tokens N] [--lm-head-draft]
//                        [--tasks FILE]  (###ID id / prompt lines / ###END blocks, eval batch)
#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

class Sink final : public OutputSink {
public:
    void start(GenerationStart) override {}
    void progress(PromptProgress) override {}
    void timing(GenerationTimingObservation) override {}
    void publish(OutputDelta delta) override {
        if (delta.channel == OutputChannel::Content) {
            std::cout << delta.text;
            std::cout.flush();
        }
    }
};

// ---- interactive chat mode -------------------------------------------------
// Streaming sink for the REPL: prints content immediately, prints reasoning in
// dim grey when enabled, and accumulates both so the turn can be appended to the
// conversation history.
class ChatSink final : public OutputSink {
public:
    std::string content;
    std::string reasoning;
    bool show_reasoning = false;
    bool color          = false;

    void start(GenerationStart) override {}
    void progress(PromptProgress) override {}
    void timing(GenerationTimingObservation) override {}

    void publish(OutputDelta delta) override {
        if (delta.channel == OutputChannel::Reasoning) {
            reasoning += delta.text;
            if (!show_reasoning) { return; }
            if (!reasoning_open_) {
                std::cout << (color ? "\x1b[90m[thinking] " : "[thinking] ");
                reasoning_open_ = true;
            }
            std::cout << delta.text;
            std::cout.flush();
            return;
        }
        if (reasoning_open_) {
            std::cout << (color ? "\x1b[0m\n" : "\n");
            reasoning_open_ = false;
        }
        content += delta.text;
        std::cout << delta.text;
        std::cout.flush();
    }

private:
    bool reasoning_open_ = false;
};

#ifdef _WIN32
    #include <windows.h>
static void console_utf8(bool enable_color) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (out != INVALID_HANDLE_VALUE && GetConsoleMode(out, &mode)) {
        SetConsoleMode(out, mode | (enable_color ? ENABLE_VIRTUAL_TERMINAL_PROCESSING : 0));
    }
}
static bool stdout_is_console() {
    return GetConsoleWindow() != nullptr;
}
#else
static void console_utf8(bool) {}
static bool stdout_is_console() { return false; }
#endif

ChatMessage make_message(ChatRole role, const std::string& text) {
    ChatMessage msg;
    msg.role = role;
    msg.parts.push_back(MessagePart{MessagePartKind::Text, text, {}});
    return msg;
}

std::string trim_ascii(const std::string& s) {
    std::size_t b = 0;
    std::size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) { ++b; }
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) { --e; }
    return s.substr(b, e - b);
}

struct Task {
    std::string id;
    std::string prompt;
};

// Task file format (line-based, encoding-safe without a JSON dependency):
//   ###ID <id>\n <prompt lines...> \n###END\n   repeated; prompt runs until ###END.
std::vector<Task> load_tasks(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open tasks file: " + path);
    std::vector<Task> tasks;
    std::string line;
    bool in_prompt = false;
    Task current;
    std::ostringstream body;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r')) line.pop_back();
        if (line.rfind("###ID ", 0) == 0) {
            if (in_prompt) tasks.push_back(current);
            current = Task{};
            current.id = line.substr(6);
            body.str("");
            body.clear();
            in_prompt = true;
        } else if (line == "###END") {
            if (in_prompt) {
                current.prompt = body.str();
                tasks.push_back(current);
            }
            in_prompt = false;
        } else if (in_prompt) {
            body << line << '\n';
        }
    }
    if (in_prompt) {
        current.prompt = body.str();
        tasks.push_back(current);
    }
    return tasks;
}

#include <chrono>

// AMD port: raw device transfer bandwidth, so a slow run can be attributed to the card itself
// (clock/power latch) instead of the engine's access pattern.
static void* g_bw_buf = nullptr;
static const std::size_t k_bw_bytes = 64ull << 20;

static void bw(const char* tag) {
    if (g_bw_buf == nullptr) { return; }
    const auto once = [](void* buf) {
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::steady_clock::now();
        const cudaError_t e = cudaMemcpy(buf, static_cast<char*>(buf) + (k_bw_bytes >> 1),
                                         k_bw_bytes >> 1, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();
        if (e != cudaSuccess) { return -1.0; }
        const double sec = std::chrono::duration<double>(t1 - t0).count();
        return sec > 0.0 ? static_cast<double>(k_bw_bytes) / sec / 1e9 : 0.0;
    };
    once(g_bw_buf);
    double best = 0.0;
    for (int i = 0; i < 4; ++i) {
        const double g = once(g_bw_buf);
        if (g > best) { best = g; }
    }
    std::fprintf(stderr, "[xfer] %s d2d_GBs=%.1f\n", tag, best);
    std::fflush(stderr);
}

// AMD port: actual device VRAM before/after each phase. total is fixed; free tells whether the
// driver is paging the KV pool out of VRAM.
static void vram(const char* tag) {
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    const cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[vram] %s unavailable (%d)\n", tag, static_cast<int>(err));
        return;
    }
    std::fprintf(stderr, "[vram] %s free=%zu total=%zu used=%zu\n", tag, free_b, total_b,
                 total_b - free_b);
    std::fflush(stderr);
}

// AMD port: alloc-until-fail. Answers "how much can this process actually hold?" without guessing
// at the WDDM budget: allocate 128MB chunks (memset 1MB of each so the driver backs them) until the
// runtime refuses. NINFER_VRAM_CEILING=pre|post|both gates it.
static void vram_ceiling(const char* tag) {
    const std::size_t step = 128ULL << 20;
    std::vector<void*> held;
    std::size_t total = 0;
    const std::size_t limit = 24ULL << 30;
    while (total < limit) {
        void* p = nullptr;
        if (cudaMalloc(&p, step) != cudaSuccess) {
            (void) cudaGetLastError();
            break;
        }
        static const bool touch_all = std::getenv("NINFER_VRAM_TOUCH") != nullptr;
        (void) cudaMemset(p, 0, touch_all ? step : (1u << 20));
        held.push_back(p);
        total += step;
    }
    std::fprintf(stderr, "[ceiling] %s allocatable=%zu bytes (%.2f GiB) chunks=%zu\n", tag, total,
                 static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0), held.size());
    std::fflush(stderr);
    vram(tag);
    for (void* p : held) { (void) cudaFree(p); }
}

static bool ceiling_wanted(const char* which) {
    const char* v = std::getenv("NINFER_VRAM_CEILING");
    if (v == nullptr) { return false; }
    const std::string s(v);
    return s == "both" || s == which;
}

void run_one(Engine& engine, const Task& task, const RequestOptions& request, bool think,
             bool stream) {
    PromptInput input;
    ChatMessage msg;
    msg.role    = ChatRole::User;
    msg.parts.push_back(MessagePart{MessagePartKind::Text, task.prompt, {}});
    input.messages.push_back(std::move(msg));
    input.options.enable_thinking = think;

    vram("before-gen");
    bw("before-gen");
    PreparedPrompt prepared = engine.prepare(std::move(input));
    GenerationResult result = engine.generate(std::move(prepared), request);
    bw("after-gen");
    vram("after-gen");
    const std::size_t decoded = result.generated_token_ids.empty()
                                    ? 0 : result.generated_token_ids.size() - 1;
    const double prefill_tps = result.timings.prefill_seconds > 0.0
                                   ? result.prompt.prompt_tokens / result.timings.prefill_seconds
                                   : 0.0;
    const double decode_tps  = result.timings.decode_seconds > 0.0
                                   ? decoded / result.timings.decode_seconds : 0.0;
    std::cout << "===TASK " << task.id << "===\n" << result.content << "\n";
    std::cout << "===IDS " << task.id << "===";
    for (const TokenId id : result.generated_token_ids) { std::cout << " " << id; }
    std::cout << "\n";
    if (!result.reasoning.empty()) {
        std::cout << "===REASONING " << task.id << "===\n" << result.reasoning << "\n";
    }
    std::cout << "===END " << task.id << "=== finish=" << static_cast<int>(result.finish_reason)
              << " prompt=" << result.prompt.prompt_tokens << " gen=" << decoded
              << " prefill=" << prefill_tps << " decode=" << decode_tps << "\n";
    std::cout.flush();
    (void) stream;
}

// Interactive REPL. Each turn resubmits the whole conversation so the engine's
// resident prefix reuse can skip the previously computed KV instead of re-prefilling.
int run_chat(Engine& engine, const RequestOptions& request, bool think,
             const std::string& system_prompt, std::uint32_t ctx_capacity,
             std::uint32_t think_budget) {
    const bool tty   = stdout_is_console();
    const bool color = tty;
    console_utf8(color);

    std::vector<ChatMessage> history;
    if (!system_prompt.empty()) { history.push_back(make_message(ChatRole::System, system_prompt)); }
    const std::size_t history_base = history.size();
    std::size_t turns = 0;
    double last_decode = 0.0;

    auto banner = [&]() {
        std::cout << (color ? "\x1b[1m" : "") << "NInfer-HIP chat" << (color ? "\x1b[0m" : "")
                  << "  ctx=" << ctx_capacity << "  thinking=" << (think ? "on" : "off");
        if (think && think_budget > 0) { std::cout << "  think-budget=" << think_budget; }
        std::cout << "\n";
        std::cout << "commands: /help /clear /stats /think on|off /system <text> /exit\n";
        std::cout.flush();
    };
    banner();

    std::string line;
    while (true) {
        std::cout << (color ? "\x1b[36m" : "") << "you > " << (color ? "\x1b[0m" : "");
        std::cout.flush();
        if (!std::getline(std::cin, line)) { std::cout << "\n"; break; }
        const std::string text = trim_ascii(line);
        if (text.empty()) { continue; }

        if (text[0] == '/') {
            if (text == "/exit" || text == "/quit") { break; }
            if (text == "/clear") {
                if (history.size() > history_base) { history.resize(history_base); }
                turns = 0;
                std::cout << "[history cleared]\n";
                continue;
            }
            if (text == "/stats") {
                std::cout << "[stats] turns=" << turns << " history_messages=" << history.size()
                          << " last_decode_tps=" << last_decode << "\n";
                continue;
            }
            if (text == "/help") {
                std::cout << "/clear 清空对话  /stats 统计  /think on|off 思考开关  "
                             "/system <text> 设置系统提示  /exit 退出\n";
                continue;
            }
            if (text.rfind("/think", 0) == 0) {
                const std::string arg = trim_ascii(text.substr(6));
                if (arg == "on") { think = true; } else if (arg == "off") { think = false; }
                std::cout << "[thinking " << (think ? "on" : "off") << "]\n";
                continue;
            }
            if (text.rfind("/system", 0) == 0) {
                const std::string arg = trim_ascii(text.substr(7));
                if (history.size() > history_base) { history.resize(history_base); }
                if (!arg.empty()) { history.push_back(make_message(ChatRole::System, arg)); }
                std::cout << "[system prompt updated, history cleared]\n";
                continue;
            }
            std::cout << "[unknown command, try /help]\n";
            continue;
        }

        history.push_back(make_message(ChatRole::User, text));

        bool done       = false;
        bool trimmed    = false;
        while (!done) {
            PromptInput input;
            input.messages                = history;
            input.options.enable_thinking = think;

            ChatSink sink;
            sink.show_reasoning = think;
            sink.color          = color;

            // AMD port: thinking turns are capped so a self-verifying reasoning stream cannot
            // eat the whole --max-new budget and leave nothing for the answer.
            RequestOptions turn_request = request;
            if (think && think_budget > 0 &&
                turn_request.execution.requested_output_tokens > think_budget) {
                turn_request.execution.requested_output_tokens = think_budget;
            }
            PreparedPrompt prepared = engine.prepare(std::move(input));
            GenerationResult result;
            try {
                result = engine.generate(std::move(prepared), turn_request, &sink);
            } catch (const RequestError& error) {
                if (error.kind() == RequestErrorKind::ContextLengthExceeded && history.size() > history_base + 2 && !trimmed) {
                    history.erase(history.begin() + static_cast<std::ptrdiff_t>(history_base),
                                  history.begin() + static_cast<std::ptrdiff_t>(history_base + 2));
                    trimmed = true;
                    std::cout << "\n[context full, dropped the oldest turn and retried]\n";
                    continue;
                }
                std::cout << "\n[error] " << error.what() << "\n";
                if (history.size() > history_base) { history.pop_back(); }
                break;
            }
            done = true;

            const std::size_t generated = result.generated_token_ids.empty()
                                              ? 0 : result.generated_token_ids.size() - 1;
            const double prefill_tps = result.timings.prefill_seconds > 0.0
                                           ? result.prompt.prompt_tokens / result.timings.prefill_seconds : 0.0;
            const double decode_tps  = result.timings.decode_seconds > 0.0
                                           ? generated / result.timings.decode_seconds : 0.0;
            last_decode = decode_tps;

            std::cout << "\n";
            std::cout << (color ? "\x1b[32m" : "") << "[" << result.prompt.prompt_tokens << " prompt tokens ("
                      << result.reused_prompt_tokens << " reused), " << generated << " new, prefill "
                      << static_cast<long>(prefill_tps + 0.5) << " t/s, decode "
                      << (static_cast<long>(decode_tps * 10.0 + 0.5) / 10.0) << " t/s]"
                      << (color ? "\x1b[0m" : "") << "\n";
            std::cout.flush();

            // AMD port: the chat budget used to be small enough that a thinking model spent all of
            // it inside the reasoning stream and printed nothing for the answer, which looks like
            // a broken stream.
            // Say so explicitly instead of ending silently.
            const bool force_close = think && think_budget > 0 &&
                                    result.finish_reason == FinishReason::OutputLimit &&
                                    sink.content.empty() && !sink.reasoning.empty();
            if (result.finish_reason == FinishReason::OutputLimit && !force_close) {
                std::cout << (color ? "\x1b[33m" : "")
                          << "[note] stopped at the --max-new budget ("
                          << turn_request.execution.requested_output_tokens
                          << " tokens): the answer was cut off";
                if (sink.content.empty()) { std::cout << " - thinking used up the whole budget"; }
                std::cout << ". raise --max-new for long answers."
                          << (color ? "\x1b[0m" : "") << "\n";
                std::cout.flush();
            }

            // AMD port: a 3-bit thinking model can self-verify forever and never emit the
            // close marker, so the answer never appears (the poem request spent 4095 tokens
            // re-checking tone patterns). Force the close instead of ending silently: replay
            // the reasoning as text the assistant already produced, append the canonical close
            // marker, and resume generation with thinking disabled.
            if (force_close) {
                std::string carried = sink.reasoning;
                while (!carried.empty() &&
                       (carried.back() == '\n' || carried.back() == ' ' || carried.back() == '\r')) {
                    carried.pop_back();
                }
                // AMD port: cutting the reasoning mid-sentence makes the resumed stream look
                // like the answer started inside the reasoning (the model finishes the broken
                // clause first). Cut back to the last complete sentence, then append the
                // canonical close marker so the model starts a fresh answer block.
                {
                    std::size_t best    = std::string::npos;
                    std::size_t best_ln = 0;
                    const char* terms[] = {"\u3002", "\uff01", "\uff1f", "\uff1b", "\n", ".", "!", "?"};
                    for (const char* term : terms) {
                        const std::string needle(term);
                        const std::size_t hit = carried.rfind(needle);
                        if (hit != std::string::npos && (best == std::string::npos || hit > best)) {
                            best    = hit;
                            best_ln = needle.size();
                        }
                    }
                    if (best != std::string::npos && best + best_ln < carried.size()) {
                        carried.resize(best + best_ln);
                    }
                }
                // AMD port: closing the stream alone did not switch the model out of its
                // self-verification state (it kept checking tone patterns in the answer), so
                // hand it an explicit cue in its own voice.
                carried += "\n\n\n\n";
                carried += "好的，分析到此为止，现在直接给出最终答案：\n";
                std::cout << (color ? "\x1b[33m" : "") << "[note] reasoning budget ("
                          << think_budget << " tokens) reached, forcing the answer"
                          << (color ? "\x1b[0m" : "") << "\n";
                std::cout.flush();
                PromptInput follow;
                follow.messages = history;
                ChatMessage carry;
                carry.role = ChatRole::Assistant;
                carry.parts.push_back(MessagePart{MessagePartKind::Text, carried, {}});
                follow.messages.push_back(std::move(carry));
                follow.options.enable_thinking = false;
                // AMD port: continuing the same assistant message made the model keep analysing
                // (or stop immediately). A fresh, explicit user turn is the strongest signal to
                // switch from reasoning to answering.
                follow.messages.push_back(make_message(ChatRole::User,
                    "（继续：请直接给出最终答案，不要重复分析过程。）"));
                ChatSink sink2  = sink;
                sink2.show_reasoning = false;
                sink2.content.clear();
                try {
                    PreparedPrompt prepared2 = engine.prepare(std::move(follow));
                    GenerationResult tail = engine.generate(std::move(prepared2), request, &sink2);
                    sink.content = sink2.content;
                    const std::size_t tail_generated = tail.generated_token_ids.empty()
                                                          ? 0 : tail.generated_token_ids.size() - 1;
                    const double tail_tps = tail.timings.decode_seconds > 0.0
                                                ? tail_generated / tail.timings.decode_seconds : 0.0;
                    std::cout << (color ? "\x1b[32m" : "") << "[" << tail_generated
                              << " answer tokens, decode " << (static_cast<long>(tail_tps * 10.0 + 0.5) / 10.0)
                              << " t/s]" << (color ? "\x1b[0m" : "") << "\n";
                    std::cout.flush();
                    if (sink.content.empty()) {
                        std::cout << (color ? "\x1b[33m" : "")
                                  << "[note] the model still produced no answer after the forced close"
                                  << (color ? "\x1b[0m" : "") << "\n";
                    }
                } catch (const RequestError& error) {
                    std::cout << "\n[error] forced close failed: " << error.what() << "\n";
                }
            }

            ChatMessage assistant;
            assistant.role = ChatRole::Assistant;
            assistant.parts.push_back(MessagePart{MessagePartKind::Text, sink.content, {}});
            assistant.reasoning_content = sink.reasoning;
            history.push_back(std::move(assistant));
            turns += 1;

            if (result.finish_reason == FinishReason::ContextCapacity) {
                std::cout << "[note] hit context capacity; /clear or shorter prompts\n";
            }
        }
    }
    std::cout << "bye\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (std::getenv("NINFER_ARGV") != nullptr) {
        for (int i = 0; i < argc; ++i) { std::fprintf(stderr, "[argv] %d=%s\n", i, argv[i]); }
        std::fflush(stderr);
    }
    if (argc < 2) {
        std::cerr << "usage: ninfer_mini.exe <artifact> [prompt] [--ctx N] [--max-new N] "
                     "[--kv bf16|fp8|nvfp4|k8v4] [--no-graph] [--temp F] [--top-p F] [--top-k N] "
                     "[--think] [--think-budget N] [--seed N] [--tasks FILE]\n";
        return 1;
    }

    // Only run the transfer probe when asked; it costs a 64MB allocation and is noise in chat mode.
    if (std::getenv("NINFER_BW") != nullptr) {
        if (cudaMalloc(&g_bw_buf, k_bw_bytes) == cudaSuccess) {
            cudaMemset(g_bw_buf, 0, k_bw_bytes);
        } else {
            g_bw_buf = nullptr;
        }
        bw("startup");
    }

    // AMD port: diagnostic pre-allocation. Shifting every engine allocation keeps the layout but
    // resamples the physical page assignment, which separates "capacity is expensive" from
    // "this particular placement is unlucky".
    static void* g_prealloc = nullptr;
    if (const char* pmb = std::getenv("NINFER_PREALLOC_MB")) {
        const std::size_t bytes = static_cast<std::size_t>(std::atoll(pmb)) * 1024ULL * 1024ULL;
        if (bytes > 0) {
            if (cudaMalloc(&g_prealloc, bytes) == cudaSuccess) {
                cudaMemset(g_prealloc, 0, bytes);
                std::fprintf(stderr, "[prealloc] held %zu bytes at %p\n", bytes, g_prealloc);
            } else {
                std::fprintf(stderr, "[prealloc] cudaMalloc(%zu) FAILED\n", bytes);
            }
            std::fflush(stderr);
        }
    }

    EngineOptions options;
    options.artifact_path = argv[1];
    bool chunk_explicit = false;
    if (const char* pc = std::getenv("NINFER_PREFILL_CHUNK")) {
        const long v = std::strtol(pc, nullptr, 10);
        if (v > 0) {
            options.prefill_chunk = static_cast<std::uint32_t>(v);
            chunk_explicit        = true;
        }
    }
    std::string prompt    = "Hello!";
    std::string ids_file;
    std::string tasks_file;
    std::uint32_t max_new = 64;
    float temp            = 0.0F;
    float top_p           = 1.0F;
    std::int32_t top_k    = 0;
    float presence_penalty    = 0.0F;
    bool presence_penalty_set = false;
    bool think            = false;
    std::uint32_t think_budget = 2048;
    bool have_seed        = false;
    std::uint64_t seed    = 0;
    bool chat_mode        = false;
    std::string system_prompt;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next     = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--ctx") {
            const std::uint32_t ctx =
                static_cast<std::uint32_t>(std::stoul(next()));
            options.max_context = ctx;
            options.kv_capacity  = KvCapacityPolicy::explicit_capacity(ctx);
        }
        else if (a == "--kvcap") {
            // Diagnostic: keep --ctx for the attention envelope but allocate a smaller KV pool
            // so pool size can be varied independently of the visible-key limit.
            options.kv_capacity =
                KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(std::stoul(next())));
        }
        else if (a == "--max-new") { max_new = static_cast<std::uint32_t>(std::stoul(next())); }
        else if (a == "--kv") {
            std::string kv = next();
            if (kv == "bf16") options.kv_cache = KvCacheStorage::BFloat16;
            else if (kv == "int8") options.kv_cache = KvCacheStorage::Int8Group64;
            else if (kv == "fp8") options.kv_cache = KvCacheStorage::Fp8E4M3Row256;
            else if (kv == "nvfp4") options.kv_cache = KvCacheStorage::Nvfp4Group16;
            else if (kv == "k8v4") options.kv_cache = KvCacheStorage::Fp8KeyNvfp4Value;
            else if (kv == "rk4v4-e8") options.kv_cache = KvCacheStorage::RotatedInt4KeyInt4ValueE8;
            else if (kv == "q4_0") options.kv_cache = KvCacheStorage::Q4_0;
        }
        else if (a == "--resident-tokens" || a == "--sink-tokens" ||
                 a == "--archive-mib" || a == "--kv-retrieval") {
            // Window/archival knobs are read straight from the environment deep inside the memory
            // planner, so the flags publish them there instead of threading options through.
            const std::string v    = next();
            const char* env_name   = a == "--resident-tokens" ? "NINFER_RESIDENT_TOKENS"
                                   : a == "--sink-tokens"     ? "NINFER_SINK_TOKENS"
                                   : a == "--archive-mib"     ? "NINFER_KV_ARCHIVE_MIB"
                                                              : "NINFER_KV_RETRIEVE";
            _putenv_s(env_name, v.c_str());
            if (a == "--archive-mib") { _putenv_s("NINFER_KV_ARCHIVE", "1"); }
        }
        else if (a == "--no-graph") { options.use_cuda_graph = false; }
        else if (a == "--spec") {
            const std::string backend = next();
            if (backend == "mtp") {
                options.speculative.backend = SpeculativeBackend::Mtp;
            } else if (backend == "dflash") {
                options.speculative.backend = SpeculativeBackend::DFlash;
            } else if (backend == "dflash2") {
                options.speculative.backend = SpeculativeBackend::DFlash2;
            } else {
                throw std::runtime_error("invalid --spec backend: " + backend);
            }
        }
        else if (a == "--draft-tokens") {
            options.speculative.draft_tokens =
                static_cast<std::uint32_t>(std::stoul(next()));
        }
        else if (a == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        }
        else if (a == "--presence-penalty") {
            presence_penalty     = std::stof(next());
            presence_penalty_set = true;
        }
        else if (a == "--chat") { chat_mode = true; }
        else if (a == "--system") { system_prompt = next(); }
        else if (a == "--ids") { ids_file = next(); }
        else if (a == "--tasks") { tasks_file = next(); }
        else if (a == "--temp") { temp = std::stof(next()); }
        else if (a == "--top-p") { top_p = std::stof(next()); }
        else if (a == "--top-k") { top_k = std::stoi(next()); }
        else if (a == "--think") { think = true; }
        else if (a == "--think-budget") { think_budget = static_cast<std::uint32_t>(std::stoul(next())); }
        else if (a == "--seed") { seed = std::stoull(next()); have_seed = true; }
        else { prompt = a; }
    }

    // AMD port: auto prefill chunk. 2048 measures +7.1% prefill at 32K and +4.1% at 65K with an
    // unchanged decode, but it costs +232 MB of committed memory (workspace 240 -> 462 MB), which
    // eats into the ~13.5 GB high-speed residency budget and would push the 69K..131K cliff down by
    // roughly 13K tokens of context. So keep 2048 below the cliff and stay at 1024 above it.
    if (!chunk_explicit) {
        options.prefill_chunk = (options.max_context <= 65536U) ? 2048U : 1024U;
    }

    // AMD port: keep the chat console clean. In chat mode all engine diagnostics are redirected
    // to ninfer-chat.log instead of interleaving with the streamed answer
    // (set NINFER_CHAT_LOG=0 to keep them on stderr, e.g. when debugging a crash).
    if (chat_mode) {
        const char* cl = std::getenv("NINFER_CHAT_LOG");
        if (cl == nullptr || cl[0] != '0') {
            if (std::freopen("ninfer-chat.log", "w", stderr) == nullptr) {
                std::fprintf(stdout, "[i] warning: could not open ninfer-chat.log\n");
            }
            std::fprintf(stdout, "[i] engine diagnostics -> ninfer-chat.log\n");
            std::fflush(stdout);
        }
    }

    try {
        std::cerr << "[mini] before-engine\n";
        std::cerr.flush();
        vram("before-engine");
        if (ceiling_wanted("pre")) { vram_ceiling("pre-engine"); }
        Engine engine(std::move(options));
        std::cerr << "[mini] engine-ready\n";
        vram("after-engine");
        if (ceiling_wanted("post")) { vram_ceiling("post-engine"); }
        {
            // AMD port: VRAM budget dump (weights / sequence / workspace / reservation).
            const MemorySummary ms = engine.memory_summary();
            std::fprintf(stderr,
                         "[mem] max_ctx=%u kv_cap=%u kv_payload=%zu weights=%zu/%zu seq=%zu/%zu "
                         "ws=%zu/%zu runtime_res=%zu min_res=%zu graph=%zu slack=%zu "
                         "avail_after_weights=%zu avail_after_startup=%zu\n",
                         ms.max_context, ms.kv_capacity, ms.kv_payload_bytes,
                         ms.weights.used_bytes, ms.weights.capacity_bytes,
                         ms.sequence.used_bytes, ms.sequence.capacity_bytes,
                         ms.workspace.used_bytes, ms.workspace.capacity_bytes,
                         ms.runtime_reservation_bytes, ms.minimum_runtime_reservation_bytes,
                         ms.cuda_graph_allowance_bytes, ms.planned_slack_bytes,
                         ms.available_after_weights_bytes, ms.available_after_startup_bytes);
            std::fflush(stderr);
        }
        std::cerr.flush();
        engine.reset_memory_peaks();

        // AMD port: diagnostic dummy commit. Holding extra VRAM without ever touching it in the
        // hot path separates "WDDM residency/demotion pressure" from "KV pool address span".
        void* dummy_commit = nullptr;
        if (const char* dmb = std::getenv("NINFER_DUMMY_MB")) {
            const std::size_t bytes = static_cast<std::size_t>(std::atoll(dmb)) * 1024ULL * 1024ULL;
            if (bytes > 0) {
                if (cudaMalloc(&dummy_commit, bytes) == cudaSuccess) {
                    cudaMemset(dummy_commit, 0, bytes);
                    std::fprintf(stderr, "[dummy] committed %zu bytes at %p\n", bytes, dummy_commit);
                    vram("after-dummy");
                } else {
                    std::fprintf(stderr, "[dummy] cudaMalloc(%zu) FAILED\n", bytes);
                }
                std::fflush(stderr);
            }
        }

        RequestOptions request;
        request.execution.requested_output_tokens = max_new;
        request.execution.sampling.temperature    = temp;
        request.execution.sampling.top_p          = top_p;
        request.execution.sampling.top_k          = top_k;
        if (presence_penalty_set) {
            request.execution.sampling.presence_penalty = presence_penalty;
        }
        if (have_seed) { request.execution.sampling.seed = seed; }

        if (chat_mode) {
            return run_chat(engine, request, think, system_prompt, options.max_context, think_budget);
        }

        if (!tasks_file.empty()) {
            const std::vector<Task> tasks = load_tasks(tasks_file);
            std::cerr << "tasks: " << tasks.size() << "\n";
            for (const Task& task : tasks) { run_one(engine, task, request, think, false); }
            return 0;
        }

        std::cerr << "[mini] before-prepare\n";
        std::cerr.flush();
        PreparedPrompt prepared;
        if (!ids_file.empty()) {
            // Raw token-id prompt for repeatable correctness and performance measurement.
            std::ifstream in(ids_file);
            std::vector<TokenId> ids;
            TokenId id;
            while (in >> id) ids.push_back(id);
            prepared = engine.prepare_tokens(std::move(ids), false);
        } else {
            PromptInput input;
            ChatMessage msg;
            msg.role    = ChatRole::User;
            msg.parts.push_back(MessagePart{MessagePartKind::Text, prompt, {}});
            input.messages.push_back(std::move(msg));
            input.options.enable_thinking = think;

            prepared = engine.prepare(std::move(input));
        }

        Sink sink;
        GenerationHandle generation = engine.submit(std::move(prepared), std::move(request),
                                                    OutputConsumerMode::Streaming);
        GenerationResult result     = generation.wait(&sink);

        const std::size_t decoded  = result.generated_token_ids.empty()
                                         ? 0 : result.generated_token_ids.size() - 1;
        const double prefill_tps   = result.timings.prefill_seconds > 0.0
                                         ? result.prompt.prompt_tokens / result.timings.prefill_seconds : 0.0;
        const double decode_tps    = result.timings.decode_seconds > 0.0
                                         ? decoded / result.timings.decode_seconds : 0.0;
        std::cerr << "\n--- finish=" << static_cast<int>(result.finish_reason)
                  << " prompt=" << result.prompt.prompt_tokens
                  << " generated=" << result.generated_token_ids.size()
                  << " prefill=" << prefill_tps << " tok/s"
                  << " decode=" << decode_tps << " tok/s"
                  << " total=" << result.timings.total_seconds << "s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
