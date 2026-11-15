#include "ops/linear/gguf/gguf_linear.h"

#include "core/layout.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::detail {
namespace {

// Rows the dequantize-and-GEMM route converts per pass for a type with no integer kernel.
constexpr std::size_t kDequantizedScratchBytes = std::size_t{16} << 20;
// Up to this many columns such a type loops the vector kernel instead: those are decode and verify
// batches, captured in graphs, where a cuBLAS call is best avoided.
constexpr std::int32_t kDequantizedVectorColumns = 64;
// The widest stream-k fixup plane any launch can ask for: one 128 x 128 FP32 tile per SM.
constexpr std::size_t kFixupTileBytes = 128 * 128 * sizeof(float);

std::int64_t row_bytes(const Weight& w) {
    const auto block = gguf_block_shape(w.qtype);
    return std::int64_t(w.k / block.elements) * block.bytes;
}

bool matrix_kernel(const Weight& w) { return gguf::has_matrix_kernel(gguf_type(w.qtype)); }

gguf::Epilogue bridge_epilogue(GgufEpilogue epilogue) {
    switch (epilogue) {
    case GgufEpilogue::Store: return gguf::Epilogue::Store;
    case GgufEpilogue::Accumulate: return gguf::Epilogue::Accumulate;
    case GgufEpilogue::GateProduct: return gguf::Epilogue::GateProduct;
    }
    throw std::invalid_argument("gguf: invalid epilogue");
}

// The destination of columns [first, ...) of a product.
gguf::VectorOutput output(const GgufProduct& p, std::int32_t first) {
    gguf::VectorOutput out;
    out.epilogue = bridge_epilogue(p.epilogue);
    if (p.f32 != nullptr) {
        out.f32           = p.f32 + std::int64_t(first) * p.f32_rows + p.row;
        out.column_stride = p.f32_rows;
    } else {
        out.bf16 = static_cast<__nv_bfloat16*>(p.out->data) + std::int64_t(first) * p.out->ne[0] +
                   p.row;
        out.column_stride = p.out->ne[0];
    }
    if (p.epilogue == GgufEpilogue::GateProduct) {
        out.gate               = p.gate + std::int64_t(first) * p.weight->n;
        out.gate_column_stride = p.weight->n;
    }
    return out;
}

void require_activation(const Tensor& x, const char* label) {
    if (x.dtype != DType::BF16 || x.ne[0] <= 0 || x.ne[1] <= 0 || x.ne[2] != 1 || x.ne[3] != 1 ||
        !x.is_contiguous() || x.data == nullptr) {
        throw std::invalid_argument(std::string(label) + ": x must be contiguous BF16 [K,T]");
    }
}

void require_product(const Tensor& x, const GgufProduct& p) {
    require_gguf(*p.weight, "gguf product");
    if (p.weight->k != x.ne[0]) { throw std::invalid_argument("gguf product: K differs from x"); }
    if (p.f32 != nullptr) {
        if (p.row < 0 || p.row + p.weight->n > p.f32_rows) {
            throw std::invalid_argument("gguf product: rows exceed the FP32 plane");
        }
        return;
    }
    if (p.out == nullptr || p.out->dtype != DType::BF16 || !p.out->is_contiguous() ||
        p.out->ne[1] != x.ne[1] || p.row < 0 || p.row + p.weight->n > p.out->ne[0]) {
        throw std::invalid_argument("gguf product: rows exceed the BF16 destination");
    }
    if (p.epilogue == GgufEpilogue::GateProduct && p.gate == nullptr) {
        throw std::invalid_argument("gguf product: the gate product needs its gate plane");
    }
}

std::size_t fixup_bound() {
    return std::size_t(gguf::multiprocessor_count()) * kFixupTileBytes;
}

} // namespace

gguf::GgmlType gguf_type(QType format) {
    switch (format) {
    case QType::GGUF_Q8_0: return gguf::GgmlType::Q8_0;
    case QType::GGUF_Q2_K: return gguf::GgmlType::Q2_K;
    case QType::GGUF_Q3_K: return gguf::GgmlType::Q3_K;
    case QType::GGUF_Q4_K: return gguf::GgmlType::Q4_K;
    case QType::GGUF_Q5_K: return gguf::GgmlType::Q5_K;
    case QType::GGUF_Q6_K: return gguf::GgmlType::Q6_K;
    case QType::GGUF_IQ2_XXS: return gguf::GgmlType::IQ2_XXS;
    case QType::GGUF_IQ2_XS: return gguf::GgmlType::IQ2_XS;
    case QType::GGUF_IQ2_S: return gguf::GgmlType::IQ2_S;
    case QType::GGUF_IQ3_XXS: return gguf::GgmlType::IQ3_XXS;
    case QType::GGUF_IQ3_S: return gguf::GgmlType::IQ3_S;
    case QType::GGUF_IQ1_S: return gguf::GgmlType::IQ1_S;
    case QType::GGUF_IQ1_M: return gguf::GgmlType::IQ1_M;
    case QType::GGUF_IQ4_NL: return gguf::GgmlType::IQ4_NL;
    case QType::GGUF_IQ4_XS: return gguf::GgmlType::IQ4_XS;
    default: break;
    }
    throw std::invalid_argument("gguf: not a GGUF block format");
}

void require_gguf(const Weight& w, const char* label) {
    const auto block = gguf_block_shape(w.qtype);
    if (!is_gguf(w.qtype) || w.layout != QuantLayout::GgufBlocks || w.qdata == nullptr ||
        w.n <= 0 || w.k <= 0 || w.k % block.elements != 0 || w.k % 512 != 0) {
        throw std::invalid_argument(std::string(label) +
                                    ": expected a resident GGUF matrix over 512-column groups");
    }
}

std::size_t gguf_project_workspace_bytes(std::span<const GgufShape> shapes,
                                         std::int32_t min_tokens, std::int32_t max_tokens) {
    if (shapes.empty() || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gguf workspace: invalid products or token interval");
    }
    const std::int32_t k = shapes.front().k;
    std::vector<int> layouts;
    std::size_t rows = 0;
    bool dequantized = false;
    for (const GgufShape& shape : shapes) {
        const auto block = gguf_block_shape(shape.qtype);
        if (!is_gguf(shape.qtype) || shape.rows <= 0 || shape.k != k || k <= 0 ||
            k % block.elements != 0 || k % 512 != 0) {
            throw std::invalid_argument("gguf workspace: unsupported product geometry");
        }
        const auto type = gguf_type(shape.qtype);
        if (gguf::has_matrix_kernel(type)) {
            const int layout = gguf::matrix_activation_layout(type);
            if (std::find(layouts.begin(), layouts.end(), layout) == layouts.end()) {
                layouts.push_back(layout);
            }
        } else {
            dequantized = true;
        }
        rows = std::max<std::size_t>(rows, shape.rows);
    }
    // Every product may carry its own input gather, so each may quantize its own activation.
    const std::size_t activations = shapes.size();
    std::size_t widest            = 0;
    if (min_tokens <= gguf::kMaxVectorColumns) {
        const std::int32_t t = std::min(max_tokens, gguf::kMaxVectorColumns);
        WorkspaceLayoutBuilder layout;
        for (std::size_t i = 0; i < activations; ++i) {
            (void)layout.alloc_bytes(gguf::vector_activation_bytes(k, t));
        }
        widest = layout.peak_bytes(1);
    }
    if (max_tokens > gguf::kMaxVectorColumns) {
        const std::int32_t t = max_tokens;
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc_bytes(rows * t * sizeof(float));
        (void)layout.alloc_bytes(fixup_bound());
        if (dequantized) {
            (void)layout.alloc_bytes(gguf::vector_activation_bytes(k, gguf::kMaxVectorColumns));
            (void)layout.alloc_bytes(kDequantizedScratchBytes);
        }
        for (std::size_t i = 0; i < std::max<std::size_t>(layouts.size(), 1) * activations; ++i) {
            (void)layout.alloc_bytes(gguf::matrix_activation_bytes(k, t));
        }
        if (dequantized) { (void)layout.alloc_bytes(std::size_t(k) * t * 2); }
        widest = std::max(widest, layout.peak_bytes(1));
    }
    return widest;
}

void gguf_project(const Tensor& x, std::span<const GgufProduct> products,
                  WorkspaceArena& workspace, cudaStream_t stream) {
    require_activation(x, "gguf product");
    if (products.empty()) { return; }
    for (const auto& p : products) { require_product(x, p); }
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const auto* xb       = static_cast<const __nv_bfloat16*>(x.data);
    auto scope           = workspace.scope();

    if (t <= gguf::kMaxVectorColumns) {
        struct Activation {
            const std::int32_t* columns;
            void* data;
        };
        std::vector<Activation> activations;
        // AMD DEBUG (q6_k A/B): Q6_K only appears in the v3 artifact's MTP head, so a defect in its
        // vector decoder is invisible everywhere else. NINFER_Q6K_DEQUANT routes those products
        // through the dequantize + BF16 GEMM path instead, giving a numeric reference for the same
        // weights without touching the rest of the model.
        const bool force_q6k_dequant = std::getenv("NINFER_Q6K_DEQUANT") != nullptr;
        void* q6k_scratch            = nullptr;
        if (force_q6k_dequant) {
            q6k_scratch = workspace.alloc_bytes(kDequantizedScratchBytes).data;
        }
        for (const auto& p : products) {
            const Weight& w = *p.weight;
            auto found      = std::find_if(activations.begin(), activations.end(),
                                           [&](const Activation& a) { return a.columns == p.input_columns; });
            if (found == activations.end()) {
                void* data = workspace.alloc_bytes(gguf::vector_activation_bytes(k, t)).data;
                gguf::quantize_vector_activation(xb, k, t, p.input_columns, data, stream);
                activations.push_back({p.input_columns, data});
                found = activations.end() - 1;
            }
            if (force_q6k_dequant && w.qtype == QType::GGUF_Q6_K && p.input_columns == nullptr) {
                auto* plane_t = static_cast<float*>(
                    workspace.alloc_bytes(std::size_t(w.n) * t * sizeof(float)).data);
                gguf::dequantized_product(gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, k, xb, t,
                                          plane_t, w.n, q6k_scratch, kDequantizedScratchBytes,
                                          stream);
                gguf::store_plane(plane_t, w.n, w.n, t, output(p, 0), stream);
            } else {
                gguf::vector_product(gguf_type(w.qtype), w.qdata, row_bytes(w), w.n, k, found->data,
                                     t, output(p, 0), stream);
            }
            if (std::getenv("NINFER_VECTOR_CHECK") != nullptr) {
                // A stream failure is only visible through Synchronize; hipGetLastError after a
                // kernel fault reads stale hipSuccess and the next launch then fails spuriously.
                const cudaError_t sync = cudaStreamSynchronize(stream);
                std::fprintf(stderr, "[vec] rows=%d qtype=%d t=%d sync=%d (%s)\n", w.n,
                             static_cast<int>(w.qtype), t, static_cast<int>(sync),
                             cudaGetErrorString(sync));
                std::fflush(stderr);
                CUDA_CHECK(sync);
            }
            if (std::getenv("NINFER_VECTOR_CHECK") != nullptr && w.qtype == QType::GGUF_IQ2_XXS) {
                cudaStreamSynchronize(stream);
                std::uint8_t qbuf[16] = {};
                CUDA_CHECK(cudaMemcpy(qbuf, w.qdata, 16, cudaMemcpyDeviceToHost));
                std::uint16_t obuf[8] = {};
                const auto* obp = static_cast<const __nv_bfloat16*>(output(p, 0).bf16);
                CUDA_CHECK(cudaMemcpy(obuf, obp, 16, cudaMemcpyDeviceToHost));
                std::fprintf(stderr,
                             "[vec-iq2xxs] w.n=%d w.k=%d rb=%lld qdata0=%02x,%02x,%02x,%02x,%02x,%02x,%02x,%02x "
                             "out0=%u,%u,%u,%u,%u,%u,%u,%u\n",
                             w.n, int(w.k), static_cast<long long>(row_bytes(w)), qbuf[0], qbuf[1],
                             qbuf[2], qbuf[3], qbuf[4], qbuf[5], qbuf[6], qbuf[7],
                             static_cast<unsigned>(obuf[0]), static_cast<unsigned>(obuf[1]),
                             static_cast<unsigned>(obuf[2]), static_cast<unsigned>(obuf[3]),
                             static_cast<unsigned>(obuf[4]), static_cast<unsigned>(obuf[5]),
                             static_cast<unsigned>(obuf[6]), static_cast<unsigned>(obuf[7]));
                std::fflush(stderr);
            }
            if (std::getenv("NINFER_GGUF_ACT_PROBE") != nullptr && p.out != nullptr) {
                cudaStreamSynchronize(stream);
                std::uint16_t out0[4] = {};
                CUDA_CHECK(cudaMemcpy(out0, p.out->data, sizeof(out0), cudaMemcpyDeviceToHost));
                std::fprintf(stderr,
                             "[act] rows=%d k=%d t=%d wtype=%d out0=%u,%u,%u,%u\n",
                             w.n, k, t, static_cast<int>(w.qtype),
                             static_cast<unsigned>(out0[0]), static_cast<unsigned>(out0[1]),
                             static_cast<unsigned>(out0[2]), static_cast<unsigned>(out0[3]));
                std::fflush(stderr);
            }
        }
        return;
    }

    struct Activation {
        int layout;
        const std::int32_t* columns;
        void* data;
    };
    std::vector<Activation> activations;
    std::int32_t plane_rows = 0;
    bool dequantized        = false;
    for (const auto& p : products) {
        plane_rows = std::max(plane_rows, p.weight->n);
        dequantized |= !matrix_kernel(*p.weight);
    }
    auto* plane = static_cast<float*>(
        workspace.alloc_bytes(std::size_t(plane_rows) * t * sizeof(float)).data);
    void* fixup = workspace.alloc_bytes(fixup_bound()).data;
    void* vector_activation = nullptr;
    void* scratch           = nullptr;
    if (dequantized) {
        vector_activation =
            workspace.alloc_bytes(gguf::vector_activation_bytes(k, gguf::kMaxVectorColumns)).data;
        scratch = workspace.alloc_bytes(kDequantizedScratchBytes).data;
    }

    for (const auto& p : products) {
        const Weight& w  = *p.weight;
        const auto type  = gguf_type(w.qtype);
        const bool direct = p.f32 != nullptr && p.epilogue == GgufEpilogue::Store;
        float* out        = direct ? p.f32 + p.row : plane;
        const std::int64_t stride = direct ? p.f32_rows : w.n;
        if (gguf::has_matrix_kernel(type)) {
            const int layout = gguf::matrix_activation_layout(type);
            auto found = std::find_if(activations.begin(), activations.end(), [&](const Activation& a) {
                return a.layout == layout && a.columns == p.input_columns;
            });
            if (found == activations.end()) {
                void* data = workspace.alloc_bytes(gguf::matrix_activation_bytes(k, t)).data;
                gguf::quantize_matrix_activation(type, xb, k, t, p.input_columns, data, stream);
                activations.push_back({layout, p.input_columns, data});
                found = activations.end() - 1;
            }
            gguf::matrix_product(type, w.qdata, row_bytes(w), w.n, k, found->data, t, out, stride,
                                 fixup, stream);
        } else if (t <= kDequantizedVectorColumns) {
            for (std::int32_t first = 0; first < t; first += gguf::kMaxVectorColumns) {
                const std::int32_t columns = std::min(gguf::kMaxVectorColumns, t - first);
                gguf::quantize_vector_activation(xb + std::int64_t(first) * k, k, columns,
                                                 p.input_columns, vector_activation, stream);
                gguf::vector_product(type, w.qdata, row_bytes(w), w.n, k, vector_activation,
                                     columns, output(p, first), stream);
            }
            continue;
        } else {
            const __nv_bfloat16* source = xb;
            if (p.input_columns != nullptr) {
                auto* gathered = static_cast<__nv_bfloat16*>(
                    workspace.alloc_bytes(std::size_t(k) * t * 2).data);
                gguf::gather_columns(xb, k, t, p.input_columns, gathered, stream);
                source = gathered;
            }
            gguf::dequantized_product(type, w.qdata, row_bytes(w), w.n, k, source, t, out, stride,
                                      scratch, kDequantizedScratchBytes, stream);
        }
        if (!direct) { gguf::store_plane(plane, w.n, w.n, t, output(p, 0), stream); }
    }
}

void gguf_linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    GgufProduct product{&w, &out, 0, GgufEpilogue::Store};
    product.input_columns = w.input_columns;
    gguf_project(x, {&product, 1}, workspace, stream);
}

void gguf_linear_add(const Tensor& x, const Weight& w, Tensor& residual,
                     WorkspaceArena& workspace, cudaStream_t stream) {
    GgufProduct product{&w, &residual, 0, GgufEpilogue::Accumulate};
    // GGUF matrices may store their input columns in a permuted (tiled value-head) order;
    // the activation must be gathered through Weight::input_columns before quantization.
    product.input_columns = w.input_columns;
    gguf_project(x, {&product, 1}, workspace, stream);
}

std::size_t gguf_swiglu_workspace_bytes(const GgufShape& gate, const GgufShape* up,
                                        std::int32_t min_tokens, std::int32_t max_tokens) {
    if (up == nullptr) {
        const GgufShape halves[] = {{gate.qtype, gate.rows / 2, gate.k},
                                    {gate.qtype, gate.rows / 2, gate.k}};
        return gguf_project_workspace_bytes(halves, min_tokens, max_tokens) +
               std::size_t(gate.rows / 2) * max_tokens * sizeof(float) + 256;
    }
    const GgufShape parents[] = {gate, *up};
    return gguf_project_workspace_bytes(parents, min_tokens, max_tokens) +
           std::size_t(gate.rows) * max_tokens * sizeof(float) + 256;
}

void gguf_swiglu(const Tensor& x, const Weight& gate, const Weight* up, Tensor& out,
                 WorkspaceArena& workspace, cudaStream_t stream) {
    require_activation(x, "gguf swiglu");
    require_gguf(gate, "gguf swiglu gate");
    const std::int32_t t    = x.ne[1];
    const std::int32_t rows = out.ne[0];
    if (out.dtype != DType::BF16 || !out.is_contiguous() || out.ne[1] != t) {
        throw std::invalid_argument("gguf swiglu: out must be contiguous BF16 [N,T]");
    }
    auto scope = workspace.scope();
    Weight gate_rows = gate;
    Weight up_rows;
    if (up == nullptr) {
        if (gate.n != 2 * rows) { throw std::invalid_argument("gguf swiglu: parent is not [gate; up]"); }
        if (t <= gguf::kMaxVectorColumns) {
            void* activation =
                workspace.alloc_bytes(gguf::vector_activation_bytes(gate.k, t)).data;
            gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.data), gate.k, t,
                                             nullptr, activation, stream);
            gguf::vector_swiglu(gguf_type(gate.qtype), gate.qdata, row_bytes(gate), rows, gate.k,
                                activation, t, static_cast<__nv_bfloat16*>(out.data), rows,
                                stream);
            return;
        }
        gate_rows.n = rows;
        up_rows     = gate;
        up_rows.n   = rows;
        up_rows.qdata = static_cast<const std::byte*>(gate.qdata) + std::int64_t(rows) * row_bytes(gate);
    } else {
        require_gguf(*up, "gguf swiglu up");
        if (gate.n != rows || up->n != rows || up->k != gate.k) {
            throw std::invalid_argument("gguf swiglu: gate and up must both be [N,K]");
        }
        // Dual bindings: one fused launch streaming both weights. The two-product form below costs
        // an extra launch and an fp32 gate plane; measured 0.192 vs 0.205 ms per call. This is also
        // the only way the [gate;up] fused kernel contract is reachable for a dual-binding artifact.
        if (t <= gguf::kMaxVectorColumns && gguf_type(up->qtype) == gguf_type(gate.qtype)) {
            // NINFER_GGUF_COPYTEST=1 times the same launch twice: once on the artifact's mapped
            // weights and once on a plain cudaMalloc copy. It decides whether the in-engine 2x
            // slowdown lives in the mapping or in the kernel. Runs once; needs eager decode.
            if (std::getenv("NINFER_GGUF_COPYTEST") != nullptr) {
                static bool done = false;
                if (!done) {
                    done = true;
                    const std::size_t gate_bytes = std::size_t(rows) * row_bytes(gate);
                    const std::size_t up_bytes   = std::size_t(rows) * row_bytes(*up);
                    void* g2 = nullptr;
                    void* u2 = nullptr;
                    CUDA_CHECK(cudaMalloc(&g2, gate_bytes));
                    CUDA_CHECK(cudaMalloc(&u2, up_bytes));
                    CUDA_CHECK(cudaMemcpy(g2, gate.qdata, gate_bytes, cudaMemcpyDeviceToDevice));
                    CUDA_CHECK(cudaMemcpy(u2, up->qdata, up_bytes, cudaMemcpyDeviceToDevice));
                    void* probe_act =
                        workspace.alloc_bytes(gguf::vector_activation_bytes(gate.k, t)).data;
                    gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.data), gate.k,
                                                     t, nullptr, probe_act, stream);
                    cudaEvent_t ev0 = nullptr;
                    cudaEvent_t ev1 = nullptr;
                    CUDA_CHECK(cudaEventCreate(&ev0));
                    CUDA_CHECK(cudaEventCreate(&ev1));
                    const auto time_once = [&](const void* gp, const void* upp) {
                        CUDA_CHECK(cudaEventRecord(ev0, stream));
                        gguf::vector_swiglu_pair(
                            gguf_type(gate.qtype), gp, row_bytes(gate), upp, row_bytes(*up), rows,
                            gate.k, probe_act, t, static_cast<__nv_bfloat16*>(out.data), rows, stream);
                        CUDA_CHECK(cudaEventRecord(ev1, stream));
                        CUDA_CHECK(cudaEventSynchronize(ev1));
                        float ms = 0.0F;
                        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
                        return ms;
                    };
                    const float t_artifact = time_once(gate.qdata, up->qdata);
                    const float t_copy     = time_once(g2, u2);
                    // 200 back-to-back reps of the artifact pair: steady state without any other
                    // kernel in between. If this reaches the standalone bench number the data is
                    // fine and the decode-time 0.19 ms comes from interleaving, not from the weights.
                    double total = 0.0;
                    float best = 1.0e9F;
                    for (int rep = 0; rep < 200; ++rep) {
                        const float ms = time_once(gate.qdata, up->qdata);
                        total += double(ms);
                        if (ms < best) { best = ms; }
                    }
                    const double mean = total / 200.0;
                    // Second arm: the real decode does quantize -> GEMV as two launches. Timing the
                    // pair back-to-back isolates what one extra launch costs.
                    cudaEvent_t q0 = nullptr;
                    cudaEvent_t q1 = nullptr;
                    CUDA_CHECK(cudaEventCreate(&q0));
                    CUDA_CHECK(cudaEventCreate(&q1));
                    CUDA_CHECK(cudaEventRecord(q0, stream));
                    for (int rep = 0; rep < 200; ++rep) {
                        gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.data),
                                                         gate.k, t, nullptr, probe_act, stream);
                        gguf::vector_swiglu_pair(
                            gguf_type(gate.qtype), gate.qdata, row_bytes(gate), up->qdata,
                            row_bytes(*up), rows, gate.k, probe_act, t,
                            static_cast<__nv_bfloat16*>(out.data), rows, stream);
                    }
                    CUDA_CHECK(cudaEventRecord(q1, stream));
                    CUDA_CHECK(cudaEventSynchronize(q1));
                    float pair_ms = 0.0F;
                    CUDA_CHECK(cudaEventElapsedTime(&pair_ms, q0, q1));
                    std::fprintf(stderr,
                                 "[ggufcopy] first=%.4f copy=%.4f | 200 reps mean=%.4f best=%.4f ms "
                                 "(%.1f GB/s) | 200x(quantize+fused) mean=%.4f ms -> gap/launch=%.4f\n",
                                 t_artifact, t_copy, mean, double(best),
                                 2.0 * double(gate_bytes) / (mean * 1e-3) / 1e9,
                                 double(pair_ms) / 200.0,
                                 (double(pair_ms) / 200.0) - mean);
                    std::fflush(stderr);
                }
            }
            void* activation = workspace.alloc_bytes(gguf::vector_activation_bytes(gate.k, t)).data;
            gguf::quantize_vector_activation(static_cast<const __nv_bfloat16*>(x.data), gate.k, t,
                                             nullptr, activation, stream);
            gguf::vector_swiglu_pair(gguf_type(gate.qtype), gate.qdata, row_bytes(gate), up->qdata,
                                     row_bytes(*up), rows, gate.k, activation, t,
                                     static_cast<__nv_bfloat16*>(out.data), rows, stream);
            return;
        }
        up_rows = *up;
    }
    auto* gate_plane =
        static_cast<float*>(workspace.alloc_bytes(std::size_t(rows) * t * sizeof(float)).data);
    GgufProduct products[2];
    products[0].weight   = &gate_rows;
    products[0].f32      = gate_plane;
    products[0].f32_rows = rows;
    products[1].weight   = &up_rows;
    products[1].out      = &out;
    products[1].epilogue = GgufEpilogue::GateProduct;
    products[1].gate     = gate_plane;
    gguf_project(x, products, workspace, stream);
}

void gguf_embedding(const Weight& table, const Tensor& ids, Tensor& out, cudaStream_t stream) {
    require_gguf(table, "gguf embedding");
    if (ids.dtype != DType::I32 || !ids.is_contiguous() || out.dtype != DType::BF16 ||
        !out.is_contiguous() || out.ne[0] != table.k || out.numel() != std::int64_t(table.k) * ids.numel()) {
        throw std::invalid_argument("gguf embedding: expected I32 ids and BF16 [K, ids]");
    }
    gguf::dequantize_rows(gguf_type(table.qtype), table.qdata, row_bytes(table), table.k,
                          static_cast<const std::int32_t*>(ids.data),
                          static_cast<int>(ids.numel()), static_cast<__nv_bfloat16*>(out.data),
                          table.k, stream);
}

} // namespace ninfer::ops::detail
