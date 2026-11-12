# 改动清单（相对迁移初期的上游移植基线）

本仓库没有 git，回滚靠同目录 `.bak_*` 备份文件。下面按文件列出**本版本里生效的功能性改动**，
每条都标注了对应的备份文件（回滚点）。

## ops/softmax_attention/dense/causal_cache/decode_t1_simt.cuh  （T=1 decode 的 SIMT attention，核心）
- `partial_m[stat] = gmax_v[r]`：scaled 域的最大值，不再二次乘 scale。**这是长上下文读不出 needle 的根因之一。**
- warp 局部在线 softmax：每 warp 独立 `(m,l,o)`，热循环零 `__syncthreads()`，收尾一次 flash 合并（`m_warp/l_warp/gmax_v/lsum_v/o_stage`）。
- int4（KInt4E8）读路径：lane 拥有连续 8 dim（1 个 uint32 字 + 1 个组 scale），写回 `partial_acc[acc_base + lane*8 + u]`；
  Q 的 Hadamard 旋转改经 `__shared__ float q_rot_s[256]` 重排（避免 96 float 同时活跃导致 spill）。
- 4-key 展开分支（`if constexpr (KInt4E8)`，warp 一次发 4 个 K/V word）；实测零收益，保留。
- witness 校验和（`w_ksum/w_vsum/w_psum/w_pgs`）包进 `#if defined(NINFER_KERNEL_WITNESS)`（默认不编译）。
- `__launch_bounds__(kSimtThreads, (QRows <= 2 ? 4 : 1))`。
- 回滚点：`.bak_perf5_20261030`（warp 局部改写前）、`.bak_perf6_head`（4-key 展开前）、`.bak_kint8`、`.bak_kprefetch`。

## ops/softmax_attention/dense/causal_cache/decode_t1_wmma_launch.cu
- `simt_group` 默认值改成按格式：`if (e) return e[0] != '0'; return !KInt4E8;`（int4 用 2-row CTA，+8%）。
- `NINFER_ATTN_SIMT`/`NINFER_SIMT_SPLITS`/`NINFER_SIMT_GROUP`/`NINFER_SKIP_ATTN_SIMT`/`NINFER_SIMT_DBGP` 调试门。
- `[simt-splits] ... mfix=scaled` 标记行（用于确认换库生效）。
- multi-token verify 入口 `causal_decode_t1_simt_multi_token_launch`（`tokens<2||tokens>8` 返回 false）**仍是 per-token 循环**，这是 96K+ MTP 不赚的结构原因。
- 回滚点：`.bak_simtdefault_20261027`、`.bak_cache_dump`。

## ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp
- int4（`RotatedInt4KeyInt4ValueE8`）加入 width>16 的 prompt 路由白名单（prefill O(n²)→O(n)，171→698 t/s）。
- `NINFER_CACHED_WMMA` 门控默认关（cached WMMA 变体会产出全 0，MTP prefill final 列曾因此恒 0）。
- `NINFER_ATTN_XCHECK` 对拍分支（SIMT vs SmallT；**不能当 oracle**，两内核会重复 append KV）。
- `decode_t1_wmma_enabled()` 在 `prompt.cu:33-46`，读 `NINFER_DECODE_WMMA`。

## ops/softmax_attention/dense/causal_cache/prompt_bf16_wmma.cuh + prompt.cu
- 新增 `bool KInt4` 模板形参：K 走 G64 int8 scale 索引、V 走 G32 int4 scale 索引；
  `NINFER_INT4_PROMPT_WMMA`（默认 ON）门控。→ int4 prefill 171→698 t/s。
- `cudaFuncAttributeMaxDynamicSharedMemorySize = kCausalPromptWmmaInt8SmemBytes`。
- 回滚点：`.bak_int4bf16_20261027`（int4-s8 实验回退到这里）。

## ops/softmax_attention/dense/causal_cache/small_t.cu
- `causal_small_t_launch_capacity()` 的 device_cu 块读 `NINFER_SPLIT_CTAS_PER_SM`（默认 4）。
  实测分片数对 decode 无影响（32/64/85 splits 同速）。**巨型 TU，重编 15-20 分钟。**

## core/paged_kv_cache.cpp
- `DeviceKVPagePool` 构造函数加 `NINFER_KV_ZERO_INIT`（默认 ON）逐 plane `cudaMemset`；
  根因是 `zero_pages` 只在 graph capture 时调一次，运行时新分配的 page 从不清零。→ int4 从随机抖动变位级确定。
- 回滚点：`.bak_kvzero_20261027`。

## ops/kv_cache/append/launch.cu + kernel.cuh
- 新增 `NINFER_KV_DUMP` / `NINFER_KV_DUMP_SEQ`（按 append 序号 dump KV 池，用于确定性对拍）。
- append 内核里 `__shfl_sync` 广播改成每线程直读 `paged_kv_physical_page(block_table, position)`。
- 回滚点：`.bak_kvdump`。

## models/qwen3_5/execution/mtp.cpp
- 移植上游的 `unpacked()/gguf_rows()/row_bytes()` 与四个分支，使 MTP 的 q/k/g/v 支持 **gguf_q6_k**
  （官方 v3 artifact 的 MTP 权重就是 q6_k，否则报 `linear_pair: invalid first weight`）。

## models/qwen3_5/execution/text.cpp
- `NINFER_PROFILE_DECODE`（每 64 次 layer-call 打一次 attn/gdn/mlp 的 ms；**需要 --max-new ≥ 70**）。
- `NINFER_PROFILE_MIXER`、`NINFER_SKIP_*` 系列消融开关（注意 SKIP_MLP/SKIP_GDN 在 MTP 路径基本是 no-op）。
- `NINFER_DEBUG_LAYERS` + `NINFER_DUMP_MIN_T`/`NINFER_DUMP_MAX` 逐 op 校验和探针。
- `NINFER_MTP_INGRESSDBG` MTP 每轮 ingress 探针（有 graph 时只有 capture 能跑，body 不执行）。
- 回滚点：`.bak_mixerenv_20261027`、`.bak_profmlp_20261027`、`.bak_packedfinal_pre` 等。

## ops/linear/gguf/ggml_bridge_vec.cuh
- multi-column（T>1）行流 4→2 + `vec_step<T>` + `#pragma unroll 4`。**实测零收益**（保留，已验正确）。
- 回滚点：`.bak_mtcols_20261030`、`.bak_iq3fix`、`.bak_iq4fix`、`.bak_ksunroll_20261027`、`.bak_kvecwarps_20261027`。

## 启动/工作区
- `startup.cpp:362` 的 prefill output-head scratch 按 1 列预留（原按 chunk 列 → 白吃 1.02 GiB → prefill 126→494 t/s）。
- `milestones/20260928_decode26_prefill431/probe/mini_main.cpp` 加 `NINFER_PREFILL_CHUNK` 环境变量覆盖。

## 未修（明确的技术债）
- `ops/common/mma.cuh:584-590` 的 `mma_fp8_e4m3` 是 `__builtin_trap(); // fp8 WMMA not ported yet`；
  fp8 KV 的 prefill QK 调用后整层 NaN。修法：反量化到 half + `mma_f16`（PV 已在用）。fp8 最多到 ~96K。
- `mma.cuh:678` nvfp4 block-scale MMA 同样是 trap。
- int8 KV（Int8Group64）走真 WMMA 的 KInt8 分支，in-kernel fresh-row append 缺 `owns_fresh_row` 守卫，
  非确定性（前 12 token 相同后分叉）。
- dflash2：decode 5.6 t/s、输出垃圾。
## 本版（pre-r4）新增条目

- `models/qwen3_5/program/decode.cpp`：`[mtp-diag]` 由无条件打印改为 `NINFER_MTP_DIAG=1` 才打印
  （原来每轮 MTP 都会往 stderr 写一行，双击聊天时串进回答里）。
- `models/qwen3_5/execution/text.cpp`：`[rmsnorm-ptr]` 的 NaN 预扫描 + 打印由无条件改为 `NINFER_DBG_PTR=1`；
  该扫描是 kernel + 同步，原来每 decode step 都跑一次。
- `ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp`：`[attn-fallback]` 改为 `NINFER_ATTN_TRACE=1`。
- `milestones/20260928_decode26_prefill431/probe/mini_main.cpp`：chat 模式下把 stderr `freopen` 到
  `ninfer-chat.log`（`NINFER_CHAT_LOG=0` 可关闭），控制台只留回答与每轮统计。

## 本版（pre-r2）新增条目

### ops/softmax_attention/dense/causal_cache/decode_t1_wmma_launch.cu
- multi-token verify 入口改为**批量**：`grid.z == tokens`，外层按 split×kv_head，一次发全部草稿 token
  （`NINFER_MT_BATCH=0` 回退 per-token 循环；`NINFER_MT_SPLITS` 单独调 verify 的 splits）。
  pre-r1 文档里「仍是 per-token 循环」的说法作废。
- `NINFER_STEP_T` 每 token 墙钟计时（比 cudaEvent 分段表可靠）。

### core/arena.cu
- **分配带宽探针默认关**：`alloc_bw_enabled()` 从 `e == nullptr || e[0] != '0'` 改为 `e != nullptr && e[0] != '0'`。
  探针会为每个 buffer 做一次 64MB D2D 测速，改变分配布局且多占显存，生产不应带。
- `NINFER_ALLOC_LOG` 逐次分配日志。

### models/qwen3_5/program/planning/startup.cpp
- `NINFER_KVCAP_DIAG`：放宽 `kv_capacity must be at least max_context` 与
  `kv_capacity is outside the usable range` 两道校验（仅实验用；第三道 `kv_capacity.cpp` 的 curve 校验仍会拦，
  说明 pool 与 envelope 在模型 curve 里被绑死，无法真正解耦）。

### milestones/20260928_decode26_prefill431/probe/mini_main.cpp（测量外壳，不影响引擎）
- `[mem]` 引擎显存账本（`Engine::memory_summary()`：kv_payload / weights / seq / ws / runtime_res / graph / slack / avail_*）。
- `[vram]` 真实 `cudaMemGetInfo`（before-engine / after-engine / before-gen / after-gen）。
- `[xfer]` 64MB D2D 带宽探针（**必须前后 cudaDeviceSynchronize**，否则读数荒谬）。
- `NINFER_DUMMY_MB`（引擎之后 commit 一块牺牲品显存）、`NINFER_PREALLOC_MB`（引擎之前占用）、`NINFER_PREFILL_CHUNK`、
  `NINFER_VRAM_CEILING=pre|post|both`（alloc-until-fail 探针：128 MB 分块分配直到失败，打印可分配总量）。
  实测干净进程可分配 ≥24.00 GiB、建好引擎后仍可分配 7-8 GiB ⇒ **cudaMalloc 是虚拟超额提交，分配不失败**。

### 构建/跑测脚本（`build/`，不入快照）
- `_bmem.bat`（只重建 mini_main：`compile_abs.py` + `NINFER_NORDC=1 link_exe.py`）。
- `_rebuild_g.bat <相对 src 路径>`：通用单 TU 重建（旧的 `_rebuild.bat` 写死 decode TU，传参会被忽略）。
- `_rel.bat` / `_rel2.bat`（发布验收矩阵）、`_snap.bat`（快照 stage）、`_ll4.bat`（llama.cpp 对照）。

## 本版确认的不可行路线（别再试）
- `--kvcap` 小于 `--ctx` 来「大 envelope + 小池子」：三道校验依次拦，最后一道在 `runtime/engine/kv_capacity.cpp:54/:68`
  （`SequenceCapacityCurve` 要求 pages 落在 [minimum, maximum] 区间），模型 curve 里 pool 与 envelope 绑死。
- 后置牺牲品显存（`NINFER_DUMMY_MB`）：不改变快态、也不救慢态。
- cudaEvent 分段 profile 表（`NINFER_PROFILE_DECODE` 的 per-op 行）：数值不可信，只用 `[prof-decode] ... layers/norm+lmhead/step-total`。
## 0.1.1 追加（2026-11-12）

### src/ops/common/mma.cuh
- **gfx12 WMMA 的 B 操作数打包修正**：硬件 WMMA 的 B 片段按**输出列**分布——lane L 持有列
  `L & 15`，K 半区由 `L < 16`（K[0:8]）还是 `L >= 16`（K[8:16]）决定。原实现按行打包，
  导致 packed attention 的输出只有 1/8 通道非零。现为
  `col = L & 15; src = 4 * (col & 7) + i;` 取 `b0`/`b1` 后按 `L < 16` 选择，
  并把逻辑 n8 之外的列（`col >= 8`）置零以保持 dummy 列不引入额外 gather。
- C 片段的 seed/read 保持原文档映射（`D(i + 8*(L>=16), L%16)`）不变；
  另有一版「按替代 C 映射」的实验已完整回退，不要再用。
- 验证：孤立 16×16×16 identity GEMM（软 MMA 0 错 / 硬 WMMA 120 错）→ 修复后两者 0 错；
  链接型 packed attention identity 探针 tile 16/32/64 全 `mismatches=0/9216`；
  非零 Q/K/V 8-token 双段对拍 `rel_l2=0.00186 / max_error=0.00516`。

### src/ops/kernel/layer_norm.cuh
- **paired D=1152 专用核的 shuffle UB**：诊断统计分支原为
  `if (threadIdx.x == 0) { g_ln_stat[3] = __shfl_sync(0xffffffffu, local.count, 0); }`，
  单 lane 参与全 mask shuffle 属未定义行为，使专用核输出的统计量变垃圾。
  改为 lane 0 直接读已约简的 `local.count`。

### src/models/qwen3_5/execution/vision.cpp
- **merger 输出默认不再缩 0.1**：改为仅当显式设置 `NINFER_VISION_SCALE` 且因子有效时缩放。
  原因：merger 的输出本就与文本 embedding 同空间，压 1/10 后单色块的位置信号被压没，
  模型退化成文本先验（位置题一律答 top-left）。
- **BF16 覆盖加载器**：patch embedding 支持双帧权重（`[c][t]` 交错 + loader 转置约定）；
  layer-norm 与 post_ln 的 gamma/beta 改为**逐位拷贝、不加偏移**（此前多加 1.0，
  使全 BF16 档塌成银灰）。实验开关 `NINFER_VISION_NORM_PLUS_ONE` 删除。

### src/models/qwen3_5/frontend/processor.cpp
- `swap_hw` 默认改成与参考实现一致（`swap_env != nullptr && swap_env[0] != '0'`），
  此前未设置环境变量时默认交换 h/w。

### src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh
- dispatcher `mma16_swiz_x_raw_d` 引用其后才定义的 `mma16_swiz_x_raw`，补前向声明。
  干净 CMake 构建此前会直接失败（`use of undeclared identifier 'mma16_swiz_x_raw'`），
  增量构建因为用着陈旧 obj 而看不出来。

### src/models/qwen3_5/program/storage/{kv_archive.h,kv_archive.cpp,kv_retrieval.h,kv_retrieval.cu,context.cpp}
- KV 归档 + 检索召回（r9 已发布，本版未改语义），仅随视觉/服务一起发布。

### src/media/decode/decode_stub.cpp
- 换成 WIC 实现：PNG / JPEG / GIF / BMP / TIFF；带 alpha 时合成白底；无外部依赖
  （不再需要 FFmpeg / zlib）。

### src/ops/linear/bf16/bf16_generic.cu、bf16_dispatch.cpp
- BF16 视觉覆盖档的通用 GEMM 回退路径（白名单未命中时走它）。

### harness/serve_mini_main.cpp、harness/build_serve.py
- 新增 OpenAI 兼容服务源码与独立构建脚本（不注册进 CMake，见 docs/PATCHLIST 之外的说明）。

### harness/full_rebuild.py
- 归档改用 llvm-ar 响应文件（对象数量会触发 `WinError 206 文件名或扩展名太长`）。
- 注意：**必须**用 `build/_rearchive_current.py` 归档，`full_rebuild.py` 的"归档全部 obj"
  路径会把探针/重复 obj 一起进库，导致 `duplicate symbol: ternary_dispatch_basis…`。