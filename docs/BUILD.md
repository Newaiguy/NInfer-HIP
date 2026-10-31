# 构建链（Windows / HIP SDK 7.2.3 / gfx1201）

单 TU 增量重建 —— 全量 cmake 在 Windows 上不可用，这是唯一路径：

1. `python probe_rebuilt\compile_one.py <相对 infer/src 的路径>`
   - 例：`ops/softmax_attention/dense/causal_cache/decode_t1_wmma_launch.cu`
   - **永远 exit 0**；真实判据 = 输出里的 `rc=0 <obj 名>` 行。
   - obj 落在 `build_nordc/<路径下划线化>.o|.obj`。
   - GEMV 存根 TU 必须带 `set NINFER_EXTRA_DEFS=-DNINFER_VEC_DEBUG_OFF`（否则 undefined symbol `g_vec_rdbg`）。
2. `python probe_rebuilt\swap_lib.py <同一路径>` → 换进 `build_nordc/ninfer_hip.lib`
   （GEMV 的 obj 也在这个库里，已确认）。
3. `set NINFER_NORDC=1` + `python probe_rebuilt\link_exe.py` → `build\ninfer_mini_nordc.exe`
   - **不带 NINFER_NORDC=1 会生成 build\infer_mini.exe**，老 exe 时间戳不变 → 误判构建没生效。
   - 链接前必须 `taskkill /F /IM ninfer_mini_nordc.exe`（否则 permission denied）。

编译耗时：attention/prompt TU 2-5 分钟；GEMV 存根 ~110s；`small_t.cu` 15-20 分钟（巨型 TU）。

## 安全铁律（2026-10-30 黑屏事故后新增）
- **绝不对卡在 GPU 上的引擎进程用 `Stop-Process`**：杀进程不会终止已在 GPU 上运行的 kernel，
  一个死循环 kernel 会把 GPU 队列永久卡死 → 黑屏 → 硬复位。要么等，要么重启机器。
- 不要跑 `--no-graph` + `NINFER_PROFILE_DECODE=1` + `--spec mtp` 这个组合（已确认挂死）。
- 长测试用后台 job + 轮询，单次前台命令受 120s 硬上限。
## 跑测外壳与日志
- **引擎 exe 不能直接调**（exit 0xC0000135，DLL 找不到）。一律：
  `python probe_rebuilt\run_capture.py build\infer_mini_nordc.exe <artifact> <args...>`。
- 日志重定向必须 `cmd /c "bat > log 2>&1"`；PowerShell 的 `>` / `*>` 写 UTF-16LE，读日志用 Python 以 UTF-8 打开。
  PowerShell 控制台是 GBK，中文会二次编码成「乱码」，这不代表引擎输出坏了。
- **判二进制新旧用 `Get-ChildItem <path> | Select Length,LastWriteTime`**（node `fs.statSync` 与 `cmd dir` 会读旧缓存值）。
- 后台跑测只能用 `tools.pwsh` 的 `run_in_background: true`；`Start-Process cmd.exe -ArgumentList "/c","... > log"` 在本机静默失败（日志 0 字节）。

## 环境变量速查（全部默认关/默认最优，生产不需要设置）

| 变量 | 作用 |
|---|---|
| `NINFER_KV_ZERO_INIT` | KV 池构造时全池清零（默认 ON，packed 格式必须） |
| `NINFER_ALLOC_BW` / `NINFER_ALLOC_LOG` | 按 buffer 的带宽探针 / 分配日志（诊断，默认关） |
| `NINFER_STEP_T` | 每 token 墙钟计时 |
| `NINFER_SIMT_SPLITS` / `NINFER_SIMT_GROUP` | decode attention 的 splits / 每 CTA 处理行数 |
| `NINFER_MT_SPLITS` / `NINFER_MT_BATCH` | multi-token verify 的 splits / 批量开关（0=回退 per-token） |
| `NINFER_SKIP_MLP` / `NINFER_SKIP_GDN` / `NINFER_SKIP_ATTN_SIMT` / `NINFER_SKIP_MIXER` | 消融（注意：MTP 路径上 SKIP_MLP/SKIP_GDN 基本是 no-op；SKIP_ATTN 会改变 token 流） |
| `NINFER_PROFILE_DECODE` / `NINFER_PROFILE_MIXER` / `NINFER_PROFILE_PREFILL` | profile（前两个需 `--no-graph`；PROFILE_MIXER 会挂死，禁用） |
| `NINFER_DEBUG_LAYERS` + `NINFER_DUMP_MIN_T` / `NINFER_DUMP_MAX` | 逐 op 校验和探针 |
| `NINFER_KV_DUMP` / `NINFER_KV_DUMP_SEQ` | 按 append 序号 dump KV 池 |
| `NINFER_KVCAP_DIAG` | 放宽 `--kvcap` 校验（仅实验） |
| `NINFER_PREFILL_CHUNK` | prefill 分块（默认 1024；128K 下 4096 会 OOM） |

## 跑测纪律（硬件安全）
- **同一时刻只允许一个 GPU 作业**；GPU 实验严格串行。
- 构建/链接内含 `taskkill` 且要写 exe ⇒ **绝不在 GPU 扫测进行中编译**（会把正在跑的测试杀掉，数据作废）。
- 长测试用后台 job + 轮询；前台命令受 120s 硬上限。