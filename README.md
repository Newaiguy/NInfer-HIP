# NInfer-HIP `0.1.0-pre-r4`

> 把 NInfer 推理引擎推到 AMD RDNA4（gfx1201）—— 一张 16GB 的 RX 9070 XT 上，
> 27B 模型跑到 126K 上下文、decode 22.9 t/s，开 MTP 后 65K 上下文 41.6 t/s。
> 同卡同权重下 decode 约为 llama.cpp 的 1.4x（开 MTP 约 2x）。

**为什么会有这个项目、这一个月里发生了什么 → [docs/STORY.md](docs/STORY.md)**（一个普通 AMD 用户的记录）

**预编译二进制 → [Releases](https://github.com/Newaiguy/NInfer-HIP/releases)**
（仓库不放二进制：exe 193 MB，超过 GitHub 单文件 100 MB 限制；模型权重也不在仓库内）

NInfer 引擎（原生 CUDA 项目，Apache-2.0）到 **AMD RDNA4 / gfx1201** 的 HIP 移植分支。
目标机型：RX 9070 XT 16GB（256-bit GDDR6，标称 640 GB/s），Windows 11 26200 + HIP SDK 7.2.3。
模型：`WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3`（官方 v3 单文件 artifact，15,017,456,896 B，
内含 text + vision + mtp + dflash2 四个组件，本版本只用 text 与 mtp）。

pre-r4 相对 pre-r3 的变化：

1. **聊天输出干净了**：引擎的诊断行（`[mtp-diag]` / `[rmsnorm-ptr]` / `[attn-fallback]`）全部改成
   `NINFER_MTP_DIAG` / `NINFER_DBG_PTR` / `NINFER_ATTN_TRACE` 显式开启；聊天模式下 stderr 统一重定向到
   同目录 `ninfer-chat.log`，不会再串到回答中间。
2. **顺手拿掉一处每步开销**：`[rmsnorm-ptr]` 之前每 decode step 都跑一次 NaN 扫描 kernel + 同步，
   现在只在 `NINFER_DBG_PTR=1` 时才做。

pre-r3 相对 pre-r2 的变化：

1. **引擎内置交互式聊天模式**（`--chat`）：流式输出、多轮、前缀复用（第 2 轮起只 prefill 新增 token）、
   `/think /clear /stats /system` 命令；实测短上下文 **57-60 t/s**（MTP dw2）。
2. **双击即用的启动器** `harness/launchers/chat.bat`：自动找 exe、记住模型路径、四种场景菜单，
   不需要记参数也不需要 Python 包装（引擎已能直接跑，DLL 由启动器补 PATH）。

pre-r2 相对 pre-r1 的两大变化：

1. **长上下文掉速的根因定量到「工作集越过 Windows 高速驻留预算」**（KV 页改按 PCIe 速度读），
   并给出唯一有效的操作规则：**`--ctx` 必须贴着实际 prompt 长度**。
2. **126K 口径纠正**：pre-r1 记的「126K decode 23.5」和本版初稿记的「10.1」都不对 ——
   真值是 **22.9 t/s @ `--ctx 126464`**，10.1 来自把 `--ctx` 开成 131072（多开 4,743 token 越预算）。

## 1. 状态总览（本版本实测）

全部开 CUDA graph、KV 格式 `--kv rk4v4-e8`（RotatedInt4KeyInt4ValueE8）、`--max-concurrency 1`、
单张 RX 9070 XT 16GB。完整命令见 [`docs/VALIDATION.md`](docs/VALIDATION.md)。

| 项目 | 状态 |
|---|---|
| **交互式聊天** | ✅ `--chat`：流式输出、多轮（前缀复用）、`/think /stats /clear /system /exit`；`harness/launchers/chat.bat` 双击启动 |
| 正确性 29 token ~ 126K token | ✅ 全部通过（needle 逐字符命中、数字列表、长文摘要语义正确） |
| **126K 真实上下文** | ✅ **decode 22.9 t/s · prefill 294**（`--ctx 126464`，needle `K8M2Q9` 命中） |
| 96K 真实上下文 | ✅ decode **25.2**（AR）· **34.0**（MTP dw2）· prefill 346 |
| 短上下文 decode（AR，宽 1） | 37.3 @29 token · 36.6 @529 · 35.4 @5.8K · 32.3 @32K |
| 中长上下文 decode（AR） | 30.0 @48K · 28.4 @65K · 25.2 @96K · **22.9 @126K**（均要求 `--ctx` 贴紧） |
| **MTP 推测解码（dw2）** | **77.8 @2.8K（+115%）** · 64.5 @24K（+100%） · **41.6 @65K（+47%）** · 34.9 @48K（+16%） · **34.0 @96K（+35%）** · ❌126K |
| prefill | 848 @5.8K · 610 @32K · 505 @48K · 442 @64K · 346 @96K · 294 @126K |
| int4 KV | ✅ 16GB 上跑 128K 的唯一容量路径 |
| int8 KV | ❌ 不可用（非确定性） |
| fp8 KV | ❌ 不可用（QK 的 MMA 是未移植的 trap 桩；容量也只到 ~96K） |
| dflash2 | ❌ 坏路径（decode 5.6 t/s、输出垃圾），未启用 |
| 256K | ❌ 容量不够（int4 KV 8.86GB + 权重 11.77GB > 16GB），需 KV offload |
| **多模态（图像输入）** | ❌ **本版不支持**（artifact 内含 vision 组件、引擎也编入了 vision 通路，但本版的 harness 没有图像入口；下一版接） |
| 并发 | ❌ 只支持 1（`--max-concurrency 1`） |

## 2. 最重要的一条：`--ctx` 必须贴着实际 prompt 长度

KV 池按 `--ctx` **一次性全量提交**（int4 = `capacity × 17,920 B/token`）。Windows 下驱动只愿意把
约 **12.6 ~ 13.6 GiB** 的引擎工作集留在显存（WDDM 硬预留 1/8 + 桌面占用 + 驱动开销，且**会随桌面状态漂移**）。
一旦引擎总提交越过这条线，**整个 KV 池**滑到系统内存，每 step 的 KV 读取改走 PCIe（~30 GB/s），
代价与「每 step 实际读多少 KV」成正比 ⇒ 长提示腰斩，短提示几乎无感。

**案例 A（126K 真实长文，本版最大收益）** —— 同一份 126,329 token 提示：

| `--ctx` | 引擎总提交 | prefill | decode |
|---|---|---|---|
| 131072 | 13.67 GiB | 222.2 | 10.12 |
| **126464**（prompt+20） | **13.59 GiB** | **293.97** | **22.91** |
| 126464 + `NINFER_PREFILL_CHUNK=512` | 13.48 GiB | 279.09 | 22.93 |

⇒ **2.26× 全部来自把 `--ctx` 收紧 4,743 个 token**（复测三次：22.91 / 22.93 / 23.01，可复现）。

**案例 B（65K 提示）**：`--ctx 65536`（12.58 GiB）→ 28.4 t/s；`--ctx 131072`（13.67 GiB）→ 15.0 t/s。

**悬崖位置 = 提交量，不是 ctx 值**（同一条 48K 提示，不同时刻扫）：

| `--ctx` | 引擎总提交 | 上午扫（预算 12.6 GiB） | 下午扫（预算 13.6 GiB） |
|---|---|---|---|
| 65536 | 12.58 GiB | **30.1** | — |
| 69632 | 12.64 GiB | **16.8** ← 越线 | — |
| 98304 | 13.12 GiB | 16.0 | **29.91** |
| 114688 | 13.39 GiB | — | **29.86** |
| 122880 | 13.53 GiB | — | **29.99** |
| 131072 | 13.67 GiB | 16.9 | **17.12** ← 越线 |

⇒ 同一条命令在上午是 16.8 t/s、下午是 30.0 t/s（**预算当天漂移了约 1 GiB**）。
所以「`--ctx` 设成某个固定安全值」是不可靠的，唯一稳的做法是**按实际长度设**；
若观察到速度腰斩，第一件事就是把 `--ctx` 往下调 4~8K。

机制的完整证据链、已排除的 8 条假设与候选修法见 [`docs/PERF-CAPACITY.md`](docs/PERF-CAPACITY.md)
（注意两点：① 启动时对 KV 池的 3 遍扫描在各种 `--ctx` 下都是 ~30 GB/s —— 它跑在页被写之前，
**不能**用作驻留证据；② `cudaMalloc` 是虚拟超额提交，干净进程能「分配」到 24 GiB 以上，
所以「分配失败」从来不是限制，限制是「愿不愿意留在显存」）。

## 3. 推荐配置（本引擎目前的最优参数组合）

三个开关真正重要：**`--ctx`（= prompt + max_new，一个 token 别多开）**、
**`--spec mtp --draft-tokens 2`**（收益取决于真实上下文长度与是否越过驻留预算）、
**`--kv rk4v4-e8`**（所有配置都带上）。

| 真实上下文 | 完整参数（在 exe + artifact 之后） | decode 实测 |
|---|---|---|
| ≤8K | `--kv rk4v4-e8 --ctx 16384 --spec mtp --draft-tokens 2` | **77.8**（真实 2.8K，AR 36.2 ⇒ 2.15×） |
| 16-32K | `--kv rk4v4-e8 --ctx 32768 --spec mtp --draft-tokens 2` | **64.5**（真实 24K，AR 32.3 ⇒ 2.0×） |
| 48K | `--kv rk4v4-e8 --ctx 49152 --spec mtp --draft-tokens 2` | **34.9**（真实 48K，AR 30.1 ⇒ 1.16×） |
| 64K | `--kv rk4v4-e8 --ctx 65536 --spec mtp --draft-tokens 2` | **41.6**（AR 28.3 ⇒ 1.47×） |
| 96K | `--kv rk4v4-e8 --ctx 98304 --spec mtp --draft-tokens 2` | **34.0**（AR 25.2 ⇒ 1.35×） |
| **126K** | `--kv rk4v4-e8 --ctx 126464`（**必须关 MTP**） | **22.9**（MTP 会多提交 0.57 GB 越线 ⇒ 掉到 10.6） |
| 想开大容量但对话很短 | `--ctx 81920 --spec mtp --draft-tokens 2` | **56.0**（真实 2.8K） |

四条硬规则：

1. **`--ctx` = prompt + max_new，绝不多开。** 多开 4.7K 就从 22.9 掉到 10.1。
2. **`--draft-tokens 2` 是唯一划算档位。** `3` 以上草稿链自回归退化，实测净亏；`1` 只有 ~1.3×。
3. **MTP 的内存开销是固定的 +0.57 GB**（mtp 权重 348 MB + draft state 190 MB + verify 图 73 MB），
   所以 MTP 的有效上限由**提交预算**决定：今天 96K 能跑（13.62 GiB，34.0 t/s）、126K 不能（14.12 GiB，10.6 t/s）。
   真实上下文 ≥100K 时直接关 MTP 用 AR。
4. **真实上下文越长，MTP 倍率越低**（2.15× @2.8K → 1.35× @96K），但直到 96K 都仍是净赚。

prefill 侧的额外开关（可选）：`NINFER_PREFILL_CHUNK`（默认 1024）。真实上下文 ≤32K 且 `--ctx` ≤65536 时设 4096
可再拿 ~9%（32K: 347 → 378 t/s）；`--ctx` ≥69632 时会 OOM，必须留默认。

## 4. 快速开始

### A. 双击开聊（推荐，不需要命令行的方式）

1. 把 `ninfer_mini_nordc.exe` 和 `chat.bat` 放在同一个文件夹里；
2. 双击 `chat.bat`，第一次会问模型路径（把 `.ninfer` 文件拖进窗口回车即可，会记在 `model.txt`）；
3. 选场景：

```
[1]  daily chat         ctx 16384   + MTP dw2   ~ 78 t/s
[2]  long doc / code   ctx 32768   + MTP dw2   ~ 64 t/s
[3]  long doc / code   ctx 65536   + MTP dw2   ~ 42 t/s
[4]  whole book        ctx 131072  no MTP      ~ 23 t/s
[5]  custom arguments
```

进去以后直接打字提问（引擎的诊断日志在同目录 `ninfer-chat.log`，不会串到回答里）；`/help` 看命令，`/think on` 开思考，`/stats` 看这轮速度，`/exit` 退出。
每轮结束会打印 `[N prompt tokens (M reused), K new, prefill X t/s, decode Y t/s]` ——
**M reused 就是前缀复用省下来的 token**（多轮对话里通常 50%）。

### B. 命令行 / 批量（脚本、批量评测用）

```bat
REM 1) 引擎 exe 不能直接调：必须走 capture 包装（否则 exit 0xC0000135，DLL 找不到）
python probe_rebuilt\run_capture.py build\infer_mini_nordc.exe ^
    E:\ninfer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer ^
    --tasks tasks\tasks_cp8km.txt --ctx 16384 --max-new 128 --kv rk4v4-e8

REM 2) 加 MTP 推测解码（≤96K 都划算，务必 dw2，ctx 贴紧）
python probe_rebuilt\run_capture.py build\infer_mini_nordc.exe ^
    E:\ninfer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer ^
    --tasks tasks\tasks_cp64km.txt --ctx 65536 --max-new 128 --kv rk4v4-e8 ^
    --spec mtp --draft-tokens 2

REM 3) 128K 长文（ctx = prompt + max_new，关 MTP）
python probe_rebuilt\run_capture.py build\infer_mini_nordc.exe ^
    E:\ninfer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer ^
    --tasks tasks\tasks_true128kL.txt --ctx 126464 --max-new 20 --kv rk4v4-e8
```

- **`--kv rk4v4-e8` 是 128K 的必需品**（bf16 KV 需 9.99 GB、int8 KV 需 5.83 GB，而运行时可分配只有 5.16 GB）。
- 输出读法：`run_capture.py` 结束才整段输出；日志请用 Python 以 UTF-8 读（PowerShell 控制台是 GBK，
  中文会出现二次编码假象；`| Out-File` / `*>` 默认写 UTF-16LE）。
- `tasks` 文件格式：首行必须 `###ID <name>`；**正文内任何以 `#` 开头的行都会被当成新任务分隔符**；
  文件必须是无 BOM 的 UTF-8。

## 5. 已知坑与硬限制

- **`--ctx` 开大了就掉速**（§2）。这是本版本已知的主要性能缺陷，规避手段只有「按实际长度设 ctx」。
- **MTP 到 96K 为止**：126K 加 MTP 会多提交 0.57 GB 越过预算 ⇒ 10.6 t/s（比 AR 还慢）。≥100K 真实上下文请关 MTP。
- **`--draft-tokens` 用 2**：3 以上草稿链自回归退化，每轮实际接受数掉下来。
- `--no-graph` + `NINFER_PROFILE_MIXER=1` + `--spec mtp` 会挂死 GPU 队列；`NINFER_PROFILE_DECODE` 必须配 `--no-graph`。
- **绝不对卡在 GPU 上的引擎进程 `Stop-Process`**：杀进程不会终止已在 GPU 上跑的 kernel，
  死循环 kernel 会把显示卡死（2026-10-30 已发生过一次黑屏硬复位）。要么等，要么重启机器。
- int8 / fp8 / dflash2 / 256K / 并发 > 1 均不可用（见 §1 与 [`docs/PATCHLIST.md`](docs/PATCHLIST.md) 末节）。

## 6. 目录

```
（引擎 exe 不在仓库里：从 Releases 下载 ninfer_mini_nordc.exe，193,543,168 B，
          sha256 见 VERSION.txt；把它和 harness/launchers/chat.bat 放同一个文件夹即可双击开聊）
src/      infer/{include,src,third_party} + shim（HIP 兼容层），全量可重建源码
harness/  构建与跑测工具（compile_one / swap_lib / link_exe / run_capture）
          harness/launchers/chat.bat  双击启动器（+ model.txt.example）
tasks/    验收语料（含 126K needle 语料）
docs/     STORY / VALIDATION / PERF-CAPACITY / CHANGELOG / PATCHLIST / BUILD
VERSION.txt  版本、实测指标、引擎 sha256、环境变量与纪律
```

构建方式见 [`docs/BUILD.md`](docs/BUILD.md)；每个改动的回滚点见 [`docs/PATCHLIST.md`](docs/PATCHLIST.md)。
