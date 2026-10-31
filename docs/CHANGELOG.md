# CHANGELOG

## 0.1.0-pre-r3 — 2026-10-31

本版**没有改动任何计算路径**（decode/prefill 数字与 pre-r2 完全一致），
只让引擎第一次能"像人一样用"：内置聊天模式 + 双击启动器。

### 新增能力
- **引擎内置 `--chat` 交互模式**（`mini_main.cpp`，约 +150 行）：
  `ChatSink` 实现 `OutputSink`，content 流式打印、reasoning 以灰蓝 `[thinking]` 前缀输出；
  每轮把整段 history 重新 `prepare`，靠**驻留前缀复用**只 prefill 新增 token；
  `ContextLengthExceeded` 时自动丢掉最早一轮 user+assistant 重试一次；
  `SetConsoleOutputCP(CP_UTF8)` + `ENABLE_VIRTUAL_TERMINAL_PROCESSING` 保证中文/颜色正确。
  命令：`/help /clear /stats /think on|off /system <text> /exit`。
- **`harness/launchers/chat.bat`**：双击即用。自动定位 exe（同目录或 `bin\`）、补 ROCm PATH、
  记住模型路径（`model.txt`）、四档场景菜单（ctx 16384/32768/65536/131072）、第 5 项可手输参数。
- 新增 `NINFER_ARGV` 环境变量（`main()` 最前面把 argv 逐条打到 stderr），排查"启动器传参是否正确"用。

### 实测（RX 9070 XT 16GB，`--kv rk4v4-e8 --spec mtp --draft-tokens 2 --ctx 16384`）
| 场景 | 结果 |
|---|---|
| 单轮短问（"法国的首都是哪里"） | 22 prompt tokens → 答"巴黎"，decode 28.4 t/s（1 token，不具代表性） |
| 单轮闲聊（14 new tokens） | decode **57.4 t/s** |
| 多轮第 2 轮（118 new tokens） | prefill 278 t/s、decode **60.4 t/s**，前缀复用 27/55 |
| 非 chat 回归 | `tasks_cp8km` → `7F3K9Q` ✅，prefill 842.8 / decode 35.7（与 pre-r2 一致） |

### 排障记录（值得写下来）
- **启动器一开始启动的是旧 exe**：`build\` 里同时存在 `ninfer_mini_nordc.exe`（08:31 的旧构建，
  没有 `--chat`）和 `infer_mini_nordc.exe`（当晚的新构建），启动器先匹配前者 ⇒ 表现为
  "参数都对、就是不进聊天模式"，甚至 `--chat` 被当成普通 prompt（prompt=14 tokens，
  回复 "Hello! How can I help you today?"）。定位手段：`NINFER_ARGV` 一条都没有 + `for %%F in ("%EXE%") do echo %%~zF`。
  现在启动器会打印 `engine: <名字> <字节数> <时间>`，类似的错一眼可见。
- **`chcp 65001` 会破坏 `set /p` 从重定向 stdin 读值**（菜单直接空转）；删掉即可，UTF-8 交给引擎自己做。
- 测 bat 不要用 PowerShell 的 `cmd /c "bat < file"`（嵌套引号会拿不到 stdin），用
  `python -c "subprocess.run(['cmd','/c',bat], stdin=open(f,'rb'))"`；测试输入文件必须是 CRLF。

## 0.1.0-pre-r2 — 2026-10-31

本版没有改计算路径；它把「长上下文掉速」从「原因不明」变成「已定性 + 已有可操作结论」，
并把 126K 的真实性能从 10.1 纠正到 **22.9 t/s**。发布数字全部重测（见 `VALIDATION.md`）。

### 定性（本版核心）
- **长上下文掉速＝工作集超出 Windows 的高速驻留预算，整个 KV 池按 PCIe 速度读**。
  `--ctx` 决定池大小（int4 = 17,920 B/token），因此 `--ctx` 直接决定整机提交量。
- **预算是区间且当天漂移**：上午实测 12.6 GiB（48K 提示：`--ctx 65536` 总提交 12.58 GiB → 30.1 t/s，
  `--ctx 69632` 12.64 GiB → 16.8 t/s），下午实测 13.5~13.6 GiB（`--ctx 122880` 13.53 GiB → 29.99 t/s，
  `--ctx 131072` 13.67 GiB → 17.12 t/s）。**同一条命令在两个时刻可差 1.8×。**
- **决定性案例（126K 真实长文）**：同一份 126,329 token 提示，
  `--ctx 131072`（13.67 GiB）→ **10.12 t/s**；`--ctx 126464`（13.59 GiB）→ **22.91 t/s**
  （复测 22.93 / 23.01）。**2.26× 只来自把 `--ctx` 收紧 4,743 个 token。**
- 预算 ≈ 13.5~14.2 GB = 物理 16 GB − WDDM 1/8 预留 2.14 GB − 桌面 0.78 GB − 驱动自身约 0.7 GB。
  证据链：① Windows `\GPU Process Memory\` 计数器（Local 12,412 MB、NonLocal 0，Shared 与 Local 同值 =
  同一批页双记账）；② alloc-until-fail 探针：cudaMalloc 是**虚拟超额提交**（干净进程可分配 ≥24 GiB、
  建好引擎仍可分配 7-8 GiB），分配永不失败，只有「愿不愿意留在显存」；
  ③ 台阶点可复现，且台阶跟着**提交量**走而不是跟着 `--ctx` 数值走。
- **可操作结论**：`--ctx` = prompt + max_new，一个 token 都别多开。不要设「固定安全值」（预算会漂移）。

  **注意**：pre-r1/pre-r2 初稿曾用启动期的按 buffer 带宽扫描（KV 池 604 GB/s）作为证据，**那是错的**——
  该扫描在页被写过之前执行（`arena.cu` 的 `DeviceBuffer` 构造函数内），WDDM 尚未把页提升到本地段，
  所以各档 `--ctx` 都读到 26-31 GB/s，反而证明它无效。同理 `NINFER_ALLOC_BW=1` 读三遍也提不上去
  （decode 仍 10.13）。详见 `PERF-CAPACITY.md` §3/§4。

### 归因更正（本版推翻的旧说法，共三条）
- pre-r1 记「126K decode 23.5」、pre-r2 初稿记「126K 真值 10.1，23.5 不可复现」——**两条都不对**。
  真值是 **22.9 t/s @ `--ctx 126464`**；10.1 来自把 `--ctx` 开成 131072（多开 4,743 token 越过预算）。
  「有时 23.5 有时 10」的真相是当天预算是否刚好容得下 13.67 GiB。
- pre-r2 初稿记「96K 真实上下文只有 11.8 t/s」——真值 **25.2（AR）/ 34.0（MTP dw2）**，
  同样只是把 `--ctx` 从 131072 收到 98304。
- pre-r2 初稿记「48K 以上 MTP 净亏」——**错**。MTP 的固定开销是 +0.57 GB
  （mtp 权重 348 MB + draft state 190 MB + verify 图 73 MB），只要不越预算就是净赚：
  48K +16%（34.9 vs 30.1）、96K **+35%**（34.0 vs 25.2）；
  126K 因为提交 14.12 GiB 越线，才退化成 10.6 t/s（接受率仍是 2/2 全中）。

### 测量纪律（本版新增硬规矩）
- **跨时段的性能数字不可比。** 驻留预算当天会漂移约 1 GiB，任何引擎（包括 llama.cpp）都可能在某次测量里被降级。
  本版早期的 llama 基线（65K 11.47、126K 7.15）就是预算偏低时测的，llama 自己也被挤出去了，虚低 43%~56%。
- 修正后（同窗口背靠背重测）：llama 65,135 tok → **20.17** t/s，126,294 tok → **16.19** t/s（`-c 126464`；
  `-c 131072` 时 16.32，说明 llama 不需要收紧 `-c`，因为它固定开销小）。
  我们同窗口：65K **28.30**（MTP dw2 **41.58**）、126K **22.95**。
- ⇒ 对外口径改为 **decode 领先约 1.4×（三档一致）+ MTP 后中长上下文 2 倍级**，
  原「126K 3.2× llama」的说法**作废**（那是拿我们的贴紧值比 llama 的降级值）。
- 规矩：做对照必须两只引擎在同一时段、同一批作业里连续跑；否则结论无效。

### 新增诊断能力（默认全部关闭，不影响生产）
- `mini_main` 外壳：`[mem]` 引擎自己的显存账本（weights/seq/ws/graph/slack/avail_*）、
  `[vram]` 真实 `cudaMemGetInfo`、`[xfer]` 64MB D2D 带宽探针、`NINFER_DUMMY_MB`（后置牺牲品）、
  `NINFER_PREALLOC_MB`（前置占用）、`NINFER_STEP_T`（每 token 墙钟）。
- `NINFER_ALLOC_BW`（按 buffer 的读带宽探针，**默认关**；在页被写之前跑，数字不能当驻留证据）、
  `NINFER_VRAM_CEILING=pre|post|both`（alloc-until-fail 探针，证明 cudaMalloc 是虚拟超额提交）、
  `NINFER_ALLOC_LOG`（逐次分配日志）。
- `NINFER_MT_SPLITS`（multi-token verify 的 attention 分片数，独立于 decode 的 `NINFER_SIMT_SPLITS`）。
- `NINFER_KVCAP_DIAG`（放宽 `--kvcap` 必须等于 `--ctx` 的两道校验，仅用于实验；第三道 curve 校验仍会拦）。
- `NINFER_KV_DUMP` / `NINFER_KV_DUMP_SEQ`（按 append 序号 dump KV 池，用于逐字节对拍）。

### 行为改动
- `core/arena.cu` 的分配探针默认关（`NINFER_ALLOC_BW` 从「默认开」改成「默认关」）。
- multi-token verify 已是**批量**入口（`grid.z == tokens` 一次发全部草稿 token，`NINFER_MT_BATCH=0` 可回退 per-token 循环）。
  pre-r1 的 PATCHLIST 记「仍是 per-token 循环」已过时。

### 否定结论（补充，避免重复投入）
- decode 慢与以下全部无关：热/功耗节流（三次重负载后立刻跑仍 30.09）、splits（各档都是 32）、
  零初始化 2.35 GB memset（`NINFER_KV_ZERO_INIT=0` 同速）、额外占压（后置 1 GiB 牺牲品不改变快态）、
  卡的健康（慢态下 64MB D2D 565-852 GB/s，比快态还高）、二进制版本（旧 exe 同样给 10.12）。
- `NINFER_PROFILE_DECODE` 的 cudaEvent 分段表不可信（分项累加 ~7.25 ms/step，比已知的 GDN 单项还小 6 倍）；
  可靠的只有同文件的 `[prof-decode] N steps: layers X / norm+lmhead Y / step-total Z`。
- `--kvcap` 与 `--ctx` 无法解耦（第三道 `kv_capacity` curve 校验），所以「大 envelope + 小池」的实验路线走不通。

### 未修（技术债）
- 126K + MTP：差 0.5 GB 预算（提交 14.12 GiB）。可试方向：verify graph 73 MB 复用、draft state 按 draft window 分配。
- int8 KV 非确定性；fp8 的 QK 是 `ninfer/src/ops/common/mma.cuh:584-590` 的 `__builtin_trap()` 桩；
  dflash2 坏路径；256K 需 KV offload；prefill ≥64K 落后 llama 1.7-1.9×。

## 0.1.0-pre-r1 — 2026-10-30

### 正确性（关键）
- **修 decode attention 的 partial max 双重 scale**：`decode_t1_simt.cuh` 发布 `partial_m` 时把已在
  scaled 域的 `m[]` 又乘了一次 `scale`，导致跨 split 合并权重被压平成近均匀。
  → `partial_m[stat] = gmax_v[r];`（warp 局部改写后沿用同一修复）。
- **修在线 softmax 归一化域错乱**：批内指数归一到 `m_upd`、累加器在运行最大域，缺 `beta`。
  warp 局部改写后每个 warp 维护自己的 `(m,l,o)`，合并时统一用 `exp(m_w - gmax)`，域一致。
- 结果：4K/8K/16K/23K/32K(bf16+int4)/48K/96K/**126K** 从「抄错字符 / 拒绝回答 / 乱码」变成全部命中。
- 顺带消除 32K int4 偶发 `hipErrorLaunchFailure`。

### 性能
- **warp 局部在线 softmax**：每 warp 独占一段 key，热循环零屏障；原来每 8 key 一次整块 softmax（3 个 `__syncthreads()`）。
- **int4 默认 2-row CTA**（`simt_group` 按 KV 格式自动选）：48K +8.3%、96K +8.8%。
- int4 读路径向量化（lane 拥有连续 8 dim）、V 行预取、witness ALU 门控：均在噪声内，保留但无收益。
- **MTP 复活**（短上下文 +47%）。
- int4 prefill O(n²) → O(n)（171 → 698 t/s）；prefill output-head scratch 按 1 列预留（省 1.02 GiB，126 → 494 t/s）。

### 否定结论（写进文档避免重复投入）
- decode 分片数（4/8/11 CTAs per SM）、`__launch_bounds__` minBlocks 1/2/4、4-key 展开 + forced-live、
  多列 GEMV 行流 4→2 + 深展开 —— 全部零收益。
- `NINFER_SKIP_MLP` / `NINFER_SKIP_GDN` 在 MTP 路径基本是 no-op，不能用来做消融。
- `NINFER_ATTN_XCHECK`（SIMT vs SmallT 对拍）不能当 oracle（两内核同一次调用会重复 append KV）。
