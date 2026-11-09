# NInfer-HIP `0.1.0-pre-r8`

> 把 NInfer 推理引擎推到 AMD RDNA4（gfx1201）—— 一张 16GB 的 RX 9070 XT 上，
> 27B 模型跑到 **126K 真实上下文**（decode 20.7 t/s · prefill 364），短上下文开 MTP dw2 到 **74.6 t/s**。
> 同卡同权重下 decode 约为 llama.cpp 的 **1.4–2.5x**（开 MTP 约 2x）；prefill 在 ≥64K 落后于 llama.cpp（0.64x / 0.55x）。
>
> **0.1.0-pre-r8 的新东西：KV 记忆（KV memory）。** 显存窗口装不下的老上下文不再被丢弃——
> 滑出窗口的 KV 页会归档到内存，等 prompt 读完、问题已知时再挑出相关的一段换回显存继续参与注意力。
> 96K 长文里埋在 25% / 50% / 75% 深度的答案，从「答不出来」变成「逐 token 正确」。详见 [§2](#2-kv-记忆显存装不下的上下文捞回来)。

**为什么会有这个项目、这一个月里发生了什么 → [docs/STORY.md](docs/STORY.md)**（一个普通 AMD 用户的记录）

**预编译二进制 → [Releases](https://github.com/Newaiguy/NInfer-HIP/releases)**
（仓库不放二进制：exe 191 MB，超过 GitHub 单文件 100 MB 限制；模型权重也不在仓库内）

NInfer 引擎（原生 CUDA 项目，Apache-2.0）到 **AMD RDNA4 / gfx1201** 的 HIP 移植分支。
目标机型：RX 9070 XT 16GB（256-bit GDDR6，标称 640 GB/s），Windows 11 26200 + HIP SDK 7.2.3。
模型：`WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3`（官方 v3 单文件 artifact，15,017,456,896 B，
内含 text + vision + mtp + dflash2 四个组件，本版本只用 text 与 mtp）。

## 0. 这一版（pre-r8）更新了什么

一句话：**给引擎加了「KV 记忆」——长上下文里被滑出显存的那部分，可以按问题相关性捞回来。**

1. **KV 归档（KV archive）**：窗口每往前滑一次，即将被丢弃的 KV 页先 DMA 到主机内存的一块
   arena 里（按 token 区间连续存放，满了按 FIFO 淘汰）。归档进去的是**完整的 K+V**，不是摘要。
2. **页级相关性检索**：prefill 过程中顺手为每个归档页算一份「K 均值摘要」（16 个全注意力层 ×
   4 个 KV 头 × 256 维，bf16）；prompt 读完、拿到最后一个 query 之后，用 query 与全部候选页打分，
   每个 (层, 头) 单元各自标准化后取最大，得到页相关性排序。
3. **按峰值取连续段换回显存**：以排名第一的页为中心取 **112 页（7168 token）连续区间**，
   从归档里 H2D 拷回显存——正好落在「下一次滑移本来就要丢弃的槽位」上，**不额外占显存容量**。
4. **重放最后一个 chunk**：召回发生在 prefill 结束之后，而最后一个 chunk 的注意力已经算完了，
   所以把最后那几 token（实测 4 个）重算一遍，让问题真正「看」到召回的内容。
5. **四个启动参数上命令行**：`--resident-tokens` / `--sink-tokens` / `--archive-mib` / `--kv-retrieval`，
   不再需要设环境变量（见 §2.3）。不传 `--kv-retrieval` 时行为与 pre-r7 **完全一致**。

## 1. 状态总览（本版本实测）

全部开 CUDA graph、KV 格式 `--kv rk4v4-e8`（RotatedInt4KeyInt4ValueE8）、`--max-concurrency 1`、
单张 RX 9070 XT 16GB。完整命令见 [`docs/VALIDATION.md`](docs/VALIDATION.md)。

| 项目 | 状态 |
|---|---|
| **KV 记忆（本版新增）** | ✅ 归档 + 检索召回；96K 长文 needle 深度 25/50/75% 全部命中；92% 深度（无需召回）与关闭检索逐 token 相同 |
| **交互式聊天** | ✅ `--chat`：流式输出、多轮（前缀复用）、`/think /stats /clear /system /exit`；`harness/launchers/chat.bat` 双击启动 |
| 正确性 29 token ~ 126K token | ✅ 全部通过（needle 逐字符命中、数字列表、长文摘要语义正确） |
| **126K 真实上下文** | ✅ **decode 20.7 t/s · prefill 364**（`--ctx 131072`，needle `K8M2Q9` 命中） |
| 96K 真实上下文 | ✅ decode **25.3**（AR）· prefill 422 |
| 短上下文 decode（AR，宽 1） | 37.5 @29 token · 36.5 @2.8K · 35.7 @5.8K · 32.4 @32K |
| 中长上下文 decode（AR） | 30.2 @48K · 28.6 @65K · 25.3 @96K · **20.7 @126K**（均要求 `--ctx` 贴紧） |
| **MTP 推测解码（dw2）** | **74.6 @2.8K（+100%）** · 65.0 @24K（+100%） · 41.5 @131072 容量（真实 2.8K） · ❌真实上下文 ≥100K |
| prefill | **948 @5.8K · 743 @32K · 633 @48K · 549 @65K · 422 @96K · 364 @126K** |
| int4 KV | ✅ 16GB 上跑 128K 的唯一容量路径 |
| int8 KV | ❌ 不可用（非确定性） |
| fp8 KV | ❌ 不可用（QK 的 MMA 是未移植的 trap 桩；容量也只到 ~96K） |
| dflash2 | ⚠ 已修好（输出与 AR 逐 token 相同）但**只有 12.8–14.9 t/s**，约为 MTP 的 1/6，未启用 |
| 256K | ❌ 直接塞不下（int4 KV 8.86GB + 权重 11.77GB > 16GB）；KV 记忆提供的是「长文可读」而非「显存装得下」 |
| **多模态（图像输入）** | ❌ **本版不支持**（artifact 内含 vision 组件、引擎也编入了 vision 通路，但本版的 harness 没有图像入口；下一版接） |
| 并发 | ❌ 只支持 1（`--max-concurrency 1`） |

## 2. KV 记忆：显存装不下的上下文，捞回来

### 2.1 要解决什么问题

16GB 的卡，27B 权重占 11.77 GB，int4 KV 每 token 17,920 B。想跑 126K 上下文就要 8.86 GB KV —— **装不下**。
引擎的做法是**滑动窗口**：只把「最近的 N 个 token」+「最前面 2048 个 token（attention sink）」留在显存里，
更早的页滑出去、直接释放。

代价是：**窗外的内容真的没了**。把答案埋在 96K 长文的 25% 深度，问「那是什么」——模型看不到那段，
只能编。`0.1.0-pre-r8` 之前就是这样。

KV 记忆做的事：**滑出去的页不扔，先存到主机内存；读完整段 prompt、知道问题是什么之后，
按相关性把最可能相关的一小段换回显存，再让问题读一次。**

### 2.2 怎么实现的

整条流水线横跨 4 个文件（`kv_archive` + `kv_retrieval` + `context` + `prefill`/`text`）：

**① 归档 —— 滑移的同一处，D2H 拷一份**

`context.cpp` 的 `ensure_kv_window()` 每次要滑窗时，先算出「这次会丢哪些页」，把它们的
`DeviceKVPageHandle` 收集起来，`pool.copy_to_host()` 到主机侧 arena。每个归档块记录
`begin_token` / `pages`，页区间严格递增、无重叠（这点被两次实测校验过：块 86 == 滑移 86 次，
归档 90048 token + 常驻 6119 + sink 2048 = 98215，账是平的）。

arena 大小 `--archive-mib`（默认 2048 MiB，够存 ~90K token 的 int4 KV）。满了按 FIFO 淘汰，
淘汰前 `cudaStreamSynchronize`，因为 D2H 可能还在飞。

**② 摘要 —— prefill 时顺手统计，不额外读一遍 KV**

每个全注意力层（共 16 层，索引 3,7,…,63）在 `text.cpp` 的 `attn_mix()` 里，于 RMSNorm 之后、
RoPE 之前采集 `kn` / `qn`，用 kernel 把这一页 64 个 token 的 K 按页求均值
（每页 16 层 × 4 KV 头 × 256 维 bf16 = 128 KiB）。query 同理，但只用最后一个 token。

**③ 打分 —— (层, 头) 各自标准化后取最大**

页均值和 query 点积出 64 个 (层, 头) 分数；每个单元跨全部候选做 z-score，一页取 64 个单元的**最大值**。
这一步是从「#322 名」提到「#35 名」的关键：平均掉 64 个单元会把信号冲淡，取最大保留了
「某一层某一头确实在看这段」的证据。

> 这段路走了 4 个弯路才通：页均值直接点积（分数全等）、余弦归一化、逐层 z-score、
> 候选均值中心化 —— 单独做都没用；最后是「逐 (层,头) 取最大」才把 needle 区从 #322 拉到 #35。
> 已知页均值打分对 needle 的定位误差约 ±8 页，所以下面取的是**连续段**而不是精确页。

**④ 召回 —— 用「本来就要丢的槽位」装，零额外容量**

以 top-1 页为中心取 112 页（7168 token）连续区间，`copy_from_host()` 拷回
`[sink_pages, sink_pages + 112)` 这段槽位 —— 它正好是下一次滑移要释放的位置。
所以召回**不增加显存占用**，只是把「要丢的」换成「更有用的」。

**⑤ 重放最后一个 chunk —— 因为召回发生在 prefill 之后**

这是最容易被忽略的一环，也是最后一关：自动选块版一开始**明明找回对了页、答案仍然是错的**，
因为召回是 prefill 结束后才做的，而最后一个 chunk 的注意力已经算完了。把最后一个 chunk
（实测只有 4 个 token）重算一遍，问题才真正看到召回内容。开关 `NINFER_KV_REPLAY=0` 可关。

### 2.3 怎么用

**四个命令行参数**（在 exe 和 artifact 之后）：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--resident-tokens N` | 常驻窗口大小：最近 N 个 token 始终留在显存 | 20480 |
| `--sink-tokens N` | attention sink：最前面 N 个 token 始终保留（不能是 0） | 2048 |
| `--archive-mib N` | 主机侧归档 arena 大小（MiB）；**给这个参数会自动打开归档** | 2048 |
| `--kv-retrieval 1` | 打开检索召回（不开就是纯归档，行为与 pre-r7 等价） | 0（关） |
| `--kv-recall-pages N` | 召回段宽度（页，1 页 = 64 token）；112 页 = 7168 token | 112 |

`--kv-retrieval 1` 会**自动打开归档**，并按 `--resident-tokens` 推算 arena 大小
（`2 × resident × 17920 B`，夹在 1024–4096 MiB）：20480→1024、32768→1120、
65536→2240、98304→3360 MiB。显式给 `--archive-mib` 则以后者为准。

一条完整的长文命令：

```bat
ninfer_mini_nordc.exe ^
    E:\ninfer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer ^
    --chat --kv rk4v4-e8 --ctx 262144 ^
    --resident-tokens 20480 --sink-tokens 2048 --archive-mib 4096 --kv-retrieval 1
```

> `--ctx 262144` 是给 int4 KV 池留的**逻辑容量**；实际驻留的显存由 `--resident-tokens` 控制，
> 所以这个值开大不会像 §3 那样掉速（§3 说的是 KV 池按 `--ctx` 一次性提交的老行为）。

**等价的传统环境变量**（参数内部就是回写这些变量，两个都认）：

| 命令行 | 环境变量 |
|---|---|
| `--resident-tokens` | `NINFER_RESIDENT_TOKENS` |
| `--sink-tokens` | `NINFER_SINK_TOKENS` |
| `--archive-mib` | `NINFER_KV_ARCHIVE_MIB`（并自动置 `NINFER_KV_ARCHIVE=1`） |
| `--kv-retrieval` | `NINFER_KV_RETRIEVE` |

**调优用（只有需要时设）**：`NINFER_KV_RECALL_PAGES`（召回页数，默认 112）、
`NINFER_KV_RECALL_BACKOFF`（top-1 页之前的回退页数，默认 56 = 对称取段）、
`NINFER_KV_REPLAY=0`（关闭最后 chunk 重放，用于对照实验）。

**看它干活**（诊断都走 stderr，聊天模式下进 `ninfer-chat.log`）：

| 变量 | 输出 |
|---|---|
| `NINFER_WINDOW_TRACE=1` | `[win] slide -> base=...`、`[win] archive N pages from ...` |
| `NINFER_KV_ARCHIVE_TRACE=1` | `[kv-archive] block k: N pages, tokens [a,b), ...` |
| `NINFER_KV_RETRIEVE_TRACE=1` | `[kv-rank] blocks=/pages=/candidates= top: ...`、`[kv-recall] 112 pages from token X -> slots [a,b)` |
| `NINFER_KV_REPLAY` | `[kv-replay] replayed N tokens from M` |

### 2.4 最佳配置

**短上下文（≤32K）别开**：常驻窗口 20480 + sink 2048 已经装得下，开归档只是白占内存。

**长文（≥48K，问题在末尾）用这套**：

```bat
--kv rk4v4-e8 --ctx 262144 --resident-tokens 20480 --sink-tokens 2048 --archive-mib 4096 --kv-retrieval 1
```

- `--archive-mib 4096`：96K 正文全部归档只需 ~1.7 GB，给 4096 是留余量，避免 FIFO 淘汰把要用的页淘汰掉。
- `--resident-tokens 20480` 是显存与命中率的折中：实测 96K 正文下 needle 在 75% 深度也能覆盖，
  92% 深度本来就在常驻窗口里、根本不需要召回。
- 想再省显存可以把 `--resident-tokens` 降到 12288，代价是召回频率变高、prefill 期间多几次 D2H。

**同时要 MTP 时**：`--spec mtp --draft-tokens 2` 照旧，KV 记忆与 MTP 互不影响。

**档位怎么选（实测于 RX 9070 XT 16 GB）**

| 真实上下文 | 推荐窗口 | 归档 | 说明 |
|---|---|---|---|
| ≤32K | 不开 | 不开 | 常驻就够 |
| 48K–98K | `--resident-tokens 32768` | `--archive-mib 3072` | 98K 实测逐 token 正确 |
| 128K–192K | `--resident-tokens 32768` | `--archive-mib 3072` | 192K（196,427 token）实测逐 token 正确 |
| ≥256K | 32768 起步 | 4096（FIFO 淘汰） | 归档装不下全量，靠淘汰兜 |

**窗口不要低于 32768**：192K 语料上开到 16384 时，模型会先复读噪声、再把答案补在后半段，ID 串不可用；
32768 是这套召回（112 页 ≈ 7168 token）能兜住 needle 邻域的下限。窗口显存按 int4 `17,920 B/token` 估算：
32768 → 0.59 GB、65536 → 1.17 GB，再叠加 MTP 草稿行就贴边（本卡权重已占 12.5 GB）。

### 2.5 实测效果

**主证据：召回档位（`tasks_needle_mid_q.txt`，98,218 token，答案埋在 ~48% 处，页区间 [768,800)）**

| `--resident-tokens` | 窗口覆盖 | needle 在窗口内？ | 输出 |
|---|---|---|---|
| 20480 | 320 页 | 否 —— **纯靠召回** | `45 669 24312 10846 8006 5434 12 19 19 22 16 248046` |
| 32768 | 512 页 | 否 —— **纯靠召回** | 同上，与 GT **逐 token 相同** |
| 65536 | 1024 页 | 是 | 同上（不引入回归） |

命令只给参数、不设任何环境变量：`--kv rk4v4-e8 --ctx 262144 --resident-tokens 32768 --kv-retrieval 1`。
日志：`[kv-rank] ... needle 页名次 #8、峰值页 760`、`[kv-recall] 112 pages from token 45056`、
`[kv-replay] replayed 4 tokens from 98178`。

**旁证：96K 正文 × 四个 needle 深度**（`tasks_needle_q25/q50/q75/q92.txt`，按口令字符串命中判定）

| needle 深度 | 检索关 | 检索开 |
|---|---|---|
| 25% | ❌ 编造 | ✅ 口令命中 |
| 50% | ❌ 编造 | ✅ 命中 |
| 75% | ❌ 编造 | ✅ 命中 |
| 92% | ✅ 命中 | ✅ 命中（与关闭时逐 token 相同） |

> 这四份语料在 needle 之后还夹了一行「列出 1..400 的整数」的指令，模型有时会转去答那道题，
> 所以它们**不适合**当命中判据；判召回是否成功请用上面的 mid 语料（GT token 串已知）。

四格 GATE 回归（8K / 96K / MTP，检索默认关闭路径）：
`8k decode 35.63 · prefill 931.4` / `96k decode 24.90 · prefill 422.1` / `mtp decode 80.21 · prefill 842.2`，
全部在容差内（对照 pre-r7：8K 35.72 / 96K 25.33 / MTP 78.50）。

归档本身的开销（同一份 98K prompt，开/关归档各跑两轮）：输出逐 token 相同，
prefill `801.997 → 800.446`、decode `33.14 → 33.62` —— **在噪声内**。

**192K 长文：不开窗口根本打不开，开了就逐 token 正确**

- 全上下文跑不动（不是我们的改动导致）：引擎按整段 KV 预留，128K 就要 5.27 GB > 可用 3.81 GB，
  直接报 `the largest capacity this device can hold is about 189,646 tokens or less`。
- 窗口路径可以：`--ctx 262144 --resident-tokens 32768 --archive-mib 3072 --kv-retrieval 1 --spec mtp --draft-tokens 2`，
  196,427 token 的 prompt **输出与 GT 逐 token 相同**（`45 669 24312 10846 8006 5434 12 19 19 22 16 248046`、
  明文 `NINFER-ZEBRA-4471`），prefill **535.7**、decode **33.8**。

**与 llama.cpp kvmem 的对比（同机、同模型、同量化，两边都开 MTP2）**

| prompt | 上游 kvmem prefill / decode | 本引擎 prefill / decode |
|---|---|---|
| 16K | 762.9 / 34.56 | **799.7 / 52.9** |
| 32K | 633.3 / 35.69 | **806.9 / 37.5** |
| 64K | 454.0 / 35.16 | **693.8 / 30.6** |
| 98K | 380.4 / 13.98 | **556.0 / 35.4** |
| 192K | 336.8 / 22.27 | **535.7 / 33.8** |

上游参数：`-c 262144 --kvmem --kvmem-budget 32768 --kvmem-gen-reserve 32768 --kvmem-block-tokens 128
-ctk q8_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 2`；本引擎为上面那套窗口+召回配置。
192K 档 prefill **+59%**、decode **+52%**；本引擎的 prefill 从 98K 到 192K 几乎不变（535 vs 556），
因为映射量由窗口决定、与上下文长度解耦。

### 2.6 代价与限制（先看这段再开）

- **吃内存**：归档存在**主机内存**，最多 `--archive-mib`。96K 正文实测 ~1.7 GB。
- **吃显存**：页摘要 64 KiB/页，1311 页 ≈ 82 MB（另有 2048 槽环形缓冲，128 MiB）。这是**额外的**显存开销。
- **prefill 变慢一点点**：做过一次 112 页（~122 MiB）的 H2D 召回 + 重放 4 个 token，实测在噪声内。
- **召回是启发式，不是精确注意力**：页均值打分定位误差约 ±8 页，所以取 112 页连续段兜住。
  只在这套 needle 语料上验证过；如果答案需要精确到几十个 token 的定位、或者问题不在 prompt 末尾，
  命中率没有保证。
- **必须读完 prompt 才能召回**：query 来自 prompt 最后一个 token，所以「边读边召回」不支持。
- **环境**：只在 RX 9070 XT + Windows 11 + HIP SDK 7.2.3 上验证，别的卡没测。

## 3. 最重要的一条：`--ctx` 必须贴着实际 prompt 长度

> 这条针对**不开 KV 记忆**的老路径：KV 池按 `--ctx` 一次性全量提交。
> 开了 `--resident-tokens` 的窗口模式后驻留量由窗口决定，不受这条约束。

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

## 4. 推荐配置（本引擎目前的最优参数组合）

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
| 长文 + KV 记忆 | `--kv rk4v4-e8 --ctx 262144 --resident-tokens 20480 --sink-tokens 2048 --archive-mib 4096 --kv-retrieval 1` | 见 §2.5 |
| 想开大容量但对话很短 | `--ctx 81920 --spec mtp --draft-tokens 2` | **56.0**（真实 2.8K） |

四条硬规则：

1. **`--ctx` = prompt + max_new，绝不多开。** 多开 4.7K 就从 22.9 掉到 10.1。
2. **`--draft-tokens 2` 是唯一划算档位。** `3` 以上草稿链自回归退化，实测净亏；`1` 只有 ~1.3×。
3. **MTP 的内存开销是固定的 +0.57 GB**（mtp 权重 348 MB + draft state 190 MB + verify 图 73 MB），
   所以 MTP 的有效上限由**提交预算**决定：今天 96K 能跑（13.62 GiB，34.0 t/s）、126K 不能（14.12 GiB，10.6 t/s）。
   真实上下文 ≥100K 时直接关 MTP 用 AR。
4. **真实上下文越长，MTP 倍率越低**（2.15× @2.8K → 1.35× @96K），但直到 96K 都仍是净赚。

prefill chunk 现在**自动分档**：`--ctx` ≤65536 用 2048、>65536 用 1024（相对 1024 全局快 +10~15%）。
仍可用 `NINFER_PREFILL_CHUNK` 覆盖；4096 不再更快，且 `--ctx` ≥69632 时会 OOM。

## 5. 快速开始

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

**想试 KV 记忆**：选 `[5] custom arguments`，在里面加
`--resident-tokens 20480 --sink-tokens 2048 --archive-mib 4096 --kv-retrieval 1`。

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

REM 3) 长文 + KV 记忆（96K 正文，答案埋在深处）
python probe_rebuilt\run_capture.py build\infer_mini_nordc.exe ^
    E:\ninfer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer ^
    --tasks tasks\tasks_needle_q50.txt --ctx 262144 --max-new 16 --kv rk4v4-e8 ^
    --resident-tokens 20480 --sink-tokens 2048 --archive-mib 4096 --kv-retrieval 1
```

- **`--kv rk4v4-e8` 是 128K 的必需品**（bf16 KV 需 9.99 GB、int8 KV 需 5.83 GB，而运行时可分配只有 5.16 GB）。
- 输出读法：`run_capture.py` 结束才整段输出；日志请用 Python 以 UTF-8 读（PowerShell 控制台是 GBK，
  中文会出现二次编码假象；`| Out-File` / `*>` 默认写 UTF-16LE）。
- `tasks` 文件格式：首行必须 `###ID <name>`；**正文内任何以 `#` 开头的行都会被当成新任务分隔符**；
  文件必须是无 BOM 的 UTF-8。

## 6. 已知坑与硬限制

- **`--ctx` 开大了就掉速**（§3）。不开 KV 记忆时这是主要性能缺陷，规避手段只有「按实际长度设 ctx」。
- **KV 记忆是启发式召回**（§2.6）：定位误差 ±8 页、要求问题在 prompt 末尾、必须重放最后 chunk。
  它提升的是「长文里找信息」的命中率，不是无损上下文扩展。
- **MTP 到 96K 为止**：126K 加 MTP 会多提交 0.57 GB 越过预算 ⇒ 10.6 t/s（比 AR 还慢）。≥100K 真实上下文请关 MTP。
- **`--draft-tokens` 用 2**：3 以上草稿链自回归退化，每轮实际接受数掉下来。
- `--no-graph` + `NINFER_PROFILE_MIXER=1` + `--spec mtp` 会挂死 GPU 队列；`NINFER_PROFILE_DECODE` 必须配 `--no-graph`。
- **绝不对卡在 GPU 上的引擎进程 `Stop-Process`**：杀进程不会终止已在 GPU 上跑的 kernel，
  死循环 kernel 会把显示卡死（2026-10-30 已发生过一次黑屏硬复位）。要么等，要么重启机器。
- int8 / fp8 / dflash2 / 256K / 并发 > 1 均不可用（见 §1 与 [`docs/PATCHLIST.md`](docs/PATCHLIST.md) 末节）。

## 7. 目录

```
（引擎 exe 不在仓库里：从 Releases 下载 ninfer_mini_nordc.exe，191,691,776 B，
          sha256 见 VERSION.txt；把它和 harness/launchers/chat.bat 放同一个文件夹即可双击开聊）
src/      infer/{include,src,third_party} + shim（HIP 兼容层），全量可重建源码
          新增 kv_archive.{h,cpp} / kv_retrieval.{h,cu}（KV 记忆）
harness/  构建与跑测工具（compile_one / swap_lib / link_exe / run_capture）
          harness/launchers/chat.bat  双击启动器（+ model.txt.example）
tasks/    验收语料（含 126K needle 语料、96K 多深度 needle 语料 q25/q50/q75/q92）
docs/     STORY / VALIDATION / PERF-CAPACITY / KVMEM / CHANGELOG / PATCHLIST / BUILD
VERSION.txt  版本、实测指标、引擎 sha256、环境变量与纪律
```

构建方式见 [`docs/BUILD.md`](docs/BUILD.md)；每个改动的回滚点见 [`docs/PATCHLIST.md`](docs/PATCHLIST.md)；
KV 记忆的实现细节与全部实测数据见 [`docs/KVMEM.md`](docs/KVMEM.md)。
