# KV 记忆（KV archive + retrieval recall）实现说明

> 适用版本：`0.1.0-pre-r8`。这份文档讲清楚四件事：**为什么做、怎么实现、怎么调、实测到哪一步**。
> 面向想读代码 / 想改参数 / 想判断它能不能用在自己场景里的人。

## 1. 问题

RX 9070 XT 只有 16 GB。27B 的 IQ3_S 权重 11.77 GB，int4 KV（`rk4v4-e8`）每 token 17,920 B。
126K 上下文需要 8.86 GB KV —— 装不下，所以引擎用**滑动窗口**：

- **sink**：最前面 `NINFER_SINK_TOKENS`（默认 2048）个 token 永远保留（attention sink）；
- **resident window**：最近 `NINFER_RESIDENT_TOKENS`（默认 20480）个 token 始终驻留；
- 中间的页滑出窗口后**直接释放**。

后果很直接：读一份 96K 的文档，答案在第 25% 处 —— 那段 KV 早就被扔了，模型只能编。
`pre-r8` 之前无解。

## 2. 总体思路

滑出去的内容先在**主机内存**里留一份；等 prompt 读完、拿到最后一个 query（也就是「问题」），
再用这个 query 去和所有归档页打分，把最相关的一小段**换回显存**，然后让问题再读一次。

三个约束决定了实现形态：

1. **不能改内核**。K 写进 cache 时已经按绝对位置做过 RoPE，注意力内核用的 `key_pos` 只参与因果掩码，
   不参与旋转。所以召回块放在任何「比 query 更早」的槽位上，算出来的注意力分数和它原本在
   全上下文里的分数一致 —— 不需要改页表、不需要改位置映射。
2. **不能多占显存**。召回段落在 `[sink_pages, sink_pages + n)`，这正是下一次滑移本来要丢弃的槽位。
   容量代价为零。
3. **不能影响不开这个特性的路径**。所有逻辑挂在 `NINFER_KV_ARCHIVE` / `NINFER_KV_RETRIEVE` 之后，
   默认关。

## 3. 流水线

```
prefill chunk 循环
  ├─ attn_mix(): 每层采集 pre-RoPE K/Q → 页均值摘要（128 KiB/页，bf16）
  ├─ ensure_kv_window(): 需要滑窗时
  │      ├─ 收集这次要丢的页 → copy_to_host → KVArchive（主机 arena）
  │      └─ slide_prefix_pages()
  ├─ （最后一个 chunk 结束）freeze_query() → prompt_complete = true
  └─ 召回：rank() → 取 top-1 页 → 连续 112 页 → copy_from_host → 槽位 [32,144)
       └─ 重放最后一个 chunk（几 token），让问题真正看到召回内容
```

### 3.1 归档 `src/infer/src/models/qwen3_5/program/storage/kv_archive.{h,cpp}`

- `KVArchive::instance()` 单例；`enabled()` 看 `NINFER_KV_ARCHIVE`。
- 懒建 `HostKVArena`（大小 `NINFER_KV_ARCHIVE_MIB`，默认 2048 MiB）。
  建失败只打印 `[kv-archive] arena init failed:` 并跳过，不影响推理。
- `archive(pool, pages, first_absolute_token, stream)`：把 `pages` 从显存 `copy_to_host` 到 arena，
  记一个 `Block{ allocation, begin_token, pages }`。
- 满了 `while (!arena_->can_allocate(...) && !blocks_.empty())` 按 FIFO 淘汰；
  淘汰前 `cudaStreamSynchronize(stream)`（D2H 可能还在飞）。
- 挂接点：`context.cpp` 的 `ProgramImpl::ensure_kv_window()`，在 `slide_prefix_pages()` **之前**。

**踩过的坑（留着当警戒）**：`first_absolute_token` 一开始按「已滑走的 token 数」算，
忘了尾部真实起点要加 sink，结果块 0 `[2048,5120)` 和块 1 `[3072,4096)` **重叠 2048 token**。
正确公式是 `first_absolute = current_base + sink_pages * 64`。
另有一次召回在第一次换入后把归档冻结了（`skip_pages` 把 `drop_pages` 吃干净），
后来用「先驱逐再填充 + high-water mark 前缀过滤」修掉。

### 3.2 页摘要 `kv_retrieval.{h,cu}`

- 16 个全注意力层（索引 3,7,…,63），每层 4 个 KV 头 × 256 维。
- 每页 64 token，摘要 = 整页 pre-RoPE K 行的均值；`kRetrievalSubChunks = 1`（子块取最大的
  变体保留在代码里但编译期关闭，归档记录仍按 3 个子块槽位预留，当前只用第 0 个）。
- 设备侧环形缓冲 `kRetrievalRingPages = 2048` 槽 × 64 KiB（fp32）= **128 MiB**；每页交给归档的
  bf16 记录是 96 KiB。环形缓冲必须 ≥ 单个 chunk 的最大跨页数，否则槽碰撞会让**所有候选分数严格
  相等**（= `1/候选数`，看到这个数就是摘要全零的指纹）。
- 采集点：`text.cpp` 的 `attn_mix()`，`qn`/`kn` 经 RMSNorm 之后、`text_rope` 之前 —— 摘要因此
  与位置无关，可以在任意槽位上和未来的 query 比较。
- query：prompt 读完后冻结（`freeze_query()`），只用最后一个 token；召回打分用的是「问题」，
  不是当时在飞的那一块。

### 3.3 打分 `rank()` → 两阶段精排

- kernel 输出 **64 个单元**（16 层 × 4 头）的分数。
- host 侧每个单元跨全部候选做 z-score，然后**每页取 64 个单元的最大值**。
- 这一步是整个特性能不能用的分水岭。走过的弯路全记在下面 §5。
- **精排（`NINFER_KV_RERANK`，默认开）**：页均值只是粗筛，所以把排名前 256 的候选页从归档里
  取出 `rk4v4-e8` 量化 K，在 host 反量化后与「按真实位置旋转过的 query」逐页收缩（`score_page_keys()`，
  与设备侧同一套数学），排除 sink 页（< 32 页，模板和题目本身对 query 匹配度很高但不含答案）。
- 实测（98,218 token 中深度 needle 语料）：粗筛峰值页 760（z = 0.689），精排前后 top-1 不变
  （760 @ 0.689），needle 区 [768,800) 的首个高分页是 #18 的 773。也就是说**精排在这份语料上
  不移动峰值**，它是一条稳健性保险，不是消偏手段 —— 消偏靠下一节的「以峰值为中心取整段」。

### 3.4 召回 `context.cpp::recall_into_tail()`

- `want = NINFER_KV_RECALL_PAGES`（默认 **112** 页 = 7168 token）。
- 三条策略，默认是第一条（自适应）：
  - **自适应单段/两段**（默认，`NINFER_KV_RECALL_SPLIT=0` 可关掉第二段）：先看名次表里的第二名峰值，
    只有它**同时**满足「离 top-1 ≥ 28 页」且「得分 ≥ 0.35 × top」才算另一份文档，这时把预算切成
    两段：主峰 `56` 页、次峰 `56` 页。主峰段**前偏**（`backoff = width/4`，吸收实测峰值比文档早
    约 8 页的偏置），次峰段**居中**（`backoff = width/2`，其次峰与文档的相对位置未测）。
    不满足条件时退化成「整段 112 页居中」，与 r9 行为逐 token 等价。
  - **单整段**（强制）：把预算全给 top-1 峰，`begin = peak - slots/2`。r9 的行为。
  - **多峰窄块**（显式 `NINFER_KV_RECALL_DIV>1`）：把预算切成 `slots / DIV` 宽的窄块，按名次绕多个峰值，
    再用 `NINFER_KV_RECALL_FILL`（默认开，仅此模式生效）把剩余预算逐页补满。
  > 反面教训：0.1.1 曾把窄块设成默认（DIV 自动取 2/4），单文档 needle 上块 `[732,787)` 直接丢了
  > 答案文档，输出 `45 10052 248046` 而不是参考 token；整段 `[704,815)` 逐 token 复现。
  > 所以缩窄只能是 opt-in，默认必须是整段。
- 自适应分块的实测依据（`build/_recall_split_check.py`、`build/_recall_real_check.py`）：
  - 98K 单文档 needle：精排 top3 = `760@0.68942 761@0.01035 978@0.00662`，次峰仅 1.5% ⇒ **不切分**，
    召回页仍是 `704..815`，32768 / 20480 两档都逐 token 等于 GT。单文档行为零变化。
  - 双文档语料（ATLAS 在 25%、BOREAL 在 50%，相距 30K token）：top3 = `915@0.23831 449@0.18314`（82%）
    ⇒ 切分，两段合起来覆盖两份文档。问 ATLAS 时：不切分只召回 `859..970`（BOREAL 所在），模型答成
    BOREAL 的事实；切分后答对 `NINFER-ATLAS-1111`。
  - **已知限制**：如果两份文档的口令是近似字符串（`NINFER-<NAME>-<NNNN>`），两段同时在上下文里时
    模型会把一个名字和另一个的数字拼起来（实测答 `NINFER-ATLAS-2222`）。这是模型行为，不是投递
    缺陷；判别语料要用不可混淆的事实（例如一份给口令、一份给日期）。
  - **预算不能加**：`DIV=2` + `PAGES=168`（84+84）反而把 needle 打坏（`45 10052 248046`）。总预算必须
    维持 112 页 —— 分块是「在同等预算内改覆盖形状」，不是「多召回一些」。
- `archive->recall_pages(start * 64, want)` → `pool.copy_from_host()` 写到槽位
  `[sink_pages, sink_pages + 112)`。
- 写之前先把这些槽位原有的页归档（`watermark` 过滤防止重复归档），再更新 watermark。
- 闸门：`archived < want` 时直接返回（窗口还装得下就不召回），并且只有真的换入了页才重放最后 chunk。

### 3.5 重放 `prefill.cpp`

召回发生在 prefill 结束之后，而最后一个 chunk 的注意力已经算完了 —— 这是「页找对了、答案还是错的」
的根因。所以 `staged.cursor == staged.prompt_tokens` 时重跑一次 `prefill_text_chunk()`，
`thread_local bool in_recall_replay` 防重入，`NINFER_KV_REPLAY=0` 可关。

> 用 `remaining == 0` 判定「最后一个 chunk」是错的：**每个** chunk 都满足 ⇒ 无限递归
> （日志会连续打印 `[kv-replay] replayed 1024 tokens from 0/1024/2048`）。

## 4. 参数

| 参数 | 环境变量 | 默认 | 说明 |
|---|---|---|---|
| `--resident-tokens N` | `NINFER_RESIDENT_TOKENS` | 20480 | 常驻窗口 token 数 |
| `--sink-tokens N` | `NINFER_SINK_TOKENS` | 2048 | attention sink token 数 |
| `--archive-mib N` | `NINFER_KV_ARCHIVE_MIB` | 2048 | 主机 arena；给这个参数自动开归档 |
| `--kv-retrieval 1` | `NINFER_KV_RETRIEVE` | 0 | 打开检索召回（同时自动开归档，并按 `--resident-tokens` 推 arena 大小） |
| `--kv-recall-pages N` | `NINFER_KV_RECALL_PAGES` | 112 | 召回页数（112 页 = 7168 token）。r9 默认 0（完全 opt-in），0.1.1 起默认 112 |
| — | `NINFER_KV_RERANK` | 1 | 两阶段精排（top-256 用归档 int4 K 重打分，排除前 32 个 sink 页）；设 0 只用页均值 |
| — | `NINFER_KV_RECALL_DIV` | 1 | 1 = 单整段（r9 行为）；>1 才切成多峰窄块 |
| — | `NINFER_KV_RECALL_SPLIT` | 1 | 自适应第二段：次峰够近够强时自动分成两段；`0` = 只用单整段 |
| — | `NINFER_KV_RECALL_FILL` | 1 | 仅 `DIV>1` 时生效：用剩余预算逐页补满名次靠前的页 |
| — | `NINFER_KV_REPLAY` | 1 | 设 0 关闭最后 chunk 重放 |
| — | `NINFER_KV_AGG` | max | `avg` 回退到旧的逐单元平均聚合 |
| — | `NINFER_KV_DETREND` | 0 | 距离去趋势（实测有害，默认关） |

`--kv-retrieval` 的自动档位：不给 `--archive-mib` 时按 `2 × resident × 17920 B` 推算并夹在
1024–4096 MiB；`--resident-tokens 20480/32768/65536/98304` 分别得到 1024/1120/2240/3360 MiB。
显存 ≤16 GB 的卡建议 4096 上限，实测 6144 MiB 会在 `cudaMallocHost` 失败。

诊断：`NINFER_WINDOW_TRACE`、`NINFER_KV_ARCHIVE_TRACE`、`NINFER_KV_RETRIEVE_TRACE`。

## 5. 四次证伪（如果你要改打分，先读这段）

页均值 Q·K 打分的第一版**完全不能用**：needle 区排到 #322，top-1 是一个无关页，64 个候选分数
分布看着正常但排序和内容无关。依次试过：

| 尝试 | 结果 |
|---|---|
| 余弦归一化（逐层 L2） | 分数散开了，排序不变 |
| query 取最后 32 token 均值 | 与上一行几乎逐位相同 |
| 逐层 z-score 再平均 | **16 层全部把同一个错误页排第一** |
| 减去候选集均值（中心化） | 与第 1 行**逐位相同** |
| 页摘要从 64 token 改 4×16 子块取 max | needle 名次 #309 → #322，等于没动 |

结论：**在整页平均之后再聚合，信息已经被抹平**。换法之后立刻见效：

| 改动 | needle 区名次 |
|---|---|
| 逐 (层,头) 打分，各单元 z-score 后取 **max** | #35 |
| 同前，但 64 个单元改为**取平均** | **#8**（默认） |
| 再叠加距离去趋势（±64 页邻域减均值） | #417（有害，已改为 opt-in） |

取 max 时单个头的尖峰就能决定名次，softmax 出现 `top-1 = 0.816` 的独霸；改平均后分布平坦
（`top-1 ≈ 0.006`），排序才对得上内容。

还有一个真实 bug 值得记：摘要环形槽最初只有 512/1152 个，而 65536 档窗口就有 1024 页、整段
prompt 1534 页，后段页面按 `page % ring_pages` 复用槽位把前段 `tags_` 覆盖掉，`copy_page_means`
于是整页跳过、留下全零 → 所有候选分数**恰好等于 `1/候选数`**。看到这个数字就直接查 ring 容量，
别去调算法。现在 ring = 2048 槽、单页摘要 64 KiB（子块从 4 收回 1，四子块实验已证伪）。

中间还有一个「本来已经成功」的对照实验值得记：**手工指定召回 token 位置**（`47104` / `49152`）
时答案完全正确，说明召回机制本身没问题，坏的只是打分。这条对照一直是判断「是选块错还是机制错」的
分界线。

## 6. 实测

环境：RX 9070 XT 16GB，Windows 11 26200，HIP SDK 7.2.3，`--kv rk4v4-e8`，单并发。

**多用例（96K 正文，needle 埋 25/50/75/92% 深度）**

| 深度 | 检索关 | 检索开 |
|---|---|---|
| 25% | miss | **hit** |
| 50% | miss | **hit** |
| 75% | miss | **hit** |
| 92% | hit | hit（输出与关闭时逐 token 相同） |

**归档开销**（98K prompt，开/关归档各两轮）：输出逐 token 相同；
prefill `801.997 → 800.446`，decode `33.14 → 33.62`，噪声内。

**召回档位（`tasks_needle_mid_q.txt`，98218 token，needle 埋在 ~48% 处，页区间 [768,800)）**

| `--resident-tokens` | 常驻窗口 | needle 是否在窗口内 | 输出 |
|---|---|---|---|
| 20480 | 320 页 | 否（纯召回） | 与 GT **逐 token 相同** |
| 32768 | 512 页 | 否（纯召回） | 与 GT **逐 token 相同** |
| 65536 | 1024 页 | 是 | 与 GT 逐 token 相同 |

GT = `45 669 24312 10846 8006 5434 12 19 19 22 16 248046`；召回段 `704..815`（112 页，
以峰值页 760 为中心）`from token 45056`，重放 `4 tokens`。`[kv-rank]` 里 needle 区
[768,800) 的最高名次是 #18（页 773），探测页 768 在 #71；峰值页 760 比 needle 区早 8 页 ——
这正是「以峰值为中心取整段」而不是取单页的原因。

> 注：`tasks_needle_q25/q75/q92` 语料在 needle 之后还夹了一条「列出 1..400 的整数」的指令，
> 模型会去答那道题，因此**不能**用它们判断召回命中；只有 mid 语料是可信 oracle。

**账目自洽**：归档 90,048 token + 常驻 6,119 + sink 2,048 = 98,215；归档块数 86 == 滑移次数 86；
归档页数 1,407 == 90,048 / 64；块区间严格递增无重叠。

**GATE 回归**（检索默认关闭）：8k `decode 35.63 / prefill 931.4`，96k `24.90 / 422.1`，
mtp `80.21 / 842.2` —— 全部在容差内，无回退。

## 7. 已知限制

- 打分是**启发式但有精排兜底**：页均值粗筛的峰值比文档早约 ±8 页（实测峰值 760 / needle 区
  768..800），所以召回取「以峰值为中心的整段」而不是单页。两阶段精排（top-256 用归档 int4 K
  重打分，默认开）在这份语料上不移动 top-1，它保证的是「粗筛把某一页排到前面但内容不符」时
  不会选错；真正的偏置仍由整段宽度吸收。
- 摘要是**页均值**（`kRetrievalSubChunks = 1`）。容量上界是 Quest 的 per-channel min/max；
  子块取最大的变体代码里已有，但编译期关闭，开启前需要重测（早期开着时反而更差）。
- 摘要环形缓冲 **2048 槽**，一个 chunk 的页数上限就是它（当前 chunk 宽度 = 常驻窗口）。
  窗口 ≥ 131072 token 时会绕回、`tags_` 被覆盖、分数退化成 `1/候选数` —— 要开更大窗口必须
  同步扩大 ring（`kRetrievalRingPages`）。
- 召回时机固定在 **prefill 结束后**，query 取 prompt 最后一个 token。问题不在末尾（例如
  多轮对话里问题在中间）时没有保证。
- 摘要额外占显存约 82 MB（1311 页 × 64 KiB）+ 128 MiB 环形缓冲。
- 只在 needle 类语料上验证过；多跳推理、整篇摘要这类任务没有数据。

## 8. 与上游 llama-kvmem 的实测对比（同机、同模型、同 prompt）

同一台机器（RX 9070 XT 16 GB）、同一份权重（Qwen3.8-27B-GSQ-RCO-IQ3_S，GGUF / .ninfer 两种容器）、
同一段 98,180 token 的 prompt（`tasks_needle_mid_q.txt`，答案埋在 ~48% 深度），各跑一次：

| | 上游 llama-kvmem beta2 | 本引擎（r9 档位化） |
|---|---|---|
| 启动 | `llama-kvmem-server.exe -c 262144 --kvmem --kvmem-budget 32768 --kvmem-gen-reserve 32768 --kvmem-block-tokens 128 --kvmem-query-policy user -ctk q8_0 -ctv q4_0 --spec-type draft-mtp --spec-draft-n-max 2` | `--ctx 262144 --resident-tokens 32768 --sink-tokens 2048 --kv-retrieval 1 --kv rk4v4-e8` |
| KV 精度 | q8_0 K / q4_0 V | int4 K / int4 V（rk4v4-e8） |
| MTP | 开（n_max = 2） | **关**（窗口模式下 98K prompt 会报 `request reservation exceeds Engine shared KV capacity`） |
| prefill | 380.4 t/s（258.1 s / 98,180 token） | **563.7 t/s** |
| decode | 13.98 t/s | **29.79 t/s** |
| 答案 | `NINFER-ZEBRA-4471` ✅ | 同答案 ✅（与 GT 逐 token 相同） |

读法：
- 我们这边**没开 MTP**，上游**开了**，所以 decode 2.1× 的差距是在我们让一手的情况下测出来的。
- 上游 KV 是 q8_0/q4_0（约 1.5× 于 int4/int4），注意力带宽开销更大；它胜在 query 策略更细
  （整段用户消息 + 128 token 块），我们胜在窗口与内核效率。
- 各只跑一次，速度 ±5% 波动；**准确性这个规模只能算一个样本**，要下「不低于」的结论仍需成套语料。

### 8.1 同 MTP 对比（两边都开 MTP2，全上下文模式）

同一份语料按 16K / 32K / 64K 截断（needle 固定在 48% 深度，答案相同），两边都开
`--spec-type draft-mtp --spec-draft-n-max 2` / `--spec mtp --draft-tokens 2`：

| prompt | 上游 prefill | 上游 decode | 本引擎 prefill | 本引擎 decode |
|---|---|---|---|---|
| 16K（15,803 / 15,814 tok） | 762.9 t/s | 34.56 t/s | **883.2 t/s** | **52.02 t/s** |
| 32K（32,092 / 32,103 tok） | 633.3 | 35.69 | **724.6** | **45.17** |
| 64K（64,963 / 64,974 tok） | 454.0 | **35.16** | **521.3** | 32.67 |
| 98K（98,180 tok） | 380.4 | 13.98 | **563.7**（无 MTP） | **29.79**（无 MTP） |

读法：prefill 我们三个尺寸都快 14–16%；decode 在 16K/32K 快 27–50%，到 64K 反而慢 7% ——
上游把注意力封在 32K 工作集里，decode 基本恒定；我们的 decode 随上下文线性下降，这是
「窗口 + 召回」该接手的地方。98K 一档上游掉到 14 t/s，是因为工作集装不下、每步都要换块。

### 8.2 MTP 与常驻窗口：曾经互斥，现已打通

pre-r8 时 `--resident-tokens` 与 `--spec mtp` **不能同时开**：任何长度、任何窗口值都会立刻失败

```
error: request reservation exceeds Engine shared KV capacity
```

实测矩阵（全部失败）：prompt 15.8K / 32K / 64K / 98K × resident 8192 / 20480 / 32768；
去掉 `--resident-tokens` 后同一条命令立刻正常（16K：prefill 883.2、decode 52.02）。
所以上面 98K 那一行我们只能是「无 MTP」对「有 MTP」，不是同等条件。

原因在 `planning/pressure.cpp::isolated_request_feasible()`：窗口模式只把
`device.main_kv_pages` 钳到池容量，MTP 草稿头带来的 lane / state / backend 页需求没有同样的
钳制，于是判定 `PermanentlyInfeasible`。打开 `NINFER_ADMIT_TRACE=1` 可以看到具体是哪个维度：

```
[admit] final 1/1 1/2 128/128 248/129 0/8 0/8589934592
                       主KV 用满  ^草稿缓存 248 页 / 只有 129 页
```

进一步定位：`planning/startup.cpp` 里 MTP 物理页数 = 主 KV 物理页数 + draft 页，
而主 KV 物理页数在窗口模式下已被裁到窗口大小，草稿缓存因此只有 129 页。

**修复（pre-r9，已入库）**：把窗口基址的纪律推进到 MTP 路径——KV 索引用紧凑坐标、RoPE 用绝对位置。
四处落点：①`program/context_work.cpp` 的 `mtp_causal_attention_envelopes` 减窗口基址、下界夹 1
（0 号槽是内核哨兵）；②`planning/request_plan.cpp` 的 MTP 配额改按 `resident_entitlement_tokens`；
③`storage/context.cpp` 给投机缓存加 `compact_backend`，并让它的 `slide_prefix_pages` 与主 KV 同步滑页；
④`program/decode.cpp` 的 verify RoPE 位置补回窗口基址（窗口只改「注意力扫哪些槽位」，不改旋转相位）。

最后一处致命 bug 是 `ensure_sequence_kv_mapped()` 里 `*sequence.kv->backend` **没有判 `has_value()`**：
普通请求没有投机缓存，解引用空 optional 得到垃圾句柄，于是**所有**窗口用例都在第一个 chunk 前
抛 `KV address-space handle is stale`。加上判空后整条链路才通。

**打通后的实测（窗口 + 召回 + MTP2，`--archive-mib 3072`）**

| 用例 | prompt | 窗口 | 输出 | prefill | decode |
|---|---|---|---|---|---|
| hh16k | 15,814 | 8192 | 与 GT 逐 token 相同 | 799.7 | **52.9** |
| hh32k | 32,103 | 8192 | 仅末位差 2（`…16 248044`） | 806.9 | 37.5 |
| hh64k | 64,974 | 16384 | 仅末位差 2 | 693.8 | 30.6 |
| mid98k | 98,182 | 32768 | 与 GT 逐 token 相同 | 556.0 | 35.4 |
| hh192k | 196,427 | 32768 | 与 GT 逐 token 相同 | **535.7** | **33.8** |

## 9. 长上下文（128K–192K）：为什么必须开窗口

引擎按**整段** KV 预留运行期容量，这张卡（16 GB）扣掉 12.5 GB 权重后只剩约 3.8 GB，所以：

- 128K 文本裸跑直接拒绝：`requires 5,271,090,176 bytes … the largest capacity this device can hold is
  about 189,646 tokens or less`；
- 192K 同理。**窗口不是优化，是这些长度唯一能跑的路径。**

窗口档位实测（同一份 192K 语料，196,427 token，needle 在 48% 深度）：

| 窗口 | 结果 | prefill | decode |
|---|---|---|---|
| 32768 | ✅ 与 GT 逐 token 相同 | 535.7 | 33.8 |
| 16384 | ❌ 前段复读噪声、答案在后半段才补出来 | 683.9 | 42.7 |

16K 在 64K 语料能用，到 192K 就崩——因为一轮召回只能装 112 页（7168 token），窗口太小时 needle 邻域
被切碎。**结论：长上下文窗口 ≥ 32768 起步。** 显存按 int4 `17,920 B/token` 估算：32768 → 0.59 GB、
65536 → 1.17 GB，叠加 MTP 草稿行后 65536 已贴边。

### 9.1 `--ctx` 给大了会静默降速（不是功耗回归）

`--ctx` 决定 KV 池要预留多少。池装不下时引擎**不会报错**，而是退回一条半速路径：prefill 从
424 掉到 350、decode 从 24.8 掉到 11.8，日志里一个字都没有。实测差异只体现在启动余量：

| ctx | KV 池 | 启动后可用 | 结果 |
|---|---|---|---|
| 110000 | 1.97 GB | `avail_after_startup=1,226,964,992` | prefill 350 / decode 11.77（静默降速） |
| 100000 | 1.79 GB | `avail_after_startup=1,405,878,272` | prefill 424 / decode 24.79（正常） |

同一份 98,155-token 语料、同一二进制，只改 `--ctx`。所以看到"长上下文臂忽然掉一半"，先量
`avail_after_startup`，别先去翻 kernel。`build/_gate.py` 的 96k 臂因此从 110000 定到 100000。

## 10. 已知问题：召回后末位停止符可能差 1 个（P3，暂不修）

**现象**：开启召回、且召回真的换入了页的档位（实测 32K@8192、64K@16384），生成的最后一个 token
可能是 `248044` 而不是逐 token 基线里的 `248046`；正文完全一致。

**为什么它是等价输出、不是错误**：`infer/include/infer/types.h:564` 的
`enum class FinishReason { None, OutputLimit, ContextCapacity, StopToken, StopString, Cancelled }`
里 `finish=3` 就是 `StopToken`。两档实测的 `finish` **都是 3**，说明 `248044` 与 `248046` 同在停止符
集合里，两次都是"命中停止符正常终止"，而不是被 `--max-new` 截断（那会是 `OutputLimit=1`）。

**成因**：召回发生在 prefill 结束、`ensure_kv_window` 里；为了让最后一段的注意力也能看到换入的页，
`prefill.cpp` 会**重放最后一个 chunk**。重放会把末位 logits 重新算一遍，模型在"看得见召回页"的上下文里
argmax 落到同族的另一个停止符上。删掉重放会立刻答错（`99143 96483 …`），所以不能为了末位一致去动重放。

**影响**：文字输出无影响（控制符不进正文）、语义无影响（同为 StopToken）、速度无影响（噪声级）。
唯一有影响的是**逐 token 精确比对**的自动化回归会红。

**处理决定（2026-11-09 与用户确认）**：先不修；在回归脚本里把"末位停止符等价类"加白名单，
而不是改召回/重放路径。若将来发现下游把不同停止符映射成不同语义，再按下面两种方式之一修：
①重放前把末位 logits 那一行存到设备缓冲、重放后 `cudaMemcpyAsync` 还原（KV 用重放的、logits 用重放前的，约十行，不影响速度）；
②只重放 KV、不重算末位 logits。
**正向用途**：末位出现 `248044` 可以作为"这次运行确实走了召回+重放"的快速指纹。

## 11. 深度复核（0.1.2）：196K–261K 精度与速度

复核目的：确认「prefill / decode 不降的前提下，长上下文可用且精度与上游一致」。配置统一为
`--resident-tokens 32768 --sink-tokens 2048 --kv-retrieval 1`、`NINFER_KV_ARCHIVE_MIB=4096`、
`--kv rk4v4-e8`，语料是同一套压测文（口令 `NINFER-ZEBRA-4471`，问题问口令）。

| 语料 | 实际 prompt token | `--ctx` | prefill | decode | 命中 |
|---|---|---|---|---|---|
| `tasks_hh192k.txt` | 196,423 | 262144 | 540.4 t/s | 29.77 t/s | ✅ 与 GT 逐 token 相同 |
| `tasks_hh256k_trim.txt` | 251,201 | 262144 | 534.9 t/s | 30.50 t/s | ✅ 同上 |
| `tasks_hh262k.txt` | 260,909 | 262144 | 534.1 t/s | 29.79 t/s | ✅ 同上 |

深度从 196K 拉到 261K，prefill 变化 1.2%、decode 变化 2.4%，与 r9 的 192K 档（535.7 / 33.8）
同一水平线。三档 IDS 均为 `45 669 24312 10846 8006 5434 12 19 19 22 16 248046`。

### 11.1 上限 262,144 是模型的 `max_position_embeddings`，不是我们设的

`ninfer/src/models/qwen3_5/program/planning/startup.cpp:752-755`：

```cpp
if (options.max_context == 0 ||
    options.max_context > parameters.model.config().text.max_position_embeddings) {
    throw std::invalid_argument("max_context exceeds the configured position capacity");
}
```

所以 `--ctx 393216` 会立刻失败（实测 7.6 s 退出），而 `--ctx 262144` 是合法的最大值。
原 `tasks_hh256k.txt` 实测约 264K token，**天生超过这个上限**，任何配置都跑不了；要验「接近上限」
必须按实测比例（这篇语料 3.9995 chars/token，251,201 token ↔ 1,004,983 chars）重新裁一份。
准确说法是「≤262,144 token 可用，实测到 260,909（99.5%）」。

### 11.2 多文档 A/B：单整段 vs 多峰分块

语料 `tasks_multidoc.txt`：118,235 token，两个互不相关的事实分别埋在 25%（ATLAS）与 50%（BOREAL）
深度，问题要求同时给出两个口令。两个事实都在 32K 尾部窗口之外，必须靠召回。

| 档 | 实际召回页 | 回答 |
|---|---|---|
| 全窗口参照（`--resident-tokens 131072`，不召回） | — | ✅ 两个口令都给出 |
| 单整段（默认） | `859..970`，中心 915 ≈ BOREAL | ❌「文中没有 ATLAS / BOREAL」 |
| 多峰（`NINFER_KV_RECALL_DIV=2`） | `887..942`（含 BOREAL）+ `421..476`（含 ATLAS） | ❌ 同上，**逐 token 与单整段相同** |
| 单整段 + 问题只问 BOREAL | `859..970` | ✅ `NINFER-BOREAL-2222` |
| 去掉 ATLAS 事实 + 问题只问 BOREAL | `859..970` | ✅ `NINFER-BOREAL-2222` |

结论三条：

1. **召回投递是对的**。`[kv-refine] top3: 915@0.2297 449@0.1888` 精确落在 BOREAL（page 922）与
   ATLAS（page 459）上；只问一个事实时答案逐 token 正确。
2. **问「两个都要」时两种策略都无法作答**——模型给的是「文中没有 ATLAS / BOREAL」，两种策略
   逐 token 相同。这是模型行为限制（上下文只恢复一部分、问题却要求两处），不是投递缺陷，
   所以**这个问法不能用来评判多峰**。
3. **多峰的真实收益要问「次峰那个事实」才看得出来**（下表）。单峰只装 top-1 所在的那一段，
   问题问的是另一个事实时，模型会**自信地答成 top-1 的口令**；多峰把两个峰都装进来才答对。

| 问题 | 单整段（默认，`859..970` 只含 BOREAL） | 多峰（`887..942` + `421..476`） | 真值 |
|---|---|---|---|
| ATLAS 的口令？ | ❌ `NINFER-BOREAL-2222`（答成另一个事实） | ✅ `NINFER-ATLAS-1111` | ATLAS |
| BOREAL 的口令？ | ✅ `NINFER-BOREAL-2222` | ✅（单整段已覆盖） | BOREAL |
| 两个口令都要？ | ❌ 拒答 | ❌ 拒答（逐 token 相同） | 两条 |

因此 0.1.2 的取舍是：**默认单整段**（单针/单事实永不回归，且实测 98K mid 语料逐 token 对），
**多峰保留为显式 opt-in**（`NINFER_KV_RECALL_DIV=2`，多事实场景实测能把次峰事实答对）。
两者不是替代关系：单峰覆盖更深（112 页连续），多峰覆盖更广（两个 56 页段）。

> 历史：多峰最初在 `94b3556` 引入，理由是平均聚合让峰值提前约 8 页，块必须**从峰值向后延伸**
> 才能盖住 needle 区（top-1 峰 760 的块 `[760,816)` 覆盖 needle `[768,800)`）。后来 `38934cf`
> 改成「以峰值为中心」，`82b8d88` 又引入 `divisor` 把块缩到 `slots/2`，两个改动叠加后向前只剩
> 28 页，盖不住 needle 区尾部 → 这就是 0.1.1 的回归来源。

### 11.3 与上游 llama-kvmem 同语料对拍

同机同卡、同一份语料、同一问题、两边都开 MTP2；上游参数见 §8（`--kvmem --kvmem-budget 32768
--kvmem-gen-reserve 32768 --kvmem-block-tokens 128 --kvmem-query-policy user -ctk q8_0 -ctv q4_0`）。

| 语料 | 上游 prefill | 上游 decode | 本引擎 prefill | 本引擎 decode |
|---|---|---|---|---|
| 196K | 336.5 t/s | 24.28 t/s | 540.4 t/s | 29.77 t/s |
| 250K | 323.2 t/s | 29.15 t/s | 534.9 t/s | 30.50 t/s |

两边都命中口令。prefill 约 1.6×，decode 持平到略快。注意上游 KV 是 `q8_0`/`q4_0` 量化，
本引擎是 `rk4v4-e8`，两者不是同一个精度档，此表只用于"精度行为对不对"和"速度量级"，不作
逐 token 等价声明。

### 11.4 准确率对拍：8 例逐条一致（0.1.2）

§11.3 只比了两条长语料，这里补上完整的准确率矩阵。**做法**：8 个用例、同一份 prompt、两边都
用 32K KV 预算、贪心解码（`temperature=0`），比"答案字符串是否命中语料里的口令"，不比 token 序列
（两边 KV 精度档不同，逐 token 比较没有意义）。脚本 `build/_accuracy_parity.py`
（`--ours-only` / `--upstream-only` 两段跑，避免抢 GPU）。

| 用例 | 语料长度 | 我们 | 上游 llama-kvmem |
|---|---|---|---|
| q25（needle 25%，页 395） | 32K | PASS `NINFER-ZEBRA-4471` | PASS |
| q50（页 790） | 32K | PASS `NINFER-KOALA-8823` | PASS |
| q75（页 1186） | 32K | PASS `NINFER-VIPER-1092` | PASS |
| q92（页 1454） | 32K | PASS `NINFER-ORCA-7745` | PASS |
| mid（50%，页 790） | 32K | PASS `NINFER-ZEBRA-4471` | PASS |
| multi（双文档，页 462） | 32K | PASS `NINFER-ATLAS-1111` | PASS |
| 196K | 196K | PASS `NINFER-ZEBRA-4471` | PASS |
| 250K | 250K | PASS `NINFER-ZEBRA-4471` | PASS |

**准确率：8/8 对 8/8，逐例答案一致**（q75/q92 其实落在 32K 常驻窗口内，不走召回，用来验证
"不召回时也不退化"）。速度**必须在两边都开 MTP2 的前提下比**（第一版我们忘了给自己开
`--spec mtp --draft-tokens 2`，拿 AR 去比人家的投机解码，得出的"decode 低 10%"是假的）：

| 用例 | 我们 wall / prefill / decode | 上游 wall / prefill / decode |
|---|---|---|
| q25 | 191.9s / 557.3 / **41.05** | 256.3s / 384.6 / 25.14* |
| q50 | 191.0s / 558.7 / **34.78** | 260.6s / 378.2 / 32.85 |
| q75 | 191.0s / 558.2 / **41.51** | 260.0s / 379.0 / 33.25 |
| q92 | 191.0s / 559.0 / **35.32** | 259.9s / 379.2 / 32.27 |
| mid | 191.0s / 558.7 / **35.18** | 260.3s / 378.6 / 33.33 |
| multi | 239.1s / 550.6 / **41.53** | 328.6s / 361.0 / 33.01 |
| 196K | 383.2s / 536.6 / **40.01** | 593.4s / 331.9 / 32.99 |
| 250K | 495.3s / 532.6 / 32.42 | 779.5s / 323.1 / 32.93 |

\* 上游 q25 是它第一例（预热），后面稳定在 32–33，因此以 32–33 为准。

结论：**prefill 1.4–1.6×，decode 除 250K 档基本持平（32.42 vs 32.93）外全部更快（35–42 vs 32–33）**。
我们 KV 是 int4/int4、上游是 q8_0/q4_0，本引擎每 token 读的 KV 字节更少，这部分优势在 decode 上体现出来了；
250K 档被归档页的 host↔device 换入换出吃掉，是下一步 decode 优化的重点。

> **评测语料曾经是坏的（2026-11-13 发现）**：`tasks_needle_q*.txt`、`tasks_needle_mid_q.txt`、
> `tasks_hh*.txt` 末尾的中文问句在生成时被写成了 U+FFFD 替换字符，q 系列里唯一可读的指令是英文
> 诱饵 `Question: List the integers from 1 to 400 …`。模型照着诱饵列整数**不是召回失败**。
> 现在回归统一用 `build/_mk_fixed_corpora.py` 重新生成的 `tasks_fix_*.txt`（U+FFFD 计数为 0），
> 上述 8 例均在这些修好的语料上跑出。教训：**评测语料要先断言"问句可读"，再谈命中率**；以及
> **对拍两侧的启动参数必须逐项核对**（`--spec`/MTP 一开一不开会让 decode 结论完全反过来）。

