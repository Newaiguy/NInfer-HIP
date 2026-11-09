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
- 每页 64 token，切成 4 个子块（每块 16 token）各求均值，取子块间最大值 —— 把 needle 落在页面
  中间的情况救回来，避免被整页平均冲淡。
- 摘要 = 16 层 × 4 头 × 256 维 bf16 = **128 KiB/页**。
- 环形缓冲 `kRetrievalRingPages = 512`（必须 ≥ 单个 chunk 的最大跨页数 320；降到 128 会出现槽碰撞，
  表现为**所有候选分数严格相等**，等于 `1/候选数` —— 看到这个数就是摘要全零的指纹）。
- 采集点：`text.cpp` 的 `attn_mix()`，`qn`/`kn` 经 RMSNorm 之后、`text_rope` 之前。
- query：只用最后一个 token（`kLastQueryTokens` 的平均在这里等价，实测结果相同）。

### 3.3 打分 `rank()`

- kernel 输出 **64 个单元**（16 层 × 4 头）的分数。
- host 侧每个单元跨全部候选做 z-score，然后**每页取 64 个单元的最大值**。
- 这一步是整个特性能不能用的分水岭。走过的弯路全记在下面 §5。

### 3.4 召回 `context.cpp::recall_into_tail()`

- `want = NINFER_KV_RECALL_PAGES`（默认 **112** 页 = 7168 token）。
- top-1 页 → `backoff = want/2`（默认 56）→ 取 `[peak - 56, peak + 56)` 连续段。
- `archive->recall_pages(start * 64, want)` → `pool.copy_from_host()` 写到槽位
  `[sink_pages, sink_pages + 112)`。
- 写之前先把这些槽位原有的页归档（`watermark` 过滤防止重复归档），再更新 watermark。

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
| `--kv-recall-pages N` | `NINFER_KV_RECALL_PAGES` | 112 | 召回页数（112 页 = 7168 token） |
| — | `NINFER_KV_RECALL_BACKOFF` | slots/2 | top-1 之前回退页数 |
| — | `NINFER_KV_REPLAY` | 1 | 设 0 关闭最后 chunk 重放 |
| — | `NINFER_KV_AGG` | avg | `max` 回退到旧的逐单元取最大聚合 |
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

GT = `45 669 24312 10846 8006 5434 12 19 19 22 16 248046`；召回段 `112 pages from token 45056`，
重放 `4 tokens`。`[kv-rank]` 里 needle 页名次 #8、峰值页 760（含 8 页定位误差，112 页段兜住）。

> 注：`tasks_needle_q25/q75/q92` 语料在 needle 之后还夹了一条「列出 1..400 的整数」的指令，
> 模型会去答那道题，因此**不能**用它们判断召回命中；只有 mid 语料是可信 oracle。

**账目自洽**：归档 90,048 token + 常驻 6,119 + sink 2,048 = 98,215；归档块数 86 == 滑移次数 86；
归档页数 1,407 == 90,048 / 64；块区间严格递增无重叠。

**GATE 回归**（检索默认关闭）：8k `decode 35.63 / prefill 931.4`，96k `24.90 / 422.1`，
mtp `80.21 / 842.2` —— 全部在容差内，无回退。

## 7. 已知限制

- 打分是**启发式**，定位误差约 ±8 页（512 token），所以取 112 页连续段兜住，代价是显存里
  有 7K token 的「陪跑」。真正消掉误差要靠两阶段精排，而归档页是 `rk4v4-e8` 量化格式
  （`page_stride = 1146880 B`），主机侧要先写反量化才能做逐 token 精排，暂未做。
- 摘要是**页均值**，容量上界是 Quest 的 per-channel min/max 上界；后者能把「页内某一维特别
  突出」的信号留住，是下一步最现实的算法升级。
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
