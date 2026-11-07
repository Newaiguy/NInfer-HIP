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
| `--kv-retrieval 1` | `NINFER_KV_RETRIEVE` | 0 | 打开检索召回 |
| — | `NINFER_KV_RECALL_PAGES` | 112 | 召回页数 |
| — | `NINFER_KV_RECALL_BACKOFF` | 56 | top-1 之前回退页数 |
| — | `NINFER_KV_REPLAY` | 1 | 设 0 关闭最后 chunk 重放 |

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
| 逐 (层,头) 打分，各单元 z-score 后取 max | **#35** |
| 以峰值页为中心取 112 页连续段 | 命中（配合重放） |

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

**账目自洽**：归档 90,048 token + 常驻 6,119 + sink 2,048 = 98,215；归档块数 86 == 滑移次数 86；
归档页数 1,407 == 90,048 / 64；块区间严格递增无重叠。

**GATE 回归**（检索默认关闭）：8k `decode 35.63 / prefill 931.4`，96k `24.90 / 422.1`，
mtp `80.21 / 842.2` —— 全部在容差内，无回退。

## 7. 已知限制

- 打分是**启发式**，定位误差约 ±8 页（512 token），所以取 112 页连续段兜住，代价是显存里
  有 7K token 的「陪跑」。
- 召回时机固定在 **prefill 结束后**，query 取 prompt 最后一个 token。问题不在末尾（例如
  多轮对话里问题在中间）时没有保证。
- 摘要额外占显存 ~168 MB（1311 页 × 128 KiB）+ 64 MB 环形缓冲。
- 只在 needle 类语料上验证过；多跳推理、整篇摘要这类任务没有数据。
