# 0.1.0-pre-r8 — KV 记忆：把滑出显存的上下文捞回来

> 一句话：**长文里被窗口丢弃的老上下文，现在能按「和问题多相关」挑一段换回显存，继续参与注意力。**
> 96K 正文、答案埋在 25% / 50% / 75% 深度的 needle 测试：`pre-r7` 全错，本版**逐 token 命中**。

---

## 这版到底加了什么

在 16 GB 卡上跑 27B，KV 装不下 126K，所以引擎一直用**滑动窗口**：最前面 2048 token（sink）+ 最近
20480 token 留在显存，中间滑出去的页直接扔掉。后果是「读一整本书然后问第 3 章」这类事做不了——
那段 KV 已经不在了。

本版加了一条**KV 记忆**通路，四步：

1. **归档**：窗口每滑一次，即将被丢弃的 KV 页先拷到主机内存（`copy_to_host`），
   按 token 区间连续存放；arena 满了 FIFO 淘汰。存的是**完整 K+V**，不是摘要。
2. **页摘要**：prefill 时顺手统计每页 64 个 token 的 K 均值（16 个全注意力层 × 4 KV 头 × 256 维，bf16，
   128 KiB/页）。不额外读一遍 KV。
3. **检索召回**：prompt 读完、拿到最后一个 query 后，给所有归档页打分 ——
   **每个 (层, 头) 单元单独标准化，一页取 64 个单元的最大值**；以 top-1 页为中心取
   112 页（7168 token）连续区间，`copy_from_host` 写回显存**下一次滑移本来就要丢弃的槽位**
   ⇒ 不额外占显存容量。
4. **重放最后 chunk**：召回发生在 prefill 之后，最后一个 chunk 的注意力已经算完，
   所以重算那几 token（实测 4 个），让问题真正看到召回内容。

不传 `--kv-retrieval` 时，行为与 `pre-r7` **完全一致**（代码路径全部在开关之后，默认关）。

## 怎么用

```bat
ninfer_mini_nordc.exe <你的 .ninfer 模型> --chat ^
    --kv rk4v4-e8 --ctx 262144 ^
    --resident-tokens 20480 --sink-tokens 2048 --archive-mib 4096 --kv-retrieval 1
```

四个新参数（等价环境变量写在括号里）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--resident-tokens N`（`NINFER_RESIDENT_TOKENS`） | 20480 | 最近 N 个 token 常驻显存 |
| `--sink-tokens N`（`NINFER_SINK_TOKENS`） | 2048 | 最前 N 个 token 永远保留（attention sink） |
| `--archive-mib N`（`NINFER_KV_ARCHIVE_MIB`） | 2048 | 主机侧归档 arena；**给这个参数自动开归档** |
| `--kv-retrieval 1`（`NINFER_KV_RETRIEVE`） | 0 | 打开检索召回 |

调优：`NINFER_KV_RECALL_PAGES`（召回页数，默认 112）、`NINFER_KV_RECALL_BACKOFF`（峰前回退，默认 56）、
`NINFER_KV_REPLAY=0`（关重放，做对照用）。
看它干活：`NINFER_WINDOW_TRACE=1`、`NINFER_KV_ARCHIVE_TRACE=1`、`NINFER_KV_RETRIEVE_TRACE=1`。

### 最佳配置

- **≤32K 上下文别开**：常驻窗口就装得下，开了只是白占内存。
- **≥48K 长文、问题在末尾**：上面那条命令就是推荐值；`--archive-mib 4096` 是为了让 96K 正文
  全部进归档（实测只需 ~1.7 GB，4096 留余量，避免 FIFO 把要用的页淘汰掉）。
- 想省显存可把 `--resident-tokens` 降到 12288，代价是召回频率变高。
- 与 MTP 不冲突：`--spec mtp --draft-tokens 2` 照旧（但真实上下文 ≥100K 仍要关 MTP，见下）。

## 实测

**多用例（96K 正文，needle 埋四个深度）**

| 深度 | `pre-r7` / 检索关 | 本版检索开 |
|---|---|---|
| 25% | ❌ 编造 | ✅ 口令逐字符命中 |
| 50% | ❌ 编造 | ✅ 命中 |
| 75% | ❌ 编造 | ✅ 命中 |
| 92% | ✅ 命中 | ✅ 命中，**与关闭时输出逐 token 相同** |

**开销**（98K prompt，开/关归档各两轮）：输出逐 token 相同，prefill `801.997 → 800.446`、
decode `33.14 → 33.62` —— 噪声内。

**回归 GATE**（8K / 96K / MTP 四格，检索默认关闭）：
`8k decode 35.63 · prefill 931.4` / `96k 24.90 · 422.1` / `mtp 80.21 · 842.2`，全部在 ±3% 容差内。

## 限制（开之前先看）

> 历史文档（r8）。这里描述的「112 页连续段」是当时的行为；0.1.2 把它恢复为默认实现，
> 详见 docs/KVMEM.md §3.4 与 RELEASE_NOTES_0.1.1.md 的 0.1.2 更正。

- 召回是**启发式**：页均值打分定位误差约 ±8 页，所以取 112 页连续段兜住，代价是显存里有 7K token 陪跑。
- **必须读完 prompt 才能召回**（query 取最后一个 token）；问题不在末尾没有保证。
- 额外显存：页摘要 ~168 MB（1311 页）+ 64 MB 环形缓冲；额外主机内存：最多 `--archive-mib`。
- 只在 needle 类语料上验证过，多跳推理 / 整篇摘要没有数据。
- 只在 RX 9070 XT + Windows 11 + HIP SDK 7.2.3 上验证。

## 直接说结论：什么时候该升级

- 你只跑 ≤32K 对话 → **不必升级**（本版对老路径零影响，但也没收益）。
- 你要读长文档、且问题在末尾 → **升级**，按上面的最佳配置开。
- 你要 128K 以上无损上下文 → 这是启发式召回，不是无损扩展，别指望。

## 下载

- `ninfer_mini_nordc.exe` — 191,691,776 B
  `sha256:5433232edbc4d8c2f90f15f98b47d0d9f6a98a50fda1c59c167d83faf32a9b2b`
- 模型权重不在仓库/发布里：`WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3`（15,017,456,896 B）
- 双击开聊：把 exe 与 `harness/launchers/chat.bat` 放同一文件夹，双击 `chat.bat`

## 文档

- [`README.md`](../blob/main/README.md) — 能力状态、KV 记忆用法与参数、快速开始、已知坑
- [`docs/KVMEM.md`](../blob/main/docs/KVMEM.md) — 实现细节、参数表、**四次证伪过程**、实测数据
- [`docs/PERF-CAPACITY.md`](../blob/main/docs/PERF-CAPACITY.md) — 长上下文掉速根因（`--ctx` 必须贴紧）
- [`docs/CHANGELOG.md`](../blob/main/docs/CHANGELOG.md) — 全版本变更

**完整提交**：`541d41a`（本 Release）← `dcbab08`
