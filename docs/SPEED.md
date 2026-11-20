# 速度矩阵（int4 KV，针必须命中）

由 `build/_speed_doc.py` 从 `build/_speed_matrix.json` 生成，**不要手改**。

| ctx | 窗口 | 深度 | prefill t/s | decode t/s | 针 |
|---|---|---|---|---|---|
| 8,192 | 否 | 1 | 926 | 53.9 | HIT |
| 8,192 | 否 | 2 | 926 | 55.8 | HIT |
| 8,192 | 否 | 5 | 921 | 63.7 | HIT |
| 32,768 | 否 | 1 | 732 | 45.9 | HIT |
| 32,768 | 否 | 2 | 731 | 45.4 | HIT |
| 32,768 | 否 | 5 | 730 | 47.5 | HIT |
| 65,536 | 否 | 1 | 549 | 37.4 | HIT |
| 65,536 | 否 | 2 | 547 | 34.7 | HIT |
| 65,536 | 否 | 5 | 547 | 34.1 | HIT |
| 98,304 | 否 | 1 | 391 | 30.6 | HIT |
| 98,304 | 否 | 2 | 426 | 28.3 | HIT |
| 131,072 | 否 | 1 | 299 | 9.5 | HIT |
| 131,072 | 否 | 2 | 298 | 8.4 | HIT |
| 262,144 | 是 | 1 | 531 | 40.3 | HIT |
| 262,144 | 是 | 2 | 531 | 39.5 | HIT |

## 每档最优深度

- 8,192 token：深度 5，63.7 t/s
- 32,768 token：深度 5，47.5 t/s
- 65,536 token：深度 1，37.4 t/s
- 98,304 token：深度 1，30.6 t/s
- 131,072 token：深度 1，9.5 t/s
- 262,144 token：深度 1，40.3 t/s

## 结论

- 深度 5 在 8,192–32,768 token 区间最优。
- 从 65,536 token 起最优点变成深度 1。
- **131,072 token 无窗口掉到 9.5 t/s**（98,304 时是 30.6）：从这一档起必须开窗口；而 262,144 token 开窗口仍有 40.3 t/s。
- 180,000 未计入：常驻容量不够（mini 总是加载视觉；serve 用 --no-vision 才能到 181K）

## 开窗口时的深度对照（2026-11-20，r7 / sha D689A367）

配置：`--kv rk4v4-e8 --resident-tokens 32768 --sink-tokens 2048 --kv-retrieval 1`，归档 4096 MiB，
`--max-new 32 --no-think`，每个臂都要求命中针 `NINFER-<WORD>-<NNNN>`。
原始数据：`build/_window_sweep.json`；脚本 `build/_window_sweep.py`。

| prompt | ctx | d1 | d2 | 结论 |
|---|---|---|---|---|
| 63K | 65536 | 41.8 | **43.2** | d2 快 3.3% |
| 95K | 98304 | **42.9** | 34.9 | d2 慢 18.7%（复跑 35.3，可复现） |
| 127K | 131072 | 42.1 | 41.6 | 打平 |
| 195K | 196608 | **41.6** | 33.7 | d2 慢 19% |
| 249K | 262144 | 40.3 | 39.5 | 打平 |

均值：d1 **41.7**（区间 40.3–42.9），d2 38.6（区间 33.7–43.2）。
原因：开窗口后 attention 已被钉在常驻窗口内，d2 每步多扫一列的代价基本恒定，能否赚回取决于草稿命中率，
而命中率随文本内容波动 —— 所以 d2 只在个别语料上赢 3%，在另一些上亏 19%。

**结论：长上下文一律 d1。** 自动档规则 `--ctx > 32768 取 1` 与此一致，无需覆盖。

## KV dtype: what actually matters (2026-11-20, r7 / sha D689A367)

All arms: 32K resident window + 4 GiB archive, `--draft-tokens 1 --no-think`, needle must hit.
Raw: `build/_dtype_sweep.json`; script `build/_dtype_sweep.py`.

| prompt | rk4v4-e8 (int4) | int8 | bf16 |
|---|---|---|---|
| 63K | **43.2** | 4.7 (MISS) | 41.8 |
| 95K | **42.9** | — | 41.0 |
| 127K | **42.1** | — | 40.3 |
| 195K | **42.3** | 6.6 (MISS) | crash: `hipErrorLaunchFailure` (device.cu:132) |
| 249K | 40.3 | — | — |

Readings:

- **int8 is not usable.** The geometry exists (`ninfer/src/core/paged_kv_storage.h:76-78`,
  `KvCacheStorage::Int8Group64`, 256-d head only) but the kernels do not produce correct output:
  4.7 t/s at 63K and 6.6 t/s at 195K, answer MISS in both cases. Treat `--kv int8` as unsupported.
- **int8 is also bigger, not smaller.** Per (token, head): int4 `rk4v4-e8` = 192 B key + 160 B value
  = 352 B; int8 = 384 B key + 384 B value = 768 B (2.18x). Measured token cost: int4 17,920 B,
  int8 ~39 KB, bf16 65,536 B. So going 8-bit buys nothing except a bit more capacity than bf16.
- **bf16 with the window is only 2-4% behind int4** (41.8 / 41.0 / 40.3 vs 43.2 / 42.9 / 42.1).
  With a 32K window the KV scan is a small slice of a ~23 ms weight-streaming step, so the dtype
  barely shows; dtype only becomes decisive when the window is large or absent (no window at
  128K, int4 d1 collapses to 9.5).
- **bf16 cannot go past ~128K with the window**: 196K aborts with `hipErrorLaunchFailure`
  inside `infer/src/core/device.cu:132`; without a window bf16 tops out near 49K tokens.

## Middle-context arms: int4, window vs no window, depth 5/6/7 (2026-11-20, r7 / D689A367)

Raw: `build/_midctx_sweep.json`; script: `build/_midctx_sweep.py`.

| prompt | window | d5 | d6 | d7 | d1 (ref) |
|---|---|---|---|---|---|
| 63K | off | 34.3 | 28.5 | 24.9 | 37.4 |
| 63K | **on** | **45.8** | 29.4 | 24.7 | 41.8 (d2 43.2) |
| 79K | off | 29.6 | 25.4 | 21.8 | 42.1 |
| 79K | **on** | **45.4** | 38.1 | 32.7 | - |
| 95K | on | 26.5 | - | - | 41.4 |
| 127K | on | 18.3 | - | - | 42.9 |
| 195K | on | 24.5 | - | - | 43.5 |

Readings:

- The window is worth 10-16 t/s at 64K-80K even for d5 (45.8 vs 34.3; 45.4 vs 29.6): a bounded
  attention scan beats a whole-prompt scan by a wide margin.
- **d5 wins under the window up to 80K** (45.4-45.8), then falls off a cliff at 95K (26.5) and
  127K (18.3). d6/d7 always lose except d6 at 79K (38.1, still below d5's 45.4).
- Therefore the middle preset is `int4 + 32K window + ctx 81920 + --draft-tokens 5` and it must
  pass the depth **explicitly**: the automatic rule (`>32768 -> 1`) would otherwise pick d1 (41.8).
