# 验证记录（0.1.0-pre-r2）

> **0.1.2 追加（2026-11-13，最高优先级）**：0.1.1 的发布二进制带着一个 KV 召回回归，
> 0.1.2 修好并重新跑完全部门禁与场景测试。二进制口径为
> `infer-mini.exe` 191,703,040 B / `infer-serve.exe` 192,038,912 B（sha256 见 VERSION.txt）。
>
> ## 0.0 0.1.2 门禁、召回与场景矩阵
>
> | 项目 | 命令 | 结果 |
> |---|---|---|
> | LayerNorm 正式用例 | `build/_build_layer_norm_test.py` | `OK layer_norm correctness` |
> | packed attention 正式用例 | `build/_build_plain_packed_attention_test.py` | `packed_softmax_attention: PASS` |
> | GATE 8k | `build/_gate.py --arms 8k` | decode **35.41** / prefill **932.7**（+5.3% / −0.5%）PASS |
> | GATE 96k | `build/_gate.py --arms 96k` | decode **25.01** / prefill **423.6**（−0.5% / −0.4%）PASS |
> | GATE mtp | `build/_gate.py --arms mtp` | decode **79.88** / prefill **854.7**（+2.5% / −0.7%）PASS |
> | 服务端场景矩阵 | `build/_serve_matrix.py` | **9/9**（含两处 500→400 修复） |
> | 视觉九图 | 三档各 **9/9**；真实照片 **4/4** 抽检 | 与 0.1.1 一致 |
>
> 长上下文（`build/_longctx_verify.py`，resident 32768 + archive 4096 MiB + 召回 + MTP2，
> 默认自适应分块）：**196K `540.4 / 29.77`、250K `534.9 / 30.50`、261K `534.1 / 29.79`，三档全部
> HIT**，ids 与参考逐 token 相同。同机同卡对上游 llama-kvmem：196K `336.5 / 24.28`、250K
> `323.2 / 29.15`，prefill 约为 1.6×，decode 持平到略快。
>
> 自适应召回分块（`build/_recall_split_check.py`、`build/_recall_real_check.py`）：
>
> | 场景 | 次峰比例 | 是否切分 | 结果 |
> |---|---|---|---|
> | 98K 单文档 needle（32768 与 20480 两档） | 1.5% | 否 | 召回页 704..815，逐 token 等于 GT |
> | 双文档，问次峰事实（切分开） | 82% | 是 | `NINFER-ATLAS-1111` PASS |
> | 双文档，问次峰事实（`SPLIT=0`） | 82% | 否 | 召回只剩 859..970，答成另一份文档的事实 |
> | 双文档，两台相似口令（`NINFER-<NAME>-<NNNN>`） | 82% | 是 | 答 `NINFER-ATLAS-2222`（模型拼接，已知限制） |
>
> 召回一致性（`build/_kvmatrix2.py`，oracle = 同 prompt 用装得下全文的窗口跑出的 token 序列）：
>
> | needle 深度 | 召回档（resident 32768 + archive）vs 完整上下文 | 说明 |
> |---|---|---|
> | 25% | 逐 token MATCH | `NINFER-ZEBRA-4471` |
> | 50% | 逐 token MATCH | `NINFER-KOALA-8823` |
> | 75% | 逐 token MATCH | `NINFER-VIPER-1092` |
> | 92% | 语义一致、空白切分不同 | needle 本就在窗口内，两条路径都答干扰指令 |
> | 98K mid（数字口令） | GT `45 669 24312 10846 8006 5434 12 19 19 22 16 248046`，召回页 704..815 | resident 20480 与 32768 两档均通过 |
>
> 96k 门禁臂的 ctx 从 110000 降到 100000：prompt 98,155 token、权重占 11.76 GB 后只剩
> 3.81 GB，110000 的池要 1.97 GB，引擎会**静默**切到半速路径（350 / 11.8 而不是 424 / 24.8）。
> 这不是功耗回归，池装得下就正常。
>
> **0.1.1 追加（2026-11-12）**：本文档主体是 pre-r2 的历史记录，速度结论仍成立。
> 0.1.1 的门禁与视觉回归单独记录在下面 §0，二进制口径为
> `infer-mini.exe` 191,703,040 B / `infer-serve.exe` 192,038,912 B（sha256 见 VERSION.txt）。
>
> ## 0. 0.1.1 门禁与视觉回归
>
> 全部串行、每例独立进程，`temperature=0.0`、`seed=42`、`max_tokens=60`。
>
> | 项目 | 命令 | 结果 |
> |---|---|---|
> | LayerNorm 正式用例 | `build/_build_layer_norm_test.py` | `OK layer_norm correctness`（默认 paired D=1152 路径；修复前四个对齐用例全部非有限值） |
> | packed attention 正式用例 | `build/_build_plain_packed_attention_test.py` | `packed_softmax_attention: PASS`（含新增段长 1 用例） |
> | GATE 8k | `build/_gate.py --arms 8k` | decode **36.50** / prefill **985.9**（+8.5% / +5.2%）PASS |
> | GATE 96k | `build/_gate.py --arms 96k` | decode **25.30** / prefill **428.4**（+4.2% / +1.2%）PASS |
> | GATE mtp | `build/_gate.py --arms mtp` | decode **82.11** / prefill **891.3**（+5.4% / +3.6%）PASS |
>
> 视觉九图（纯红 / 四象限 / 三色竖条 / 左右分色 / 上下分色 / 左上 / 右上 / 左下 / 右上角小块）：
>
> | 权重档 | 结果 | 备注 |
> |---|---|---|
> | 默认（artifact 内置量化视觉层） | **9/9** | 发布路径 |
> | patch-only BF16 覆盖 | **9/9** | 只覆盖 patch embedding |
> | 全 BF16 视觉覆盖 | **9/9** | 329 张量全覆盖 |
>
> 真实照片交叉验证（Wikimedia 原图 7 张；问题覆盖颜色/计数/OCR/形状/物体/地标/事件/
> 字母数/灯罩数/朝向/轮子数，共 16 问）：
>
> | 引擎 | 分数 | 视觉权重 |
> |---|---|---|
> | 本引擎（默认档） | **16/16** | artifact `q4/q5/q6_g64_fp16` |
> | llama.cpp `llama-mtmd-cli` | **16/16** | BF16 mmproj |
>
> 结论：两者打平，但本引擎用的是更低的量化精度，所以"打平"不等于数值等价；
> 量化损失目前没有硬用例能量出来（见 docs/VISION.md §4）。
>
> 泛化用例（不在九图集内，全部正确）：320×320 红块右下、480×360 非正方形红块左下、
> 一图两目标（红左上 + 蓝右下）、一请求两图、48×48 小目标（右上 / 左下）。
>
> 复现命令：`python build/_vprobe_cfg.py red big_tl big_tr big_bl corner_tr quad4 bars half tb`；
> 加覆盖档：`python build/_vprobe_cfg.py NINFER_VISION_WEIGHTS=D:\...\vision-bf16 red ...`。
> 注意本仓库 blob 以 CRLF 存储，`git diff --check` 会把所有新增行报成 trailing whitespace，
> 属仓库风格噪声。

全部串行跑，同一时刻只有一个 GPU 作业。引擎 `bin/ninfer_mini_nordc.exe`（193,521,152 B，
sha256 `011EBBAC…674FB44`），graph on，KV = `rk4v4-e8`（int4 K / int4 V），
artifact = `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`。

## 1. 正确性 + 速度矩阵（发布口径）

任务：语料中藏唯一串（或要求数数），判据 = 首 token 之后是否逐字符命中。
命令模板：

```bat
python probe_rebuilt\run_capture.py build\infer_mini.exe %ART% ^
  --tasks tasks_<x>.txt --ctx <N> --max-new <M> --kv rk4v4-e8 [--spec mtp --draft-tokens 2]
```

> 0.1.2 起 CLI 引擎叫 `build\infer_mini.exe`（旧名 `infer_mini_nordc.exe` 仅存在于历史目录）。

| # | 语料 | prompt tokens | --ctx | prefill t/s | decode t/s | 判定 |
|---|---|---|---|---|---|---|
| A | tasks_t500 | 529 | 4096 | 401.2 | 36.58 | ⚠ 语料本身是 GBK 乱码，模型答「appears to be mojibake」= 行为正确 |
| B | tasks_cp8km | 5,850 | 16384 | 848.5 | 35.37 | ✅ `7F3K9Q` |
| C | tasks_long | 31,483 | 32768 | 609.5 | 32.29 | ✅ 通顺技术摘要 |
| D | tasks_cp64km | 48,378 | 65536 | 504.6 | 30.12 | ✅ `7F3K9Q` |
| L | tasks_cp64km | 48,378 | 49152 | 509.5 | 29.98 | ✅ `7F3K9Q` |
| K | tasks_fw64k | 65,135 | **65536** | 441.6 | **28.36** | ✅ 数字列表 |
| N | tasks_fw96k | 98,155 | **98304** | 345.76 | **25.29** | ✅ 数字列表 |
| **M** | **tasks_true128kL** | **126,329** | **126464** | **293.97** | **22.91** | ✅ **`K8M2Q9`** |
| H | tasks_t3kL | 2,845 | 8192 | 797.8 | **77.82**（MTP dw2） | ✅ 1…95 |
| I | tasks_cp32kL | 24,251 | 49152 | 663.4 | **64.49**（MTP dw2） | ✅ 数字列表 |
| P | tasks_cp64km | 48,378 | 65536 | 506.12 | **34.91**（MTP dw2） | ✅ `7F3K9Q` |
| R | tasks_fw64k | 65,135 | 65536 | 438.05 | **41.58**（MTP dw2） | ✅ 数字列表 |
| O | tasks_fw96k | 98,155 | 98304 | 349.15 | **33.98**（MTP dw2） | ✅ 数字列表 |
| J | tasks_zh_small | 29 | 2048 | 142.1 | 37.29 | ✅ 中文通顺 |
| —— 以下为「--ctx 越预算」对照（不是推荐口径） —— |
| E | tasks_fw64k | 65,135 | 131072 | 354.7 | 14.95 | ✅ 数字列表 |
| F | tasks_fw96k | 98,155 | 131072 | 266.2 | 8.37 | ✅ 数字列表 |
| G | tasks_true128k | 126,306 | 131072 | 221.7 | 10.12 | ✅ `K8M2Q9` |
| Q | tasks_true128kL | 126,329 | 126464 | 238.31 | **10.59**（MTP dw2，越预算反例） | ✅ 接受率 2/2 |

**M vs G 是本版最重要的实测**：同一份 126K 提示，`--ctx 131072`（提交 13.67 GiB）→ 10.12 t/s，
`--ctx 126464`（提交 13.59 GiB）→ **22.91 t/s** ⇒ **把 `--ctx` 卡到实际需要的下限 = 免费 2.26×**。
复测三次：22.91 / 22.93（另加 chunk 512）/ 23.01，可复现。根因与更多证据见 `PERF-CAPACITY.md`。

**Q 是 MTP 的越线反例**：MTP 固定多提交 0.57 GB（mtp 权重 348 MB + draft state 190 MB + verify 图 73 MB），
126K 时总提交 14.12 GiB 越线 ⇒ 每轮 3 token 全接受（2/2）却只有 10.59 t/s，比 AR 的 22.91 慢一倍。

## 1.1 长上下文深度矩阵（0.1.2 复核）

目的：确认「不开 MTP、开窗口+归档+召回」时，196K 与接近 262K 上限的深度是否仍然**速度不降、精度不变**。
配置 `--resident-tokens 32768 --sink-tokens 2048 --kv-retrieval 1`、`NINFER_KV_ARCHIVE_MIB=4096`。

| 语料 | prompt tokens | --ctx | prefill t/s | decode t/s | 判定 |
|---|---|---|---|---|---|
| tasks_hh192k | 196,423 | 262144 | 540.4 | 29.77 | ✅ 与 GT 逐 token 相同 |
| tasks_hh256k_trim | 251,201 | 262144 | 534.9 | 30.50 | ✅ 同上 |
| tasks_hh262k | **260,909** | 262144 | 534.1 | 29.79 | ✅ 同上 |

三档 IDS 均为 `45 669 24312 10846 8006 5434 12 19 19 22 16 248046`。深度 196K→261K，
prefill 变化 1.2%、decode 变化 2.4% ⇒ **长上下文不是靠牺牲速度换来的**。

上限说明：`--ctx` 不能超过模型 `text.max_position_embeddings`（本模型 262,144），
超了在 `startup.cpp:752-755` 直接抛 `max_context exceeds the configured position capacity`。
所以「262K 可用」的准确口径是 **≤262,144 token，实测到 260,909（99.5%）**；原始
`tasks_hh256k.txt` 约 264K token，天生超限，必须按 3.9995 chars/token 重新裁剪才能测。

同语料与上游 llama-kvmem 对拍（同机同卡、都开 MTP2）：196K 上游 336.5 / 24.28，本引擎 540.4 / 29.77；
250K 上游 323.2 / 29.15，本引擎 534.9 / 30.50；两边都命中口令。详见 `KVMEM.md` §11.3。

### 1.2 准确率对拍：8/8 对 8/8（0.1.2）

问题："性能好但准确率会不会低于上游？" 做法：8 个用例、同 prompt、双方同 32K KV 预算、贪心解码，
判"答案是否命中语料口令"（不做逐 token 比较，因为上游 KV 是 `q8_0`/`q4_0`、本引擎 `rk4v4-e8`）。

| 用例 | 我们 | 上游 llama-kvmem | 我们 prefill/decode | 上游 prefill/decode |
|---|---|---|---|---|
| q25 | PASS | PASS | 557.3 / **41.05** | 384.6 / 25.14* |
| q50 | PASS | PASS | 558.7 / **34.78** | 378.2 / 32.85 |
| q75（窗口内，不召回） | PASS | PASS | 558.2 / **41.51** | 379.0 / 33.25 |
| q92（窗口内，不召回） | PASS | PASS | 559.0 / **35.32** | 379.2 / 32.27 |
| mid（50%） | PASS | PASS | 558.7 / **35.18** | 378.6 / 33.33 |
| multi（双文档） | PASS | PASS | 550.6 / **41.53** | 361.0 / 33.01 |
| 196K | PASS | PASS | 536.6 / **40.01** | 331.9 / 32.99 |
| 250K | PASS | PASS | 532.6 / 32.42 | 323.1 / 32.93 |

\* 上游 q25 是它的预热例，后续稳定 32–33。

**8/8 对 8/8，逐例答案一致 ⇒ 准确率不低于上游**。速度**双方都开 MTP2**：prefill 1.4–1.6×，
decode 除 250K 档持平（32.42 vs 32.93）外全部更快（35–42 vs 32–33）。我们 KV 是 int4/int4、
上游 q8_0/q4_0，每 token 读的 KV 字节更少，decode 优势由此而来；250K 档被归档页换入换出吃掉，
是下一步 decode 优化的重点。

> 对拍坑：①`tasks_needle_q*` / `tasks_needle_mid_q` / `tasks_hh*` 的中文问句原本是 U+FFFD 乱码，
> 模型只能读英文诱饵 `List the integers 1..400`，早期把这种回答误判成召回失败；回归现统一用
> `build/_mk_fixed_corpora.py` 生成的 `tasks_fix_*.txt`。②第一版对拍只给上游开了 MTP2、没给我们开，
> 得出的"decode 比上游低 10%"是配置不对称造成的假象 —— **对拍必须逐项核对两侧启动参数**。

## 2. MTP 倍率（必须用同语料的 AR 基准算）

| 语料（真实上下文） | ctx / 容量 | AR decode | MTP dw2 decode | 倍率 |
|---|---|---|---|---|
| tasks_t3kL（2,845） | 8192 | 36.20 | **77.82** | **2.15×** |
| tasks_cp32kL（24,251） | 49152 | 21.18 ⚠ | **64.49** | 3.05× ⚠ |
| tasks_t3k（2,809） | 81920 | 28.34 | **56.03** | **1.98×** |
| tasks_fw64k（65,135） | 65536 | 28.30 | **41.58** | **1.47×** |
| tasks_cp64km（48,378） | 65536 | 30.12 | **34.91** | 1.16× |
| tasks_fw96k（98,155） | 98304 | 25.18 | **33.98** | **1.35×** |
| tasks_true128kL（126,329） | 126464 | 22.91 | 10.59 ❌ | 越预算（提交 14.12 GiB） |
| tasks_t3k（2,809） | 131072 | 27.72 | 41.46 | 1.50× |

三条要点：① MTP 的收益取决于**真实上下文长度**与**是否越过驻留预算**；
② 直到 96K 真实上下文 MTP 仍是净赚（1.35×），126K 因多 0.57 GB 越线而不可用；
③ MTP 输出与 AR **逐 token 相同**（各臂 IDS 完全一致，例如 126K：`16 11 17 11 18 11 …`）。

⚠ tasks_cp32kL 的 AR 21.18 与同容量 48K 提示的 29.98 不一致（疑为单次运行离群，复测前不要把 3.05× 当结论；
t3kL 的 2.15×、fw96k 的 1.35× 是稳定值）。
MTP 只在 **`--draft-tokens 2`** 有收益；`3` 及以上草稿链自回归退化，净亏。

## 3. 与 llama.cpp 同卡硬对照（llama build 9548，ROCm）

二进制 `D:\Qwen3-27B-Deploy-migrate\llama-hip-official\llama-completion.exe`，
模型 `S:\NLP_Qwen\Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`（同权重族），
flags `-ngl 99 -ctk q4_0 -ctv q4_0 -fa on --no-warmup --no-display-prompt`。

**⚠ 测量纪律（本版踩过的坑）**：llama 的数字必须在**与我们同一时段**测。
本文档早期的 llama 值（65K 11.47、126K 7.15）是在机器驻留预算偏低时测的，llama 自己也被降级了，
虚低 43%~56%。下面是**同一时段背靠背重测**的值（先 llama 四臂，紧接着我们四臂，中间无其它作业）：

| prompt tokens | llama prefill | llama decode | 我们 prefill | 我们 decode（AR / MTP dw2） |
|---|---|---|---|---|
| 247（`-c 4096`） | 186.12 | 26.54 | — | **37.3** / 77.8 |
| 2,759（容量 81920） | 704.03 | 26.68 | 800.4 | 28.34 / **56.03** |
| 65,135（`-c 65536`） | **752.34** | **20.17** | 438.10 | 28.30 / **41.58** |
| 126,294（`-c 131072`） | **574.83** | **16.32** | 294.64（ctx126464） | 22.95 / 不可用 |
| 126,294（`-c 126464`） | 565.15 | 16.19 | 同上 | 同上 |

结论（发布口径）：
- **decode 稳定领先约 1.4×**：4K 37.3 vs 26.5（1.41×）、65K 28.30 vs 20.17（1.40×）、126K 22.95 vs 16.19（1.42×）。
  三档倍数一致，说明领先来自引擎的 GEMM/GEMV 效率，而不是某个特定长度的巧合。
- **开 MTP 后是中长上下文 2 倍级**：65K 41.58 vs llama 20.17 = **2.06×**；短上下文 77.8 vs 26.5 = 2.9×。
  （llama 上游 build 9548 没有 MTP；用户脚本里 G: 那个补丁分支的 `draft-mtp n-max 3` 属另一套实现，
  按它同档标称 ~37 t/s 算，我们的 41.6 仍高约 12%。）
- **prefill 我们落后 0.58×（65K：438 vs 752）/ 0.51×（126K：295 vs 575）** —— 下一版主目标。
- llama 在 126K 不需要收紧 `-c`（16.32 vs 16.19），因为它固定开销小、同上下文只占约 13.1 GiB；
  **我们固定开销多 573 MB（state 320 + ws 240 + graph 13），所以我们必须把 `--ctx` 贴着 prompt 设。**
- `--verbose` 日志：`offloaded 65/65 layers to GPU`，llama 没有任何层在 CPU；
  `--no-mmproj-offload` 只影响视觉投影器（931 MB，仅图片输入时使用），与文本速度无关。

## 4. 速度随 `--ctx` 的台阶（决定"能用多大上下文"）

关键：**台阶位置 = 引擎总提交量，不是 ctx 数值**；而提交预算**当天会漂移约 1 GiB**。

同一 48,378 token 提示，只改 `--ctx`：

| --ctx | 引擎总提交 | 上午扫 | 下午扫 | 2.8K 短提示 |
|---|---|---|---|---|
| 49152 | 12.16 GiB | 30.1 | — | 35.9 |
| 65536 | 12.58 GiB | **30.1** | 30.1 | 28.2 |
| 69632 | 12.64 GiB | **16.8** ← 越线 | — | 28.0 |
| 81920 | 12.85 GiB | 16.7 | 16.7 | 28.3 |
| 98304 | 13.12 GiB | 16.0 | **29.91** | — |
| 106496 | 13.26 GiB | — | **29.98** | — |
| 114688 | 13.39 GiB | — | **29.86** | — |
| 122880 | 13.53 GiB | — | **29.99** | — |
| 131072 | 13.67 GiB | 16.9 | **17.12** ← 越线 | ~28 |

⇒ 上午预算是 12.6 GiB（69632 就腰斩），下午是 13.5~13.6 GiB（122880 仍是满速）；
**同一条命令在两个时刻可以差 1.8×**。这就是 pre-r1「126K 有时 23.5 有时 10」的真相。

⇒ 短提示只有 -22% 的固定容量税（8K 36.3 → 64K 28.2），越过悬崖也不再恶化；
长提示在悬崖处腰斩 ⇒ 掉速与「每 step 读出多少 KV」成正比，与池子总大小无关。

机制、预算算术与已排除假设见 `PERF-CAPACITY.md`。

## 5. 物理上限（别再算错）

- 短上下文 AR：权重 11.76 GB ÷ 640 GB/s ≈ 54 t/s 硬件上限；实测 37.3 = 70%。
- MTP：每读一次权重出 ~2.5 token ⇒ 上限 ~135 t/s，实测 77.8 = 58%。
- 126K AR：权重 11.76 + int4 KV 2.35 ≈ 14.1 GB ÷ 640 GB/s = 22.1 ms ⇒ **45 t/s 上限**；
  实测 22.9 t/s = **51%**（pre-r1 记的 10.1 只有 22%，那是 `--ctx` 开大导致的池降级，不是硬件限制）。
- 96K AR：权重 11.76 + int4 KV 1.76 ≈ 13.5 GB ÷ 640 GB/s = 21.1 ms ⇒ 47 t/s 上限；实测 25.2 = 54%。
- 「能用多大」不是分配问题：`cudaMalloc` 是虚拟超额提交，干净进程可分配到 ≥24 GiB；
  限制是驱动愿意把多少页留在显存（实测 12.6~13.6 GiB，随桌面状态漂移）。

## 5.5 交互式聊天模式（pre-r3 新增，实测）

命令：`ninfer_mini_nordc.exe <artifact> --chat --ctx 16384 --kv rk4v4-e8 --spec mtp --draft-tokens 2 --max-new 512`
（等价于双击 `harness/launchers/chat.bat` 选 1）。整段 history 每轮重发，靠前缀复用只 prefill 新增部分。

| 轮次 | prompt tokens（复用/新增） | 新增 tokens | prefill | decode |
|---|---|---|---|---|
| 1（"你好"） | 13（0 / 13） | 14 | 112 | **57.4** |
| 2（"用三句话介绍 AMD RDNA4 的显存带宽特点"） | 55（**27 reused** / 28） | 118 | 278 | **60.4** |
| 单轮事实题（"法国的首都是哪里"） | 22（0 / 22） | 1（答"巴黎"） | 118 | 28.4（1 token，不具代表性） |

- 流式输出正常（content 边生成边打印，reasoning 走 `/think on` 的 `[thinking]` 通道）。
- **前缀复用生效**：第 2 轮 55 token 中 27 token 命中驻留前缀，prefill 只处理 28 token（278 t/s）。
- `/stats` 输出 `[stats] turns=2 history_messages=4 last_decode_tps=60.4379`。
- 非 chat 路径未回归：`tasks_cp8km` → `7F3K9Q` ✅，prefill 842.8 / decode 35.7（与 pre-r2 同）。

## 6. 已知不可用 / 未验证

| 项 | 状态 |
|---|---|
| int8 KV (`rk8v4`) | 非确定性（前 12 token 一致后分叉），**不要用于长上下文** |
| fp8 KV | QK 是 `ninfer/src/ops/common/mma.cuh:584-590` 的 `__builtin_trap()` 桩 |
| dflash2 | prefill 406 / decode 5.62，输出分叉，坏路径 |
| 256K | int4 KV 也装不下（需 KV offload），未验证 |
| 126K + MTP | 多提交 0.57 GB 越预算 ⇒ 10.6 t/s，比 AR 慢一倍；≥100K 一律关 MTP |
| prefill ≥64K | 落后 llama 1.7–1.9×，下一版处理 |
