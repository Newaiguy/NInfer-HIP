# 视频可用数据（0.1.3r5，2026-11-20 本机实测）

 > 只放能对得上号、可复现的数字，每条都写清条件与二进制。凡标"单次"的只跑过一次。

## 0. 一句话

一个 `infer-serve.exe` / `infer-mini.exe` 同时具备：256K 上下文 + KV 记忆（分页滑窗 + 归档召回）、
MTP 投机解码、视觉、OpenAI 兼容 API，且**原生 tool_calls**（非流式/流式/多轮回传都已验证）。

## 1. 三档常规配置（都实测过，可直接照抄）

| 档位 | 参数要点 | decode 实测 | 适用 |
|---|---|---|---|
| 短 ≤16K | `--kv bf16 --draft-tokens 5 --ctx 16384` | 8K **92.7**、16K **81.8** t/s | 日常对话、代码片段、agent 短会话 |
| 中 ≤80K | `--kv rk4v4-e8 --ctx 81920 --resident-tokens 32768 --sink-tokens 2048 --archive-mib 4096 --kv-retrieval 1 --draft-tokens 5` | 63K **45.8**、79K **45.4** t/s | 单仓代码阅读、中等文档问答 |
| 长 ≤256K | 同上但 `--ctx 262144 --draft-tokens 1` | 196K **41.6**、250K **40.3** t/s | 全仓/长文档、超长历史 |

prefill 全程 530–620 t/s。硬件 RX 9070 XT（16 GB）/ ROCm 7.2；模型 Qwen3.5-27B IQ3_S。

## 2. 为什么三档要分（可讲的机制）

- MTP 每步要重扫窗口里所有列：深度 5 在 ≤80K 有净收益，到 95K 崩（26.5 vs 深度 1 的 42.9），
  所以长档固定深度 1。
- KV 量化只改"每 token 占多少"：bf16 65,536 B/token，int4 17,920 B/token。
  短对话 KV 扫描可忽略，bf16 少一层反量化反而更快；长档必须 int4 才装得下。

## 3. 容量（无窗口，纯文本）

| dtype | 可用上限 | 依据 |
|---|---|---|
| bf16 | 约 **49,974** token | 52,288 起报 `largest capacity ... about 49974` |
| int4（rk4v4-e8） | 约 **181,849** token | 188,000 起失败；180,000 正常 |

开 KV 窗口后 int4 可到 262,144 上下文（常驻 32K + 归档 4 GiB）；bf16 开窗口只能到约 128K，
196K 起 `hipErrorLaunchFailure`。`--kv int8` **未实现**（4.7–6.6 t/s 且答错），不要用。

## 4. 与上游 llama-kvmem 对拍（同机同卡同 prompt，双方都开 MTP）

| 项目 | 本引擎 | 上游 |
|---|---|---|
| 准确率（8 例） | 8/8 | 8/8 |
| prefill 196K | 537.9 t/s | 331.9 t/s |
| prefill 250K | 533.5 t/s | 323.1 t/s |
| decode 250K | **40.3 t/s** | 32.9 t/s |

口径：**prefill 快 1.4–1.6×；decode 全面更快，250K 档 +22.4%**。

## 5. 工具调用（r5 新增，可现场演示）

请求带 `tools` → 返回：

```json
{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
 "tool_calls":[{"id":"call_0","type":"function",
   "function":{"name":"get_building_count","arguments":"{\"park\":\"熙街\"}"}}]}}
```

把 `role:"tool"` 结果回传，模型续答「熙街共有 **46** 栋楼（另外还有 12 家店铺）」。
流式响应同样给出 `tool_calls`，`finish_reason` 一致。sils 的 Hermes / 任何 OpenAI 客户端直接可用。

## 6. 视觉

- 九图空间/颜色回归：artifact / patch-only BF16 / 全 BF16 **三档各 9/9**（r7 二进制复验 9/9）。
- 真实照片 16 问：**16/16**，与 llama.cpp + mmproj 的 16/16 打平。
- 已知：视觉开启会多占约 1.3 GB，长档建议 `--no-vision`，可多出约 30K 上下文容量。

## 7. 长思考收敛（鹈鹕 SVG）

- 64K ctx + MTP + xhigh + 官方采样：**3 次全部产出完整 HTML**（274/274/278 KB），800/800/752 秒。
- 反例：262K ctx 下同题 29 分钟未出正文 —— 不要拿它当证据。

## 8. 引用注意

- 旧文档里的 **105 t/s** 是 r3 二进制在 ctx 8192 短预设下的数字；r5 同档（16K、bf16、深度 5）是 **92.7**。
  视频统一用 r5 的数，别混用。
- 深度 5 只在 ≤80K 成立；长上下文一律深度 1。
- `--max-context` / `--kv-dtype` / `--host-kv-mib` 是 r4 的旧参数，r5 不认（会被忽略，不报错）——演示前务必确认。

## 9. 复现命令

CLI：`infer-mini.exe model.ninfer --chat --kv bf16 --ctx 16384 --spec mtp --draft-tokens 5 --effort medium`

API：`infer-serve.exe model.ninfer --host 127.0.0.1 --port 8099 --model-name ninfer-qwen3 --ctx 65536 --resident-tokens 32768 --sink-tokens 2048 --archive-mib 2048 --kv rk4v4-e8 --spec mtp --draft-tokens 5 --effort medium`

原始记录：`build\_window_sweep.json`、`build\_midctx_sweep.json`、`build\_speed_matrix.json`、
`build\_capacity_text_only.json`、`build\_acc3_ours_mtp.txt`、`build\_acc2_upstream.txt`。
版本基线：0.1.3r5，源码 `da6a473`，`infer-mini.exe` sha `d689a367…`、`infer-serve.exe` sha `ad71a07f…`。
