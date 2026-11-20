# infer-serve 启动与对接（0.1.3r5）

一个二进制同时具备：KV 记忆（分页窗口 + 主机归档，最高 262144 上下文）、MTP 投机解码、
原生 OpenAI tool_calls（非流式 + 流式 + tool 结果回传）、视觉。

## 1. 启动行

默认（65536 上下文，日常 agent）：

    bin\infer-serve.exe <artifact.ninfer> --host 0.0.0.0 --port 8099 ^
      --model-name ninfer-qwen3 --ctx 65536 --resident-tokens 32768 --sink-tokens 2048 ^
      --archive-mib 2048 --kv rk4v4-e8 --spec mtp --draft-tokens 5 --effort medium

长上下文（262144，需主机归档）：

    bin\infer-serve.exe <artifact.ninfer> --host 0.0.0.0 --port 8099 ^
      --model-name ninfer-qwen3 --ctx 262144 --resident-tokens 32768 --sink-tokens 2048 ^
      --archive-mib 4096 --kv rk4v4-e8 --spec mtp --draft-tokens 1 --effort medium

## 2. 参数（本二进制只认这些）

| 参数 | 说明 |
|---|---|
| --host / --port | 监听地址，局域网对接用 0.0.0.0 |
| --model-name | /v1/models 与响应里的模型名，默认 ninfer-qwen3 |
| --ctx | 逻辑上下文上限（真实 token），262144 需配 --archive-mib |
| --resident-tokens | 常驻窗口大小，建议 32768 |
| --sink-tokens | 头部保留 token 数，建议 2048 |
| --archive-mib | 主机归档上限（MiB），2048~4096 实测可用，6144 会 OOM |
| --kv | KV 量化类型，默认 rk4v4-e8 |
| --spec mtp | 开启 MTP 投机解码 |
| --draft-tokens N | 草稿深度；<=64K 用 5，长档用 1 |
| --effort | 默认思考等级（low/medium/high/xhigh），本版默认 medium |
| --think / --no-think | 强制开/关思考，请求里的 reasoning_effort 优先 |
| --kv-retrieval / --kv-recall-pages | 归档召回开关与页数（长档建议开启） |

旧文档里的 --max-context、--kv-dtype、--model-id、--greedy、--host-kv-mib 属于 r4 的
"完整 serve" 二进制，本版已不再使用。

## 3. tool_calls

请求体与 OpenAI 一致：tools 数组传 function 定义，返回 finish_reason = "tool_calls"，
message.tool_calls[0].function.name 为工具名，arguments 是 JSON 字符串。把结果以
{"role":"tool","tool_call_id":...,"content":"46"} 回传即可续写。

## 4. 已验证（2026-11-19）

- 262144 上下文 + KV 记忆 + MTP 起得来，/v1/models 正常，问答正常。
- 工具：非流式 tool_calls、SSE 流式 tool_calls delta、tool 结果回传后续答，全部通过。
- 视觉九图 9/9（red / quad4 / bars / half / tb / corner_tr / big_tl / big_tr / big_bl）。
- 默认思考等级为 medium；请求级 reasoning_effort 仍可覆盖。

## 5. 对接

Hermes / 任意 OpenAI 客户端：base_url = http://<主机IP>:8099/v1，api_key 任意，
model = ninfer-qwen3。

## 6. 短档也必须有 KV 窗口（2026-11-19 复核）

65536 上下文在 16 GB 卡上不开窗口时，引擎要 4.87~5.49 GB，可用只有 3.5~3.8 GB
（`the largest capacity this device can hold is about 41956~51333 tokens`）。
所以 65536 也必须带 `--resident-tokens`（建议 16384~32768）。
缺参数时服务会直接报 `[serve] fatal: ...` 并以退出码 1 结束，**不会再静默降级到 2048**。

## 容量与速度（2026-11-20 实测，int4 KV）

- 纯文本上限：`--kv rk4v4-e8` 到 180,000 起得来，188,000 起不来（引擎报约 **181,849 token**）；`bf16` 只能到 49,152 起得来。
- 开视觉（不给 `--no-vision`）：int4 到 98,304 可直接用；131,072 及以上视觉尝试会失败，只有纯文本重试能起。
- **速度**：不开窗口时 131,072 的 decode 会掉到 9.5 t/s；开 32K 常驻窗口 + 4 GiB 归档后，262,144 仍有 **40.3 t/s**（prefill 531）。长上下文必须开窗口，不只是为了容量。
- **草稿深度**：≤32,768 用 5（8K 63.7 t/s），>32,768 用 1（64K 37.4，96K 30.6）。`--draft-tokens 0` 走自动档，源码阈值已改为 32K。
- 数据来源：`docs/CAPACITY.md` 与 `docs/SPEED.md`，都由脚本从实测 json 生成，不要手改。

## 两档常用配置（2026-11-20 实测）

**短对话（≤16K）：bf16 + 深度 5**

    --kv bf16 --draft-tokens 5 --ctx 16384

8K 92.7 t/s、16K 81.8 t/s。bf16 在这里反超 int4（同一 8K 下 int4 只有 63.7），因为一步的开销主要是 11.7 GB 权重，KV 扫描可忽略，反量化纯属额外开销。注意 bf16+d5 在 24K 已掉到 30.7、32K 只剩 11.9，**别把它用在 16K 以上**。

**长上下文（≤256K）：int4 + 窗口 + 深度 1**

    --kv rk4v4-e8 --ctx 262144 --resident-tokens 32768 --sink-tokens 2048 --archive-mib 4096 --draft-tokens 1

实测深度：d1 40.3 / d2 39.5 / d3 31.6 / d4 26.5 t/s ⇒ 262K 档用 d1（自动档在 >32K 时也正好取 1）。

### 窗口档深度：用 d1，不要用 d2

开窗口（32K 常驻 + 4 GiB 归档、int4）下同语料实测：63K d1 41.8 / d2 43.2；95K d1 42.9 / d2 34.9（复跑 35.3）；127K 42.1 / 41.6；195K 41.6 / 33.7；249K 40.3 / 39.5。
均值 d1 41.7、d2 38.6。d2 每步多扫一列是恒定成本，赚不赚看草稿命中率，而命中率随文本波动 ⇒ 只在个别语料赢 3%，另一些亏 19%。长上下文固定 d1（自动档 >32768 已取 1）。

### 三档常规配置（2026-11-20 实测，r7 / sha D689A367）

| 档 | 命令要点 | 实测 decode |
|---|---|---|
| 短（<=16K） | `--kv bf16 --ctx 16384 --spec mtp --draft-tokens 5` | 8K 92.7 / 16K 81.8 |
| 中（<=80K） | `--kv rk4v4-e8 --ctx 81920 --resident-tokens 32768 --sink-tokens 2048 --archive-mib 4096 --kv-retrieval 1 --draft-tokens 5` | 63K **45.8** / 79K **45.4** |
| 长（<=256K） | `--kv rk4v4-e8 --ctx 262144 --resident-tokens 32768 --sink-tokens 2048 --archive-mib 4096 --draft-tokens 1` | 195K 41.6 / 249K 40.3 |

- 中档必须**显式**给 `--draft-tokens 5`：自动档规则是 `ctx > 32768 取 1`，否则会掉到 d1（41.8）。
- 中档不要越过 80K：95K 同配置 d5 只剩 26.5、127K 18.3（d1 反而 41.4/42.9）。
- 短档 bf16 上限 16K（24K 30.7、32K 11.9）；长档 bf16 在 196K 会 `hipErrorLaunchFailure`，只能用 int4。
- `int8` 未实现（4.7-6.6 t/s 且答错），任何情况都不要用。
