# 工具调用（Tool Calling）使用说明

0.1.3r3 起，`infer-serve.exe` 换成完整的 OpenAI 兼容服务，支持原生 `tools` / `tool_calls`，
Hermes、OpenCode 这类按 OpenAI 协议工作的 agent 可以直接对接。

## 启动

用 `启动.bat` 菜单里的「API 服务」，或者手动：

```bat
bin\infer-serve.exe "%MODEL%" --host 127.0.0.1 --port 8099 --model-id ninfer-qwen3 ^
  --max-context 65536 --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 5 --host-kv-mib 2048 --greedy
```

要点：

- `--max-context` 决定窗口大小。agent 日常 32768/65536 足够；再大就要更多显存，且 `--draft-tokens` 应降到 1。
- `--draft-tokens` 合法范围是 **1..15**，没有 0（0 是 mini 的自动档写法，这里会被拒绝并打印 usage）。
- `--greedy` 关掉采样，适合抽取/判定类任务；要官方采样用 `--temperature 0.6 --top-p 0.95 --top-k 20`。

## 在 Hermes 里配置

- Base URL：`http://<本机IP>:8099/v1`
- Model：`ninfer-qwen3`
- 不需要额外适配层：模型接收 `tools` 后返回标准 `tool_calls`，`finish_reason` 为 `tool_calls`；
  把工具结果以 `role: "tool"` 回传后，它能接着把答案说完。

## 已验证

- 非流式：`tool_calls[0].function.name == "get_building_count"`，`arguments` 为合法 JSON。
- 流式：tool_calls 以 delta 分片下发，`finish_reason` 同为 `tool_calls`。
- 多轮：assistant 的 `tool_calls` + `role:"tool"` 结果回传后，模型给出最终答案
  （实测「熙街共有 46 栋楼」，prefix cache 命中 355 token）。
- MTP 同时可用：32768 窗口、深度 5 时 decode 约 80 tok/s。

## 思考等级

请求体的 `reasoning_effort` 支持 `low` / `medium` / `xhigh`（这是 artifact 模板允许的三档）。
服务端默认是模板默认（xhigh），不想让模型长时间思考就在请求里显式传 `medium`，
或者启动时加 `--no-thinking`。50 题实测：结构化任务 medium 优于 xhigh（0.935 对 0.840，
且 xhigh 有 3/50 次把 token 预算全部烧在思考上、返回空正文）。

## 已知限制

- 服务端按 `--max-context` 选不了 MTP 深度，只能手传 `--draft-tokens`（逐请求自适应是下一步）。
- `--vision` 打开后走的是同一套媒体通路；本机没有 FFmpeg，图片格式靠 WIC（PNG/JPEG/GIF/BMP/TIFF）。
- 工具名由模型自行选择，规则没写清时会调用错的工具；系统提示里把工具用途写具体能显著降低误调。

## 思考等级默认值（2026-11-19 起：默认 medium）

启动器的 API 行现在带 `--chat-template "%~dp0chat_template_medium.jinja"`：
该文件是从 artifact 内嵌 jinja 模板取出的原文，只把
`reasoning_effort|default('xhigh')` 改成 `default('medium')`，其余逐字未动。

实测（同一道逻辑题，prompt 差异来自注入的指令文本）：

| 请求 | prompt tokens | 思考 tokens | 总 tokens |
|---|---|---|---|
| 不传 reasoning_effort（走默认） | 48 | 576 | 629 |
| `reasoning_effort: "xhigh"` | 90 | 945 | 988 |
| `reasoning_effort: "low"` | 78 | 388 | 435 |

结论：默认已变为 medium；请求里显式给 `reasoning_effort` 仍然优先。
想临时换回 xhigh 或试 low，删掉 `--chat-template` 参数即可（不传就用 artifact 自带的 xhigh 默认）。
tool call 与该模板共存已验证：`finish_reason` 仍为 `tool_calls`。
