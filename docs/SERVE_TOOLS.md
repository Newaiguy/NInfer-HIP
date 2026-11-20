# 服务端接入（0.1.3r5）

`infer-serve.exe` 同时提供 **OpenAI 兼容聊天接口**、**KV 记忆（超长上下文）** 和 **原生 tool_calls**。
一个进程、一份模型，CLI 与 API 使用同一套参数名。

## 启动

```bat
set "PATH=C:\Program Files\AMD\ROCm\7.2\bin;%PATH%"
infer-serve.exe "E:\ninfer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer" ^
  --host 127.0.0.1 --port 8099 --model-name ninfer-qwen3 ^
  --ctx 65536 --resident-tokens 32768 --sink-tokens 2048 --archive-mib 2048 ^
  --kv rk4v4-e8 --spec mtp --draft-tokens 5 --effort medium
```

256K 长档只需换三个值：`--ctx 262144 --archive-mib 4096 --draft-tokens 1`。

参数（只列这些，其余会打印 `ignoring unknown argument` 并忽略）：
`--host --port --model-name --api-key --ctx --resident-tokens --sink-tokens --archive-mib
--kv-retrieval --kv-recall-pages --kv --spec --draft-tokens --effort --think --no-think --vision --no-vision`

**注意**：`--max-context` / `--kv-dtype` / `--host-kv-mib` 是 0.1.3r4 的名字，本版不认，写了也不生效。

## Hermes / 其他 OpenAI 客户端

- Base URL：`http://<机器IP>:8099/v1`
- 模型名：`ninfer-qwen3`
- 不需要 API key（若设了 `--api-key`，客户端要带 `Authorization: Bearer <key>`）

## 工具调用

请求里给 `tools`（OpenAI 标准 schema），服务端会把工具定义渲染进 Qwen 模板，并把模型输出的
`<tool_call>` 解析成标准结构：

```json
{
  "finish_reason": "tool_calls",
  "message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [{"id": "call_0", "type": "function",
                    "function": {"name": "get_building_count", "arguments": "{\"park\":\"熙街\"}"}}]
  }
}
```

把工具结果按 `role: "tool"` + `tool_call_id` 回传即可续算。`tool_choice` 支持 `auto` / `none` / `required` /
指定函数；流式响应会在最后一个 chunk 给出同样的 `tool_calls`。

## 已验证（2026-11-20，sha AD71A07F）

非流式 tool_calls、流式 tool_calls、工具结果回传、纯文本、SSE 首字节 0.018 s、九图视觉、256K 启动与命中针。

## 已知限制

- `--kv int8` 未实现：能启动但 4.7-6.6 t/s 且答案错误，请不要用。
- `--kv bf16` 在 196K 起会 `hipErrorLaunchFailure`；长上下文请用 `rk4v4-e8`。
- 同一时刻只跑一个实例：16 GB 显存装不下两个。
