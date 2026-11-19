# NInfer-HIP 0.1.3r4 — 原生工具调用 + 默认中等思考

发布日期：2026-11-19

## 这一版做了什么

| 变化 | 说明 |
|---|---|
| 服务端换成完整实现 | `infer-serve.exe` 从自写迷你版换成 `infer/src/serve` 的完整 OpenAI 服务端 |
| 原生 tool_calls | 非流式、流式分片、多轮回传全通；`finish_reason` 为标准 `tool_calls` |
| 默认思考 medium | 模板默认从 xhigh 改为 medium，请求级 `reasoning_effort` 仍优先 |
| 文档 | 新增 `docs/SERVE_TOOLS.md` |

## 验证

服务端协议矩阵 **10/10**：models、chat、SSE 流式（首字节 0.018 s）、tools、视觉单图、视觉双图、多轮召回、图片段缺 type（400）、坏 payload（400）、超长 prompt（400 + `context_length_exceeded`）。

文本门禁 PASS：

| 臂 | decode t/s | prefill t/s |
|---|---|---|
| 8k | 36.05 | 946.7 |
| 96k | 25.34 | 421.2 |
| mtp | 105.88 | 846.4 |

## 用法

```bat
bin\infer-serve.exe <model.ninfer> --host 127.0.0.1 --port 8099 ^
  --model-id ninfer-qwen3 --max-context 65536 --kv-dtype rk4v4-e8 ^
  --spec mtp --draft-tokens 5 --host-kv-mib 2048 --greedy ^
  --chat-template chat_template_medium.jinja
```

Hermes 侧填 `http://<IP>:8099/v1`，模型名 `ninfer-qwen3`，工具会以标准 `tool_calls` 回来。

## 校验和

见 `MANIFEST.sha256`；`bin/infer-serve.exe` sha256 `39123f589f6ac9a86c43768abfd1c1303b559032cdd93f353c482d16486b6953`。

## 已知限制

- MTP 草稿深度仍按 `--ctx` 分档（≤64K 用 5，>64K 用 1），逐请求按实际 prompt 长度自适应尚未实现。
- 视觉九图回归待补：本轮验证的是服务端图片通路，逐题空间理解回归等自适应完成后统一重跑。
