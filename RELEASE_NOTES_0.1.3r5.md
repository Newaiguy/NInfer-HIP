# 0.1.3r5 - 一个服务端，长上下文与工具调用兼得

## 这一版解决了什么

0.1.3r4 为了给 agent 提供 tool_calls，把随包的 `infer-serve.exe` 换成了完整版 OpenAI 服务端，
那个服务端没有 KV 分页，**256K 上下文在 API 上直接不可用**，文档里的 262K 也就成了空话。
r5 把 tool_calls 直接做进了「能 256K + KV 记忆」的那个服务端，于是：

- 一个 `infer-serve.exe`：256K 上下文 + KV 记忆 + 原生 tool_calls；
- 参数与 CLI 完全一致（`--ctx` / `--kv` / `--draft-tokens` / `--resident-tokens` / `--archive-mib`），
  坏值直接报错，不再是"看起来设了其实没生效"；
- `--kv` 第一次真正到达引擎：int4 KV 让无窗口容量从约 49K 提到约 181K。

## 已验证

| 项目 | 结果 |
|---|---|
| 非流式 tool_calls | `finish_reason="tool_calls"`，`get_building_count {"park":"熙街"}` |
| 流式 tool_calls | tool_calls delta + `finish_reason="tool_calls"` |
| 工具结果回传 | 模型续答「熙街共有 46 栋楼（另外还有 12 家店铺）」 |
| 256K 上下文 | `--ctx 262144 --resident-tokens 32768 --archive-mib 4096` 正常启动并命中针 |
| 视觉九图 | 9/9 |
| 文本门禁 | 8k 36.05 / 946.7，96k 25.34 / 421.2，mtp 105.88 / 846.4 |

## 性能（RX 9070 XT / ROCm 7.2 / Qwen3.5-27B IQ3_S）

| 档 | 命令要点 | decode |
|---|---|---|
| 短 <=16K | `--kv bf16 --draft-tokens 5 --ctx 16384` | 8K 92.7 / 16K 81.8 |
| 中 <=80K | `--kv rk4v4-e8 --ctx 81920 --resident-tokens 32768 --archive-mib 4096 --kv-retrieval 1 --draft-tokens 5` | 63K 45.8 / 79K 45.4 |
| 长 <=256K | `--kv rk4v4-e8 --ctx 262144 --resident-tokens 32768 --archive-mib 4096 --draft-tokens 1` | 196K 41.6 / 250K 40.3 |

prefill 全程 530-620 t/s。对比上游 llama.cpp + kvmem（同机同模型，同样开 MTP）：decode 多数档领先 12-25%。

## 升级注意

- 参数换了：`--max-context` → `--ctx`，`--kv-dtype` → `--kv`，`--host-kv-mib` 不再需要（旧名会被忽略并打印提示，
  不会报错，但也**不会生效**，务必改掉）。
- 每次只跑一个服务实例：16 GB 显存装不下两个，启动器已加单实例检查。
- `--kv int8` 不要用；`--kv bf16` 上限约 49K（无窗口）或 128K（开窗口）。

## 文件

| 文件 | 大小 | sha256 |
|---|---|---|
| `bin/infer-mini.exe` | 191,967,232 | see MANIFEST.sha256 |
| `bin/infer-serve.exe` | 192,325,120 | see MANIFEST.sha256 |

源码基线：`da6a473`（`ninfer/` 工作树干净）。二进制构建于 2026-11-20 10:35，晚于最后一次源码修改（10:00）。
