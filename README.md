# NInfer-HIP `0.1.3r2`

**在 AMD RX 9070 XT (RDNA4) + ROCm 7.2 for Windows 上运行 Qwen3.8-27B 的本地推理引擎。**
纯 HIP/C++ 实现，自带 KV 记忆（超长上下文）、MTP 投机解码、视觉多模态、OpenAI 兼容服务端。

> 一句话：单卡 16 GB 的 Windows 游戏卡，能跑 27B 模型，**262K 上下文**，8K 对话 **35.7 tok/s**，开 MTP 到 **105 tok/s**，还能看图。

---

## 这一版（0.1.3r2）改了什么

| 变化 | 说明 |
|---|---|
| 默认开启思考 | 启动即 `thinking=on`，不再需要手动加 `--think` |
| 思考等级可调 | `low` / `medium` / `xhigh`（模型模板支持的三档），CLI、菜单、API 三处都能改 |
| 思考长度放开 | 默认**不设上限**（旧版 2048 硬顶会截断答案），每轮总预算 8192 |
| 长答案不再被截 | 修复「思考吃光预算后正文被切」的问题 |
| API 支持思考设置 | 服务端 `--think/--no-think/--effort`，请求级 `reasoning_effort` 覆盖 |
| API 缺省预算修复 | 请求不带 `max_tokens` 时不再返回空内容 |
| 一键启动器 | `启动.bat`：菜单式，可选 CLI 聊天 / API 服务 / 思考等级 / 模型路径 |

---

## 30 秒开始

```bat
:: 1. 解压/克隆本仓库，确认模型文件存在
::    E:\ninfer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer

:: 2. 双击（或命令行运行）
启动.bat

:: 3. 选 1 → CLI 聊天（8K，MTP 自动）
::    选 2 → API 服务（262K + KV 记忆，端口 8099）
```

想直接命令行跑：

```bat
bin\infer-mini.exe  E:\ninfer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer ^
    --chat --ctx 16384 --kv rk4v4-e8 --spec mtp --draft-tokens 0 --think ^
    --max-new 8192
```

---

## 实测性能（RX 9070 XT 16 GB · ROCm 7.2 · 贪心解码）

| 场景 | 上下文 | prefill (tok/s) | decode (tok/s) |
|---|---|---|---|
| 短对话 AR | 8K | 948.4 | 35.69 |
| 短对话 **MTP draft 5** | 8K | 848.2 | **105.53** |
| 中长文档 AR | 96K | 423.3 | 25.26 |
| 长文档 + KV 记忆 | 196K | 540.4 | 29.77 |
| 长文档 + KV 记忆 | 250K | 534.9 | 30.50 |
| 长文档 + KV 记忆 | 261K | 534.1 | 29.79 |

- 262,144 token 是模型位置编码硬顶；靠 **KV 记忆**（sink + 常驻窗口 + 主机归档 + 自适应召回）把 192K/250K/261K 塞进 16 GB 显存。
- 与上游 `llama.cpp + kvmem` 同机对比：8 例长上下文问答**准确率 8/8 对 8/8**，prefill 快 **1.4–1.6×**，decode 持平到更快。
- 视觉：三档权重（量化 / 部分 BF16 / 全 BF16）九图定位各 **9/9**；真实照片 16 问对 llama.cpp **16/16 打平**。

---

## 能力

- **长上下文**：262K 上下文，KV 记忆自动把旧页归档到内存/主机，召回时按 query 打分挑页
- **MTP 投机解码**：draft 深度自适应（≤64K 用 5，>64K 用 4），8K 场景 3× 提速
- **视觉**：PNG/JPEG/GIF/BMP/TIFF，单图/多图/图文混排，空间位置与颜色可判读。这是**理解型多模态（VLM）**——看图说话、OCR、计数、判位置，**不生成图片/视频**；与 llama.cpp 走 mmproj 属同一类能力、同一套权重。
- **OpenAI 兼容服务**：`/v1/chat/completions`（含 SSE 流式）、`/v1/models`、`/health`
- **思考控制**：开/关 + 三档强度 + 预算（默认不限）

---

## 文档

| 文件 | 内容 |
|---|---|
| [`docs/USAGE.md`](docs/USAGE.md) | **使用教程**：安装、CLI、API、思考、视觉、长上下文配置、排错 |
| [`docs/PROJECT.md`](docs/PROJECT.md) | **项目全貌**：设计动机、架构、KV 记忆原理、全部实测数据、限制（写视频脚本看这个） |
| [`docs/KVMEM.md`](docs/KVMEM.md) | KV 记忆的实现细节与参数 |
| [`docs/VISION.md`](docs/VISION.md) | 视觉通路的实现与调试记录 |
| [`docs/BUILD.md`](docs/BUILD.md) | 从源码构建 |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | 门禁与验证矩阵 |
| [`VERSION.txt`](VERSION.txt) | 版本、二进制校验和、完整变更记录 |
| [`RELEASE_NOTES_0.1.3.md`](RELEASE_NOTES_0.1.3.md) / [`RELEASE_NOTES_0.1.3r2.md`](RELEASE_NOTES_0.1.3r2.md) | 发布说明 |

---

## 系统要求

- AMD RDNA4 GPU（本项目在 **RX 9070 XT 16 GB** 上验证；gfx1201）
- Windows 10/11 x64 + **ROCm 7.2**（含 HIP SDK），`C:\Program Files\AMD\ROCm\7.2\bin` 需在 PATH
- 16 GB 内存以上（长上下文归档会用到主机内存）
- 模型工件 `.ninfer`（Qwen3.8-27B，约 15 GB）

## 许可

见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)。
