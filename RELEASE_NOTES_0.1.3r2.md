# NInfer-HIP 0.1.3r2

**0.1.3 的生产修订版**：让"打开就能聊、答案不被截、思考可调、API 能设"。

- 发布页：https://github.com/Newaiguy/NInfer-HIP/releases/tag/v0.1.3r2
- 基线：0.1.3（标签 `v0.1.3-source`，提交 `7bfb4b5`）

---

## 1. 一句话

把 0.1.3 的引擎能力（262K 上下文 / KV 记忆 / MTP 105 tok/s / 视觉 9/9）包装成**可直接使用**的产品：
默认开思考、思考等级三档可调、思考不限长、API 支持同样的开关，并附一键启动器和完整教程。

---

## 2. 变更

| 变更 | 说明 |
|---|---|
| 默认思考开启 | `think` 默认 `true`，banner 显示 `thinking=on` |
| 思考等级 | `low` / `medium` / `xhigh`（模型模板支持的三档），CLI `--effort`、聊天 `/effort`、启动器菜单 4、API `reasoning_effort` |
| 思考不限长 | `--think-budget` 默认 `0` = 不限；旧版 2048 硬顶会截断正文 |
| 总预算 8192 | 启动器默认 `--max-new 8192`，长答案可完整生成 |
| API 思考参数 | 服务端 `--think/--no-think/--effort`；请求级 `reasoning_effort`（含 `chat_template_kwargs` 兼容写法）；非法值 400 |
| API 缺省预算 | 请求不带 `max_tokens` 时默认 8192（修复此前返回空文本） |
| 一键启动器 | `启动.bat`：1 CLI 8K / 2 API 262K / 3 CLI 262K / 4 思考等级 / 5 模型路径 |
| 文档 | 新增 `docs/USAGE.md`（教程）、`docs/PROJECT.md`（项目全貌）；重写 `README.md` |

---

## 3. 实测（RX 9070 XT 16 GB · ROCm 7.2 · 贪心）

| 场景 | prefill (tok/s) | decode (tok/s) |
|---|---|---|
| 8K AR | 948.4 | 35.69 |
| 8K MTP d5 | 848.2 | **105.53** |
| 96K AR | 423.3 | 25.26 |
| 196K + KV 记忆 | 540.4 | 29.77 |
| 250K + KV 记忆 | 534.9 | 30.50 |
| 261K + KV 记忆 | 534.1 | 29.79 |

思考等级（同一句、`--max-new 512`）：none 29 tok / low 102 tok / medium 219 tok / xhigh 60 tok。

思考放开验证（宋词请求）：思考 **2932 token**（超过旧上限 2048），正文完整输出八句，无截断提示。

---

## 4. 二进制

| 文件 | 大小 | sha256 |
|---|---|---|
| `infer-mini.exe` | 191,986,176 B | `100174a529a610594a064cf553e012a4484d3044175a7063d46e2bc20d141023` |
| `infer-serve.exe` | 192,322,048 B | `2edb37ede7c65b3724f9df30d0f5623bedb2c6c279fea0a7624d35bbcdf069fc` |

配套：`启动.bat` / `start.bat` / `model.txt` / `MANIFEST.sha256`。

---

## 5. 升级建议

已经用 0.1.3 的：直接替换两个 exe 即可，配置文件（`effort.txt`、`model.txt`）无需改。
从 0.1.2 及更早升级：请一并看 `RELEASE_NOTES_0.1.3.md`（MTP 自适应、挂死修复）。

---

## 6. 已知限制

- 单流 AR decode 35.7 tok/s：受显存带宽限制（27B 权重每 token 读一遍 11.76 GB）。
- 无批处理 / 无多并发。
- 仅在 RX 9070 XT（gfx1201）+ ROCm 7.2 Windows 验证。
- 视觉为"能读位置与颜色"，复杂场景推理未系统验证。
- 262,144 token 是模型位置编码硬顶。
