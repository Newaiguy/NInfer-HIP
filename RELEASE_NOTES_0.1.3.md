# NInfer-HIP 0.1.3

本版是一个功能完整、性能定型的版本：视觉链路修好、KV 记忆稳定、MTP 深度自适应。

## 结果

| 项目 | 数值 |
|---|---|
| 九图确定性回归 | 三档权重各 **9/9**（上一版 3/9） |
| 真实照片 16 问（对照 llama.cpp） | **16/16**，对手 16/16 |
| 8K decode（AR / MTP） | 35.69 / **105.53** t/s |
| 96K decode / prefill | 25.26 / 423.3 t/s |
| 196K decode / prefill | 29.77 / 540.4 t/s |
| 250K decode / prefill | 30.50 / 534.9 t/s |
| 261K decode / prefill | 29.79 / 534.1 t/s |
| 长上下文准确率（对照 llama-kvmem，8 例） | **8/8**，对手 8/8 |
| 门禁 8k / 96k / mtp | 全部 PASS |

## 变更

1. **视觉**：修好五处真实缺陷 —— gfx12 WMMA 的 B 操作数打包（packed attention 丢 7/8 通道）、
   paired D=1152 LayerNorm 的 shuffle 未定义行为、merger 输出被默认乘 0.1 导致位置信号被压没、
   patch 双帧权重导出缺一半、视觉 norm gamma 多加 1。三档权重（量化 / patch-only BF16 /
   全 BF16）九图全部 9/9。
2. **MTP 深度自适应**：`--draft-tokens 0`（默认）按上下文自动取 5（≤65536）或 4（>65536）。
   8K 档 decode 从 79.5 提到 105.5 t/s。上一版在 `ctx > 65536` 加的 d5/d6 拒绝已撤掉。
3. **检索加固**：每轮读回的草稿 extent 做 `0 <= next <= 15` 校验，越界直接报错，不再静默跑飞。
4. **服务端**：请求体/`messages` 形状错误、超长 prompt 统一返回 HTTP 400（`invalid_request_error`），
   不再 500。

## 兼容性

- 模型工件、CLI 参数、环境变量与本仓库 0.1.2 完全兼容。
- 二进制与本目录 `VERSION.txt` 中的 sha256 一致。

## 校验和

见 `MANIFEST.sha256`。