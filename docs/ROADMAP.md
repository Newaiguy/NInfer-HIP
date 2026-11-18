# 路线图

## 下一步：MTP 深度按请求自适应

现状：`startup.cpp` 的自动档按 `--ctx` 选深度（≤65536 → 5，>65536 → 1）。

问题：`--ctx` 是部署能力，不是这次请求的实际长度。一个 262K 部署里的 8K 短请求
会被降到深度 1（约 50 t/s），而深度 5 在 8K 上实测 62 t/s。

目标：按**本次请求的实际 prompt token 数**选深度，阈值同样取 64K ——
prompt ≤ 65536 用深度 5，> 65536 用深度 1（多轮对话每轮重选）。

需要改的三处（装载期就要为多个深度备好）：

1. `ninfer/src/models/qwen3_5/program/planning/round_buffers.cpp`
   —— 按最大深度分配缓冲（`drafts = max(1, draft_window)`、`ar_steps = draft_window - 1`）。
2. `ninfer/src/models/qwen3_5/program/planning/graph_profiles.cpp`
   —— `mtp_graph_profiles(capacity, draft_window)` 目前只给一个宽度建图；改成 {1, 5} 各建一套。
3. verify 内核的 `T = K+1` 特化 —— 需要同时实例化 T=2 与 T=6 两条路径。

保留 `--draft-tokens N` 作为手动覆盖。

预估：一天左右，主要成本是两套图的显存与装载时间回归。
