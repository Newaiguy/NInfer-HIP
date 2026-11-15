# NInfer-HIP 0.1.3

**一句话**：MTP 草稿深度改成自适应（`--draft-tokens 0`），并且修掉了一个"看起来像
产品崩溃、实际是构建污染"的挂死——上一版在 `ctx > 65536` 加的那条 d5/d6 拒绝已经撤掉。

## 1. 自适应 MTP 深度

| 上下文 | 自动 draft | 说明 |
|---|---|---|
| `<= 65536` | 5 | 单任务实测峰值在 5–6（8k: 105 t/s） |
| `> 65536` | 4 | 长上下文里草稿收益衰减，4 已验证稳定 |

`--draft-tokens 0` 与不传等价；想钉死就照旧显式给 1–15。

实测：ctx 8192 `decode 66.07`、ctx 100000 `decode 71.01`。

## 2. 那个挂死

现象：`ctx > 65536` + `--draft-tokens 5/6` 时进程静默卡住，GPU 利用率 0%，CPU 也几乎不动。

根因链：
1. `[res] drafts-copy next=-1112430859 row=0` —— host 侧读到垃圾 extent；
2. `static_cast<std::uint32_t>` 之后变成约 3.2e9，`for (step < count)` 拷贝循环几乎无限；
3. 一串排除（SIMT/WMMA 路线、attention 本体、CUDA Graph、KV 召回/重放、显存宽度）
   全部无效，而设备侧 dump 是 `next_extents = 4`；
4. 设备值正确、host 值错误 ⇒ `MtpDecodeEgress` 的 sizeof/offset 在两个翻译单元之间不一致；
5. 全量一致重建后读出 4，六个组合（65537/81920/100000 × draft 5/6）全部通过。

**根因是我自己的增量编译**（反复只重编改动的 TU，头文件改动后其余目标文件过期），
不是产品逻辑缺陷。已保留一条真实加固：读回 extent 后校验 `0 <= next <= 15`，越界抛
`MTP round reported an out-of-range draft extent`。

## 3. 门禁与回归

```
8k   decode 35.69  prefill 948.4   (+6.1% / +1.2%)
96k  decode 25.26  prefill 423.3   (+0.5% / -0.4%)
mtp  decode 105.53 prefill 848.2   (+0.1% / +1.8%)
GATE: PASS
```

视觉三档九图（artifact / patch-only BF16 / 全 BF16）各 **9/9**；本版未改动视觉路径。

## 4. 校验和

| 文件 | 大小 | sha256 |
|---|---|---|
| `infer-mini.exe` | 191,981,568 | `f545e1402be4deb2ac8eeb400016233adb32e3c89f564d79dd66c04d5a2f4fb9` |
| `infer-serve.exe` | 192,317,952 | `ce75a12ca1c6b5c0a3a8014c8c40f20d8b5bf4a5613331731d96c9eda90fceee` |

## 5. 已知限制

- KV 召回仍是"单整段 + 精排兜底"的启发式，多文档场景建议 `NINFER_KV_RECALL_DIV=2`；
- 单流 decode 受限于每 token 必须整读 11.76 GB 权重，理论 100% 带宽约 45 t/s；
  下一版主线是把 GDN 的约 7.7 ms 纯延迟压下去（ssmout 9 GB/s、recurrent 5 GB/s）。
