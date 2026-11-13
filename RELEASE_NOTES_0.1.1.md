# NInfer-HIP 0.1.1 — 视觉链路修好，OpenAI 兼容服务上线

发布日期：2026-11-12
仓库：https://github.com/Newaiguy/NInfer-HIP
上一版：`v0.1.0-pre-r9`

---

## 一句话

**图像输入从"能读但答错"变成"答对"：九图回归 3/9 → 9/9（三档权重全过），
真实照片 16/16 打平 llama.cpp；同时新增 OpenAI 兼容的 HTTP 服务（真流式 + 视觉）。**

---

## 1. 为什么上一个版本看图会答错

r9 的引擎能接受图片、也能跑完视觉塔，但空间理解系统性错误：单色块位置题几乎一律
答 `top-left`，纯色/半图/上下分色多数错。根因不是一个，是五个，全部在本版修掉。

| # | 缺陷 | 位置 | 症状 |
|---|------|------|------|
| 1 | gfx12 WMMA 的 B 操作数按行而不是按输出列打包 | `src/ops/common/mma.cuh` | packed attention 只剩 1/8 输出通道，其余 7/8 恒为 0 |
| 2 | paired D=1152 LayerNorm 在 `if (threadIdx.x==0)` 内用全 mask 调 `__shfl_sync`（UB） | `src/ops/kernel/layer_norm.cuh` | 专用核均值/计数变垃圾，norm2 全错 |
| 3 | merger 输出被默认乘以 0.1 | `src/models/qwen3_5/execution/vision.cpp` | 大面积内容还读得出，单个色块的位置信号被压没 → 退化成文本先验 |
| 4 | patch embedding 是双帧权重，导出只用了 t0 且复制到两个时间槽 | `_mkvision_bf16.py`（BF16 档） | 全 BF16 档颜色/空间全错 |
| 5 | layer-norm gamma 导出后又被多加 1.0 | 同上 | 全 BF16 档塌成银灰（quad4 四块全银） |

值得强调的是 #1 与 #3：

- **#1 是内核级 bug**，与视觉无关，任何走 packed attention 的路径都会丢 7/8 通道。
  用 identity softmax（Q=K=0、段长 1 时注意力输出必须等于 V）一眼看出：
  9216 个元素错 8064 个，正好 7/8；再用孤立 16×16×16 identity GEMM 分别跑软件 MMA
  与硬件 WMMA，软件 0 错、硬件 120/128 错，把范围锁死到硬件片段映射。
- **#3 解释了此前所有失败的坐标实验**。改 MRoPE 轴序、`POS_ZERO`、`HW_SWAP`、
  注入尺度全都"没用"，是因为信号在进语言模型之前就被压掉了 90%；
  坐标怎么调都救不回来。

## 2. 修好之后

固定 `temperature=0.0`、`seed=42`，每例独立进程，九图回归：

| 权重档 | 结果 |
|--------|------|
| 默认（artifact 内置 q4/q5/q6 量化视觉层） | **9/9** |
| patch-only BF16 覆盖 | **9/9** |
| 全 BF16 视觉覆盖 | **9/9** |

九图覆盖：纯红、四象限（TL红/TR绿/BL蓝/BR黄）、三色竖条、左右分色、上下分色，
以及四个单色块位置题（左上/右上/左下/右上角小块）。

真实照片交叉验证：Wikimedia 原图 7 张、16 个问题（颜色/计数/OCR/形状/物体/地标/
事件/字母数/灯罩数/朝向/轮子数）——

```
本引擎     16/16
llama.cpp  16/16   （BF16 mmproj）
```

两边都是满分，但精度不同源：llama 用 BF16 mmproj，本引擎默认用 artifact 内的
`q4/q5/q6_g64_fp16` 视觉线性层（相对 BF16 权重误差 42%~59%）。也就是说本版是
**用更差的权重打平**；想要高精度档可以设 `NINFER_VISION_WEIGHTS` 指向 BF16 导出目录。

泛化用例（不在九图集内，本版全部正确）：320×320 与 480×360 非正方形、一图两目标
（红左上 + 蓝右下）、一请求两图、48×48 小目标（右上/左下）。

## 3. 新增：OpenAI 兼容服务 `infer-serve.exe`

```
POST /v1/chat/completions     stream=true 时 SSE 真流式（首字节约 0.05 s）
                             content 支持 {"type":"text"} + {"type":"image_url"} 混排
GET  /v1/models
GET  /health                 存活 + 视觉是否可用
```

图像解码用 Windows 自带的 WIC，支持 PNG / JPEG / GIF / BMP / TIFF，带 alpha 时
合成白底，无外部依赖（不再需要 FFmpeg / zlib）。启动：

```bat
set PATH=C:\Program Files\AMD\ROCm\7.2\bin;%PATH%
infer-serve.exe --model E:\infer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer ^
                --ctx 32768 --kv rk4v4-e8 --spec mtp --draft-tokens 2 --port 8099
```

调用示例：

```bash
curl http://127.0.0.1:8099/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "ninfer", "stream": true, "max_tokens": 64,
  "messages": [{"role":"user","content":[
    {"type":"text","text":"图片里左上角是什么颜色？"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,...."}}]}]}'
```

注意：`image_url` 这一项**必须带 `"type":"image_url"`**，否则服务端会把它当纯文本，
视觉通路不会启动。

构建：`harness/build_serve.py`（不依赖 CMake，直接编 `serve_mini.obj` 并与
`ninfer_hip.lib` 链接）。源码在 `harness/serve_mini_main.cpp`。

## 4. 性能没有回退

GATE（graph on、`--kv rk4v4-e8`、`--ctx` 贴紧 prompt）：

| 档 | decode | prefill | 相对基线 |
|----|--------|---------|----------|
| 8k  | 36.50 t/s | 985.9 | +8.5% / +5.2% |
| 96k | 25.30 t/s | 428.4 | +4.2% / +1.2% |
| mtp | 82.11 t/s | 891.3 | +5.4% / +3.6% |

长上下文维持 r9 水平：192K（196,427 token、窗口 32768）prefill 535.7 / decode 33.8；
128K（130,700 token）544.3 / 34.4。96k 档的提升来自 GDN chunked 路径的算子修复。

## 5. 另外修的一个构建缺陷

`src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh` 里
`mma16_swiz_x_raw_d`（dispatcher）引用了在它之后才定义的 `mma16_swiz_x_raw`，
没有前向声明。之前一直靠陈旧目标文件蒙过去；**干净 CMake 构建会直接编译失败**
（`use of undeclared identifier 'mma16_swiz_x_raw'`）。本版加了前向声明。

顺带说明：`full_rebuild.py` 会无差别扫 `src/**/*.cu`，因此会撞上两个 CUDA-only 的
NVFP4 TMA TU（`CUtensorMap` / `CUresult` 未定义）——它们属于独立 CMake target
`ninfer_nvfp4_non_rdc`，ROCm 构建本就不编，属预期报错，不是发布阻塞。

## 6. 升级建议

- 只是文本推理：换成新的 `infer-mini.exe` 即可，参数不变。
- 要图像问答：用 `infer-serve.exe`，按 §3 起服务。
- 想量化视觉精度损失：设 `NINFER_VISION_WEIGHTS` 指向 `vision-bf16` 导出目录
  （BF16 档本版已修好），与默认档对比。

## 7. 已知限制（未变）

- 126K 以上 MTP 不可用（差约 0.5 GB 预算），需关 MTP。
- int8 KV 非确定性；fp8 KV 的 QK 仍是 `__builtin_trap()` 桩；dflash2 坏路径未修。
- prefill ≥64K 仍落后 llama.cpp；256K 需要 KV offload。
- KV 记忆的召回是启发式：粗排峰值可能比文档早约 8 页，由「以峰值为中心的整段 112 页」吸收；
  精排（归档 int4 K 重排 top-256）默认开；必须读完 prompt 才能召回。
  > **0.1.2 更正**：0.1.1 的发布二进制把单段缩成「预算 / 2 或 4」的窄块，单文档 needle
  > 会截断答案文档（召回页 732..787，输出 45 10052 248046；正确是 704..815）。0.1.2 已把
  > 单整段恢复为默认实现，并给出 25%/50%/75% 深度与完整上下文的逐 token 对拍。

## 8. 校验和

```
infer-mini.exe    191,703,040 B
  sha256 e89fc926bd3026ac38f5c3a92a5c3dd632bbeb4a9d17b89fcda1ed90f47c9924
infer-serve.exe   192,038,912 B
  sha256 0b6f04c667b0fbd4621404cfcd046d268ba22f9ae5fd3ceae85a268856eb283b
```

完整文件校验见 `MANIFEST.sha256`。
