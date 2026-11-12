# 视觉（多模态）现状

本文件记录视觉通路的实现、实测能力与已知限制。判断标准是"和 llama.cpp 参考实现
（同一 GGUF 文本权重 + 同一份 mmproj）在同图同 prompt 下的对照"。

## 1. 怎么用

```
infer_serve_nordc.exe <模型.ninfer> --ctx 8192 --kv rk4v4-e8 --port 8099
```

请求体走 OpenAI 兼容格式，`messages[].content` 里放 `image_url`，data URL 或 http URL 都支持：

```json
{"model":"ninfer-qwen3","max_tokens":80,
 "messages":[{"role":"user","content":[
   {"type":"text","text":"Which corner is the red square in? Answer with one of: top-left, top-right, bottom-left, bottom-right."},
   {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]}]}
```

可用的环境变量：

| 变量 | 作用 |
|---|---|
| `NINFER_VISION_WEIGHTS=<dir>` | 用 mmproj 导出的 bf16 权重覆盖视觉塔线性层（见 §4） |
| `NINFER_VISION_SCALE=<f>` | merger 输出缩放，**默认不缩放（等价 1.0）**；只留作实验旋钮。历史上的默认 `0.1` 是空间题全错的元凶，见 §3 |
| `NINFER_VISION_TSTEP=grid` | 回到"图像后文本游标跳整格"的旧行为 |
| `NINFER_VISION_HW_SWAP=0` | 关闭 h/w 轴对调 |
| `NINFER_VISION_NAN=1` / `NINFER_VISION_TRACE=1` / `NINFER_VISION_DUMP=<dir>` | 诊断 |

## 2. 已核实正确的部分

- **图像进入 LLM**：图像被切成 28×28 patch、合成 14×14=196 个视觉 token，正确插入 `<|vision_start|>…<|image_pad|>×196…<|vision_end|>` 之间。
- **预处理**：解码 → 等比缩放 → `(v/255-0.5)/0.5` → patch 展开，与参考实现逐值一致
  （引擎 `patches_in` 与 numpy 重建 **corr = 1.0000**）。
- **位置编码**：对真实 28×28 patch 网格使用 processor 的 block/merge 顺序重建坐标，Vision2D RoPE 是 split-half：旋转 `pair` 与 `pair+36`，`pair/18` 选择 y/x 轴。不要把旧 `_l0attn.py` 的相邻维旋转实现当作正确 oracle。
- **Layer-0 数值对拍（部分为 LayerNorm 修复前）**：paired D=1152 LayerNorm 修复后的真实 full-BF16 red dump，以 GPU `l0_x_attn` 同输入及 loader gamma `raw+1` 高位截断重算 `l0_norm2`，rel-L2 **1.52e-5**、corr **1.0**、max **0.015625**；正式默认 paired `test_layer_norm.cpp` 也通过。早期同 QKV dump 的 784-token CPU attention 为 corr **0.99999976**、rel **0.00068774**；旧版激活路径的 output projection / residual / FC1 / FC2 数值为 rel **0.00124 / exact / 0.0000225 / 0.01772**，这些后续阶段数字需在新 LayerNorm activation 下重测，不能视为当前视觉塔完全验证。
- **packed attention 的 WMMA B-fragment 错位已修复**：gfx12 WMMA 的 B lane 表示输出列，不是 K 行；旧 `mma_hw_pack_ab` 将 B 操作数按 K 维重排，导致视觉塔 attention 输出通道丢失。修正后，单 token segment 的 `attention(Q,K,V)=V` 在 tile 16/32/64 上均 `0/9216` mismatch；非零 Q/K/V、8 tokens、两个独立 4-token segment 对 CPU softmax oracle 的 `rel_l2=0.00185796`、`max_error=0.00515914`。扩展到 64 tokens、两个 32-token segment 后，tile 16/32/64 的 identity 均 `0/73728` mismatch；随机 Q/K/V oracle `rel_l2≈0.00185`、`max_error≈0.01981`（BF16 输出舍入）。进一步匹配实际图像长度 T=784：单 token identity 在三个 tile 都是 `0/903168` mismatch；非零 Q/K/V、完整 784-token segment、tile 64 CPU oracle `rel_l2=0.00174749`、`max_error=0.00426793`、`>0.01` 误差元素 `0/903168`。新增的正式 `plain_and_packed` operator regression PASS。
- **跨请求隔离**：视觉工作区在每次 encode 前清零（`cudaMemsetAsync`），同一进程连发多图结果稳定。
- **patch embedding 导出合同**：BF16 patch-only 覆盖后的四象限颜色题在修复 WMMA 和 LayerNorm 后通过；三色条、半图和小块位置仍会错，详见 §3 的当前九图表。

## 3. 已知限制（当前状态）

输入图真值：`img_quad4.png` 为 TL 红、TR 绿、BL 蓝、BR 黄；`bars` 为红/绿/蓝；`img_half.png` 左红右蓝；`img_tb.png` 上绿下黄。`build\_vprobe_cfg.py` 对每个请求显式设置 `temperature=0.0`、`seed=42`、`max_tokens=60`，并输出完整回答不截断。

**当前状态（2026-11-12，WMMA + paired LayerNorm + patch 双帧 + 注入缩放四处修复后）：默认配置 9/9 全对，且重复进程逐字一致。**

| 用例 | 参考答案 | 引擎默认（artifact 权重） | 仅 BF16 patch 覆盖 |
|---|---|---|---|
| `red` | Red | Red | Red |
| `big_tl` | top-left | top-left | top-left |
| `big_tr` | top-right | top-right | top-right |
| `big_bl` | bottom-left | bottom-left | bottom-left |
| `corner_tr` | top-right | top-right | top-right |
| `quad4` | TL 红 / TR 绿 / BL 蓝 / BR 黄 | 全对 | 全对 |
| `bars` | red, green, blue | 全对 | 全对 |
| `half` | left=red, right=blue | 全对 | 全对 |
| `tb` | 上绿下黄 | 全对 | 全对 |

### 根因（最后一处）：merger 输出被 0.1 缩放

merger 训练目标就是把视觉特征投影到**文本 embedding 同一空间**，所以拼接时必须原幅注入。旧代码默认 `NINFER_VISION_SCALE=0.1`，等于把图像信号压到十分之一：大面积内容（纯红、半图、上下分色、三色条、四象限）还能被读到，**单个色块的位置信息却弱到读不出来**，模型于是退化成文本先验——四种位置题全部答 `top-left`。这正是改了坐标轴序、`POS_ZERO`、注入尺度历史实验都无效的原因：它们没触及"信号被压小 10 倍"这一点。

修复：`vision.cpp` 里删除默认 0.1 分支，只在显式设置 `NINFER_VISION_SCALE` 时才缩放。实测 `SCALE=0.3` 已能让 5 个位置题全对但 `tb` 输出发糊（"GREEN-OLIVE"），`0.3`/`1.0` 均优于 `0.1`，默认取不缩放。

已知限制：目前三档（artifact 量化 / patch-only / 全 BF16）都通过同一套九图与 16 道真实照片题，尚未找到能区分精度的硬用例；若要做量化损失评估，需要更吃塔精度的任务（密集小字 OCR、大数量计数、图表读数）。

### 真实照片交叉验证（对照 llama.cpp，2026-11-12）

九图是合成的几何图，只能证明"位置/颜色信号没被压掉"。为排除过拟合到合成图，另做了一组**真实照片**对照：`build\_fetch_web_images.py` 从 Wikimedia Commons 取 7 张照片（红色双层巴士、香蕉叶片、STOP 标志、金毛犬、红绿灯、闪电中的埃菲尔铁塔、蓝色自行车），我逐张看图确定真值，再对两台引擎跑同一批 12 个问题（颜色、计数、OCR、形状、物体识别、地标、事件识别）。

结果：**本引擎 12/12，llama.cpp 12/12**，逐条答案：

| 问题 | 本引擎 | llama.cpp |
|---|---|---|
| 巴士是什么颜色 | red | red |
| 图上有几个白圈 | 2 | 2 |
| 标志上的字 | STOP | STOP |
| 图里是什么动物 | Dog | Dog |
| 红绿灯亮哪个灯 | red | red |
| 著名地标名称 | Eiffel Tower | Eiffel Tower |
| 自行车颜色 | blue | blue |
| 动物毛色 | golden | Golden |
| 标志形状 | Octagon | Octagon |
| 巴士有几层客舱 | 2 | 2 |
| 红绿灯亮几个灯 | 1 | 1 |
| 塔顶上方天空发生什么 | Lightning is striking the top of the tower | Lightning is striking the top of the tower |

复现方式（两边都用固定 `temperature=0`；本引擎侧 `_vprobe_cfg.py` 还强制 `seed=42`、`max_tokens=60`）：

```
python build\_fetch_web_images.py                       # 取图，写 build\webimg\manifest.json
python build\_mk_web_requests.py                        # 生成 build\_req_web_*.json 与 webimg\cases.json
python build\_vprobe_cfg.py web_00_bus web_01_banana ... # 本引擎（每个用例新进程）
python build\_llama_web_bench.py                        # llama.cpp 对照（llama-mtmd-cli）
```

llama.cpp 侧用 `D:\Qwen3-27B-Deploy-migrate\llama.cpp\build-hip72\bin\llama-mtmd-cli.exe` + `S:\NLP自然语言\Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` + `S:\NLP自然语言\mmproj-Qwen3.8-27B-BF16.gguf`，运行时必须把 `C:\Program Files\AMD\ROCm\7.2\bin` 放进 `PATH`，否则进程以 `0xC0000135` 直接退出。图片下载需走 `clash-proxy`（Wikimedia 直连被拦），下载后必须 `proxy.ps1 -Action stop`。

## paired D=1152 LayerNorm 的 gfx12 UB 修复

此前同输入 norm2 对拍出现约 0.067 rel-L2。正式 `test_layer_norm.cpp` 进一步表明默认 paired D=1152 kernel 四个 aligned cases 均输出非有限值，而 force-generic 路径通过。根因是 `layer_norm_d1152_warp_kernel` 的 lane-0 trace 区域执行 `__shfl_sync(0xffffffffu, local.count, 0)`，但只有 `threadIdx.x==0` 进入该分支，full-warp mask 违反 shuffle 同步合同。将其改为直接读取已经 warp-reduced 的 lane-0 `local.count` 后，正式 LayerNorm CPU-oracle 测试在默认 paired dispatch 下输出 `OK layer_norm correctness`。再用 full-BF16 red 的真实 GPU `l0_x_attn` 作为同输入，按 loader gamma `raw+1` 高位截断重算 `l0_norm2`，rel-L2=`1.521055e-5`、corr=`1.0`、max=`0.015625`；先前的 norm2 差异已消失。mini 已据修复对象重新链接并复跑九图，但整套视觉任务仍未通过。

此前 WMMA 修复前的阶段 dump/九图表是历史记录，不可用于判断当前 attention 行为。

Upstream host CPU Vision encoder 已能 standalone 跑完整 27 层。paired D=1152 LayerNorm 修复后，full-BF16 red 的 same-input norm2 GPU/CPU 复算 rel-L2=`1.52e-5`；整塔 CPU BF16 vs GPU `merged_out` red rel-L2=`0.02266`/corr=`0.999745`，但 fresh `big_bl` 为 rel-L2=`0.15419`/corr=`0.98909`。源审计确认 CPU 中间态全 FP32、GPU workspace/算子输出 BF16；两边 patch/control 顺序、坐标插值、merger gather、exact GELU 和输出布局一致，比较时 `NINFER_VISION_POST_LN` 未设置。故 red/big_bl 差异可能来自 BF16-vs-FP32 累计，暂不能据此认定 GPU bug。下一步做精度匹配的 big_bl 分层对拍；若与 CPU RNE 仿真吻合，剩余失败应转向文本模型对视觉特征的读出，否则定位首个分叉算子。详见 [`VISION_STATE.md`](D:/ninfer-hip-migrate/docs/VISION_STATE.md)。

## 4. bf16 权重档（可选，实验性）

视觉线性层在 artifact 里是 4~6 bit 量化（`q4/q5/q6_g64_fp16`），相对误差 42%~59%。
mmproj 里有同一套权重的 bf16 版本，`build/_mkvision_bf16.py` 可导出成交错裸文件：

```
python build\_mkvision_bf16.py        # 产出 build/vision-bf16/*.bf16 + manifest.tsv
```

设 `NINFER_VISION_WEIGHTS=D:\ninfer-hip-migrate\build\vision-bf16`，保持默认注入缩放（不缩放）即可启用。不要再额外设 `NINFER_VISION_SCALE=0.1`——那会把特征压小十倍。

**patch embedding 的两个合同**（2026-11-12 用 full-rank 噪声判定，别再踩）：

1. mmproj 有**两张** patch 权重 `v.patch_embd.weight`(t0) 与 `v.patch_embd.weight.1`(t1)；静止图也走两帧，导出不能只取前者。
2. 排布是**按通道交错的双帧** `H = [c0t0, c0t1, c1t0, c1t1, c2t0, c2t1]`；引擎 patch loader 会再转置一次，写入文件要用 `H.reshape(1536,1152).T`。

判定方法（可复用）：用 448px 随机噪声图（纯色 fixture 是低秩的，`red` 的 `patches_in` rank=1，无法用来反解布局）+ `NINFER_VISION_DUMP` 抓 `patches_in.f32`/`embed_proj.f32`，对候选矩阵算 `P @ M.T + bias` 与实测投影的相关。artifact 正确候选 corr `0.999947`/rel `0.01031`，修正后 override corr `0.999998`/rel `0.00196`。

当前效果（WMMA + LayerNorm + 双帧 patch + 去缩放后）：artifact、patch-only、**全 BF16 三档均 9/9**；全 BF16 档在真实照片上同样 4/4（`web_12`/`web_14`/`web_15`/`web_02`）。

**根因：norm gamma 被多加了一次 1.0**（2026-11-12 二分定位）。覆盖加载器原先假设 GGUF 的 norm scale 是 delta，读取时统一 `value += 1.0F`；而 mmproj 里存的就是最终 gamma（例：layer0 `norm1_weight` mean `0.4663`、layer10 mean `0.9652`、`post_ln` mean `1.3063`，还有负值），再 +1 把每层激活整体放大，塔输出退化成一片灰 —— 表现为 `red -> Silver`、四象限全银。修复是**逐位拷贝、不加偏移**（`ninfer/src/models/qwen3_5/execution/vision.cpp` 的 post_ln 与 layer-norm 两个分支）。二分证据（manifest 子集，`red`/`quad4`）：

| 子集 | 张量数 | red | quad4 |
|---|---|---|---|
| patch | 1 | Red | 四色正确 |
| patch+merger+post_ln | 5 | Red | 四色正确 |
| patch+attention | 109 | Red | 四色正确 |
| patch+mlp | 109 | Red | 四色正确 |
| patch+norm **bias** | 55 | Red | 四色正确 |
| patch+norm **weight** | 55 | **Silver** | **四块全银** |

可复用工具：`build/_vision_manifest_subset.py`、`build/_vision_bisect_manifests.py`（生成 `A_patch`…`G_all` 子集 manifest）、`build/_vision_bisect_run.py`（批量跑子集）、`build/_vision_norm_shift.py`（把导出文件预移 ±1 做 A/B）。

注意：
- **路径必须用 `D:\ninfer-hip-migrate`（带 n）**，用别名路径时进程内 `fopen` 会静默失败，
  manifest 读成 0 行、覆盖悄悄不生效。
- `vision/patch_embedding` 现在由 `build/_mkvision_bf16.py` 自动写入 manifest：单帧 `[1152,768]` 按通道平面复制为引擎所需的双时间帧 `[1152,1536]`，manifest 行为 `vision/patch_embedding vision__patch_embedding.bf16 1769472`。
- 仅 patch 覆盖即可让四象限题通过；全 BF16 覆盖档现已与 artifact 档同水平（9/9），可用于"量化 vs bf16"的定量对照。

## 5. 构建与回归

```
python build\_compile_cu.py <rel.cu>                 # .cu
python probe_rebuilt\compile_one.py <rel.cpp>       # .cpp（过滤 ^rc=|error C）
python build\_mklib3.py                              # 打包（LINK-ONLY，改头文件要 full_rebuild.py）
$env:NINFER_NORDC="1"; python build\_link_serve.py serve_mini.obj build\infer_serve_nordc.exe
python build\_mkvision_bf16.py                       # 329 tensors, including patch_embedding
python build\_mkvision_patchonly.py                   # isolated one-tensor override for recommended comparison
python build\_vprobe_cfg.py NINFER_VISION_WEIGHTS=D:\ninfer-hip-migrate\build\vision-bf16-patchonly red big_tl big_tr big_bl corner_tr quad4 bars half tb
python build\_gate.py --arms 8k,96k,mtp              # 文本回归门禁
python probe_rebuilt\full_rebuild.py                 # source/header changes require full rebuild
```

改过 `kv_store.h` 等头文件时必须 `python probe_rebuilt\full_rebuild.py`
（否则增量 lib 会混旧目标文件，出现"假绿/假红"）。
