# 视觉塔状态（2026-11-10 夜）

本文件记录"为什么多模态定位不准"这一轮排查得到的**可复用结论**，供后续接手的人从正确的位置继续。

> **结论订正（2026-11-12 21:05，全 BF16 档已修好：三档全部 9/9）**
>
> 覆盖加载器原先假设 GGUF 的 vision norm scale 是 delta，读取时统一 `value += 1.0F`（layer norm 与 post_ln 两个分支都有）。实际 mmproj 存的就是**最终 gamma**（layer0 `norm1_weight` mean `0.4663`、layer10 mean `0.9652`、`post_ln` mean `1.3063`，且有负值），再 +1 相当于把每层激活整体放大，塔输出退化成一片灰 —— 现象是 `red -> Silver`、四象限全银。
>
> 修复：`ninfer/src/models/qwen3_5/execution/vision.cpp` 的 post_ln 与 layer-norm 分支改为**逐位拷贝**，不加偏移。
>
> 二分证据（manifest 子集，`red`/`quad4`，固定 temperature=0/seed=42）：patch(1) → Red 四色对；patch+merger+post_ln(5) → 对；patch+attention(109) → 对；patch+mlp(109) → 对；patch+norm **bias**(55) → 对；patch+norm **weight**(55) → **Silver / 四块全银**。即唯一破坏项是 layer-norm 的 gamma。
>
> 修复后（mini `192,051,200` bytes，无其它环境变量，`temperature=0.0`/`seed=42`/`max_tokens=60`）：artifact、patch-only、**全 BF16 三档九图各自 9/9**（同一最终二进制复跑）；全 BF16 档真实照片 4/4（标志字母 4、狗面向镜头、自行车 2 轮、STOP 标志）。顺手删掉了 `NINFER_VISION_NORM_PLUS_ONE` 这个会二次加 1 的实验分支，避免再被误开。现在可以做"量化 vs bf16"对照；尚未找到能区分两者精度的硬用例。
>
> 可复用工具：`build/_vision_bisect_manifests.py`（生成 A_patch…G_all 子集 manifest）、`build/_vision_bisect_run.py`（批量跑子集）、`build/_vision_norm_shift.py`（导出文件预移 ±1 做 A/B）、`build/_vision_norm_stats.py`（打印导出 norm 统计）。

> **真实照片交叉验证（2026-11-12 19:50，对照 llama.cpp）**
>
> 九图是合成几何图，可能过拟合。另从 Wikimedia Commons 取 7 张真实照片（红色双层巴士、香蕉叶片、STOP 标志、金毛犬、红绿灯、闪电中的埃菲尔铁塔、蓝色自行车），逐张看图定真值后，两台引擎跑同一批 12 个问题：**本引擎 12/12、llama.cpp 12/12**（颜色/计数/OCR/形状/物体识别/地标/事件识别全部一致）。复现脚本：`build\_fetch_web_images.py`（取图）→ `build\_mk_web_requests.py`（生成请求）→ `build\_vprobe_cfg.py web_00_bus ...`（本引擎）→ `build\_llama_web_bench.py`（llama.cpp 对照）。llama.cpp 侧须把 `C:\Program Files\AMD\ROCm\7.2\bin` 加进 `PATH`，否则 `0xC0000135` 退出；图片下载走 `clash-proxy`，下载完必须 `-Action stop`。
>
> 注意 `build\_vprobe_cfg.py` 是唯一验证过稳定的启动方式；自建 harness 直连 `infer_serve_nordc.exe` 时曾多次遇到 `vision unavailable (... open: No error)` 并退化为纯文本，最终健康检查失败——复跑对照实验请复用 `_vprobe_cfg.py`，不要另写启动器。

> **结论订正（2026-11-12 18:30，最高优先级：视觉定位问题已解决）**
>
> 最后一处根因是 **merger 输出被 `NINFER_VISION_SCALE=0.1` 缩放**。merger 的职责就是把视觉特征投到文本 embedding 空间，拼接时必须原幅；压到十分之一后大面积内容（纯红/半图/上下分色/三色条/四象限）仍能被读到，但**单个色块的位置信号弱到读不出来**，模型退化成文本先验，所以 `big_tr`/`big_bl`/`corner_tr` 一律答 `top-left`。
>
> 修复：`ninfer/src/models/qwen3_5/execution/vision.cpp` 删除默认 0.1 分支，只有显式设置 `NINFER_VISION_SCALE` 才缩放。
>
> 实测（`temperature=0.0`、`seed=42`、`max_tokens=60`，mini 重新链接）：artifact 档 **9/9**（`red`=Red、四个单红块位置全对、`quad4` 四色全对、`bars` 红绿蓝、`half` 左红右蓝、`tb` 上绿下黄），patch-only 档同样 **9/9**；两次独立进程逐字一致。`SCALE=0.3` 也能让五个位置题全对但 `tb` 输出发糊，`1.0` 干净，故默认取不缩放。
>
> 因此下方 2026-11-11 的 “scale sweep 显示 1.0 更差、保留默认 0.1” 结论**已作废**：那是 paired LayerNorm 与 patch 双帧修复之前、且用受限 left/right 问句测的。全 BF16 线性层覆盖档已于 2026-11-12 21:05 修好（见顶部条目），三档现在都是 9/9。
>
> - MRoPE 分段映射没有未提交改动：保持 `config.cpp` 的 interleaved `pair % 3`、`text.cpp` 的 `[T,3]` axis-major positions 和 `rope.cuh` 的 `positions[axis*tokens+token]`，不做 contiguous 改动。
> - 位置表的临时 `NINFER_VISION_POS_REPACK` 实验已完成并删除。直接复制是正确合同：控制元数据和专用 kernel 都按 `patch * 4 + corner` 读取；重排会改变结果但不能修复四象限。
> - 已修复 BF16 导出器：`build/_mkvision_bf16.py` 现在将 `v.patch_embd.weight` 从单帧 `[1152,768]` 按通道平面复制为引擎需要的双时间帧 `[1152,1536]`，并写入 manifest 的 `vision/patch_embedding`（当前共 329 个 tensor）。
> - 已确认并修复 packed attention 的 gfx12 WMMA B-fragment 映射 bug：硬件 B lane 表示输出列（`L<16` 是列 `L`、K[0:8]；`L>=16` 是列 `L-16`、K[8:16]），旧 `mma_hw_pack_ab` 错把 lane 按 K 行重排。C seed/read 映射保持原样；对 dummy 列 gather 使用 `src=4*(col&7)+i`，避免 wave32 越界 shuffle。修复在 `ninfer/src/ops/common/mma.cuh`。
> - 修复前 identity attention 在每 head 的 72 个通道里只保留每 8 个中的 1 个，其余 7/8 为 0；修复后生产 packed attention identity 测试 tile 16/32/64 均为 `0/9216` mismatch。非零 Q/K/V、8 tokens、两个独立 4-token segments 对 CPU softmax oracle：rel_l2 `0.00185796`、max_error `0.00515914`、大于 0.01 的元素 `0/9216`。
> - 扩展 probe 到 64-token interleaved QKV，identity segment length=1 在 tile 16/32/64 均 `0/73728` mismatch；非零 Q/K/V、两个独立 32-token segment 的 CPU oracle rel_l2≈`0.00185`、max_error≈`0.01981`。随后扩展到实际 784-patch 视觉段：identity length=1 在 tile 16/32/64 均 `0/903168` mismatch；非零 Q/K/V、完整 784-token segment、tile 64 CPU oracle rel_l2=`0.00174749`、max_error=`0.00426793`、`>0.01` 误差 `0/903168`。长序列结果来自诊断脚本，不是正式测试 target。
> - 正式测试 `ninfer/tests/ops/softmax_attention/plain_and_packed.cpp` 已增加 interleaved-QKV 的 length-1 segment case；窄测试目标编译/链接成功，输出 `packed_softmax_attention: PASS`。`launch.cu` 与清理后的 `vision.cpp` 已针对性重新编译，response-file archive 465 成员，mini probe 重链接成功（192,053,248 bytes）。全量构建仍有 3 个无关预存失败（NVFP4 TMA 两项、GDN chunked 一项），其余 441/444 TU 成功。
> - 修复 WMMA 后、paired LayerNorm 修复前的九图（`temperature=0.0`, `seed=42`）不再支持旧的“量化 tower 抹平所有图像信息”结论。artifact 档 `quad4` 正确、`red` 答 `Red`，但 `bars` 顺序为绿/蓝/红，`half` 为白/蓝，所有单红块位置问句都偏答左上。patch-only 档 `quad4`、`bars` 正确，但纯红答 `White`、`half` 白/白、右侧/左下小块仍常答左上。full-BF16 档 `quad4`、`bars` 正确，但纯红/半图/右侧小块仍失败。以上已由 2026-11-12 paired LayerNorm 修复后的新九图取代；九图门禁仍未 9/9。
> - 旧 `NINFER_VISION_SEG1` 输出、旧阶段 dump（attention WMMA 修复前）只作为历史证据；不能继续据此说 attention 与根因无关或声称现有塔差异仅由量化造成。
> - exporter 的 patch embedding 修复仍有效：`build/_mkvision_bf16.py` 将单帧 `[1152,768]` 按通道平面复制成 `[1152,1536]`，manifest 共 329 tensors。数值 GEMM 控制：按 active loader transpose 解释 patch raw BF16，engine projection corr `0.99999785`、rel `0.00185648`；full-BF16 layer-0 norm/QKV/FC1 数值合同也已验证（前序条目），不应盲改矩阵转置。
> - 2026-11-11 21:24 更新：实际 patch-only 红图 dump 的 QKV 使用 active split-half Vision2D RoPE（`pair` 与 `pair+36`，`axis=pair/18`）重算 784-token dense attention，engine `l0_attended` 比较得 rel-L2 `0.0006877382`、corr `0.99999976`、max error `0.0078125`、`>0.02` error 为 `0/903168`。临时 `l0_attended` dump 已移除并重编/重链接 clean mini。
> - 同日 full-BF16 红图层 0 后续数值控制：attention output GEMM（先 BF16 output、再 bias）rel-L2 `0.0012399`、corr `0.99999923`；attention residual 与 `l0_x_attn` exact；FC1 kernel-round 对 `l0_up` rel-L2 `2.25e-5`、corr `1.0`。FC2 direct raw `[N,K]` interpretation 对 `l0_down` rel-L2 `0.01772`、corr `0.999845`；转置解释 rel-L2 `0.6231`，故目前没有证据修改 FC2 layout。下一步应查更深层/整塔及 LM readout，而不是继续改 layer-0 attention/位置复制。
> - 2026-11-11 20:05 后续定位：直接读取 PNG 像素确认 `img_big_tl` 红块 bbox 为 x/y `0..223`，`img_big_tr` 为 x `224..447`、y `0..223`，不是 fixture/请求标签写反。patch-only、修复后 dump 对比 TL/TR：`emb_pre` 的 token 差异只在 patch grid 上半部非零（上 14 行每行均值约 `15.9`，下 14 行为 `0`），证明前端 patch 输入/位置顺序把红块放在正确半区；视觉塔和 merger 输出仍保留明显差异（`tower_out` rel-L2 `0.7149`/corr `0.7660`，`merged_out` rel-L2 `0.5906`/corr `0.8269`）。
> - 临时运行时逐元素校验 `ops::scatter`：big_tr 的 196 个视觉 token（hidden=5120）在 text 槽 `39..234` 全部与 scatter source bit-exact 相等（mismatch=0）；该诊断已从源码移除。禁用 splice 时 big_tl/big_tr 都答 bottom-left；启用 splice 时均偏答 upper-left，说明注入影响输出，但模型未稳定解码右/左空间差异。canonical 上游坐标约定及单独切换 `NINFER_VISION_HW_SWAP=0`、`NINFER_VISION_POS_ZERO=0` 后，big_tl/big_tr 自由回答仍均为 upper-left。
> - 临时 `NINFER_VISION_POS_FLIP_X/Y` 对照已删除：x flip 的 TL/TR 自由回答仍 upper-left；y flip 只令 big_tl 自由回答变 bottom-left，big_tr 仍 upper-left，且三问受限 left/right/top/bottom 输出与默认相同（TL left、TR left、TR bottom）。这是非一致变化，不构成坐标修复。
> - patch-only、修复 WMMA 后的受限 scale sweep：`NINFER_VISION_SCALE=0.03` / `0.3` 下 TL left、TR left、TR bottom，与默认 `0.1` 相同；`1.0` 令 TL 变 right-half（更差），TR 仍 left/bottom。保留默认 `0.1`，缩放不是右上定位的修复。
> - 2026-11-11 20:55 后续静态核对：active 与 upstream 的 patch 遍历、VisionControl patch-major indices/scatter span、text chunk-relative scatter、axis-major MRoPE copy/kernel indexing 均一致；运行时 196 个 embedding 已 bit-exact 写入文本槽 `39..234`。默认前端坐标与 upstream 有差异，但 canonical 坐标及单独 h/w swap、pos-zero 对照未修复 TR 定位，不应盲改默认值。
> - 2026-11-12 新发现（patch 权重合同）：mmproj 里有**两张** patch 权重 —— `v.patch_embd.weight` 与 `v.patch_embd.weight.1`（各 `[16,16,3,1152]`），而 `_mkvision_bf16.py:37-45` 只用前者并把同一通道平面复制到两个 temporal 槽。实测 `vision__patch_embedding.bf16` 与 `dup(w0)` 逐位相同（rel=0），确认 BF16 覆盖丢掉了第二帧权重；artifact 路与 BF16 路在同 red 输入上的 patch 投影 `rel=0.988 / corr=0.5096`（`build/vd_proj_artifact`、`build/vd_proj_bf16`）。
> - 两种双帧排列都试过（patch-only、WMMA/LayerNorm 已修、固定 seed）：通道内插 `[c0t0(w0), c0t1(w1), c1t0, c1t1, ...]`（与 processor 的 `[c][t]` payload 顺序一致）使 `big_tr` **首次答对 top-right**，但 `quad4` 变 `TR=Blue / BL=Green`（绿蓝对调）、`red` 仍 `White`；帧块式 `[t0=w0(768), t1=w1(768)]` 同样绿蓝对调、`big_tr` 退化 bottom-left、`red` 变 `Gray`。单帧复制的对照仍是 `red=White`、`quad4` 全对。
> - 结论：双帧权重是真实合同，但只换 patch 权重会在“颜色顺序”和“空间答案”之间摆动，说明还有第三处约定（payload 的通道/temporal 顺序，或 merger 之后的处理）与之耦合。两个 `vision-bf16*` 目录的 `vision__patch_embedding.bf16` 已**恢复为单帧复制**，与仓库 exporter 源码一致，不留半成品。
> - 下一步（优先）：用上游 CPU encoder 判定 patch embedding 的通道/帧索引到底走 `[c][t]` 还是 `[t][c]`（`vision_cpu.cpp` 的 patch 索引是唯一无歧义参照），再同步改 exporter（必要时 processor payload 顺序）；不要在 patch 权重上继续排列组合。
> - 2026-11-12 追加的负面结果（重要，避免误判）：想用 dump 反解“引擎实际用的 patch 权重”没有成功，**不能用它下布局结论**。①现有九图 fixture 是低秩的：`red` 的 `patches_in` 784 行完全相同（rank 1），`quad4` 也只有 4 种颜色（rank 3/4），对这种输入任何矩阵都拟合不出信息。②换成 448px 随机噪声（`patches_in` rank 768，`embed_proj` rank 784）后，自由最小二乘的残差仍有 `artifact 0.1073 / bf16 0.1167`，而用 GGUF 推出的三种候选矩阵（`dup_w0_interleaved`、`w0w1_interleaved`、`frame_major_w0w1`）正反向预测与实测 `embed_proj` 的相关都只有 `~0.0005–0.003`。这说明 `patches_in` 与 `embed_proj` 这对 dump 之间不是一个干净的 `P @ W.T` 关系（dump 时点/缓冲区或存在额外的逐 token 处理），**在把这对 dump 的关系查清之前，不能再用“两路 patch 投影 corr=0.51”当作权重布局不一致的证据**（那是 rank-1 输入上的弱证据）。
> - 仍然成立的部分：mmproj 确实有两张 patch 权重、导出确实只用了 `v.patch_embd.weight`（与 `dup(w0)` 逐位相同）；补第二帧权重后 `big_tr` 首次答对 top-right，但 `quad4` 出现绿/蓝对调——这是**行为层证据**，与 dump 反解结论无关。
> - **2026-11-12 定案（重要修复）**：bf16 patch embedding 导出有两个错误，已修 `D:\ninfer-hip-migrate\build\_mkvision_bf16.py`：①只用了 `v.patch_embd.weight`，完全丢掉 mmproj 的第二帧权重 `v.patch_embd.weight.1`；②正确合同是**按通道交错的双帧** `H = [c0t0, c0t1, c1t0, c1t1, c2t0, c2t1]`，且引擎 patch loader 会多转置一次，所以文件要写成 `H.reshape(1536,1152).T`。
>   - 判定依据（full-rank 448px 噪声图 `build/vd_noise_artifact` + `build/vd_noise_fixed`，`patches_in` rank 768、`embed_proj` rank 784）：artifact 路径 = `P @ H.T + bias`，corr `0.999947` / rel `0.01031`；修正后的 override 与同一 H 达 corr `0.999998` / rel `0.00196`，且与 artifact 实测投影 corr `0.999946` / rel `0.01038`（差异只剩 BF16 舍入）。此前基于 rank-1 `red` 得出的“两路 corr=0.51、怀疑行序”结论作废。
>   - 行为效果（patch-only override，固定 `temperature=0.0`/`seed=42`）：从 3/9 提升到 **6/9** —— `red -> Red`、`quad4` 四象限颜色全对、`bars -> red, green, blue`、`half -> left=red, right=blue`、`tb -> green on top / yellow bottom`、`big_tl -> top-left`。仍错：`big_tr`、`big_bl`、`corner_tr` 全答 `top-left`。
>   - 全量 bf16 override 仍不稳（`red -> White`、`big_tl -> bottom-left`、`big_bl -> bottom-right`、`half` 说白/黑），说明除 patch embedding 外还剩其它权重/布局问题；不要用全量 override 的结论覆盖上述判定。
> - 2026-11-12 追加判定（重要）：修好 patch 权重后，artifact 档的视觉特征**确实按位置分离**——同权重下 `big_tl` vs `big_tr` 的 `merged_out` rel `0.5078`、`big_tl` vs `big_bl` rel `0.5034`，`l0_attn_out` 分别 `0.3814`/`0.3582`。也就是说视觉塔**没有**把右/下侧信息抹掉；但同一二进制下三种位置题的文本答案全部落到 `top-left`，而且 `NINFER_VISION_POS_ZERO=0`/`NINFER_VISION_HW_SWAP=0`/`NINFER_VISION_TSTEP=grid` 的 canonical 组合也仍是全部 `top-left`。结论：剩余失败更可能位于**图像 token → LM 的空间读出**（含 mrope 位置在文本侧的落地），而不是 patch/位置编码/视觉塔。
> - 相关警告：`quad4` 通过**不能**当作空间理解证据——它的答案 `Top-left: Red | Top-right: Green | Bottom-left: Blue | Bottom-right: Yellow` 与固定模板顺序（红绿蓝黄）一致，可能是模板驱动；只有 `big_tl`/`big_tr`/`big_bl`/`corner_tr` 这组单色块题才真正考位置。九图统计时要把这两类分开看。
> - 2026-11-12 追加的直接证据（文本侧位置）：在 `text.cpp` 临时 dump 图像 span 实际进入 `run_layers` 的 `[3,T]` rope positions（trace 显示预填 chunk `0..241`，图像 token 落在 39..234），实测图像 token 的坐标是 **h 增长最快、w 恒定**：
>   `tok39 (t=39,h=0,w=0)`、`tok40 (t=39,h=1,w=0)`、`tok41 (t=39,h=2,w=0)` … 而参考实现 `third_party/upstream-ninfer/src/models/qwen3_5/frontend/processor.cpp:1170-1177` 是 `for y: for x: set(1)=current+y; set(2)=current+x`（w 最快变化）。
>   原因已定位：`infer/src/models/qwen3_5/frontend/processor.cpp:743-750` 里 `swap_hw = swap_env == nullptr || swap_env[0] != '0'` —— **环境变量未设置时默认就是交换 h/w**，与参考相反；`POS_ZERO` 同样默认开启（hw_base=0，这一条与参考一致）。
> - 但交换修正**不足以**修好：修好 patch 权重后单独设 `NINFER_VISION_HW_SWAP=0` 复跑，`big_tl→top-left`（对）、`big_tr→top-left`（错）、`big_bl→top-right`（错，真值 bottom-left）、`corner_tr→top-left`（错）；默认交换时四个单色块题全答 top-left。两种顺序都没有一致映射，说明 h/w 顺序不是唯一剩余缺陷（且默认值本身与参考不符，属应修但非充分条件）。
> - 下一步：把默认改成与参考一致（不交换）并单独验证，然后继续查为何 `big_tr/big_bl` 仍不随位置走——优先怀疑视觉塔内部的 2D RoPE 坐标与 patch 顺序（2×2 交织）仍不完全自洽，或 merger 之后的 token 顺序与 LM 的 14×14 网格假定不符。


> - 2026-11-12 修复：paired D=1152 LayerNorm kernel 在 `if (threadIdx.x == 0)` 中执行 `__shfl_sync(0xffffffffu, local.count, 0)`，实际只有 lane 0 参与却声明全 warp，产生未定义行为；trace 中 `cnt=0`，默认 paired 路径正式测试四个 aligned cases 都报 `non-finite value at index 0`，而 generic 同测试 `OK`。改成直接写已约简的 lane-0 `local.count` 后，重新编译 `layer_norm.cu`、response-file rearchive，再跑正式 `test_layer_norm.cpp` 默认 dispatch：`OK layer_norm correctness`。这是有效的 paired LayerNorm 修复；不要再把此前 norm2 数值差异归因于权重布局。
> - 修复后 mini 已重链接；runner 固定 `temperature=0.0`、`seed=42`、`max_tokens=60`，完整九图复跑。patch-only：red=White；big_tl=top-left；big_tr=top-right；big_bl=top-right（错）；corner_tr=top-right；quad4 正确；bars=蓝/绿/红（错序）；half=左蓝/右紫；tb=上灰/下绿。artifact：red=Red；big_tl=top-left；big_tr/big_bl/corner_tr=top-left；quad4 正确；bars 红/绿/蓝；half 左红/右蓝；tb 上绿/下黄。full-BF16：red=White；big_tl=top-left；big_tr/big_bl/corner_tr=bottom-left；quad4、bars正确；half左红/右绿；tb上蓝/下红。视觉空间/颜色任务仍未通过九图门禁，修复并非完整解决。
> - 同一真实 full-BF16 red 输入的修复后 norm2 数值对拍：设备 captured `l0_x_attn` 输入 + loader BF16 gamma `raw+1` 高位截断 + BF16 beta + eps `1e-6`，与设备 `l0_norm2` rel-L2 `1.521055e-5`、corr `1.0`、max `0.015625`（903168 元素中 29 个非 bit-exact，符合舍入次序差异）。先前约 `0.067` 的差距已消失，paired kernel 与同输入参考一致。重新跑 upstream CPU full-BF16 red 整塔后，与修复后 GPU `merged_out` 比较 rel-L2=`0.02266161`、corr=`0.99974545`、max abs=`2`、CPU/GPU std=`0.8054/0.8074`（per-token rel p95=`0.02870`）；早先约 `0.97` merged 差距大幅收敛。同一CPU/GPU流程在 fresh full-BF16 `big_bl` 则 rel-L2=`0.15419362`、corr=`0.98908917`、max abs=`15.75`、per-token p95=`0.23096`；不能用 red 的近似吻合证明整塔各图都数值一致，需继续查输入几何/control或内容相关累计误差，且该差异本身还不是 GPU bug 的证明。mini `infer_serve_nordc.exe`（192,053,248 bytes）已链接修复对象；完整多配置九图由 `build/_vprobe_cfg.py` 重现，回复不再截断。仍不发布服务端/多模态。
> - 2026-11-12 CPU/GPU control audit：CPU reference 的 `Matrix` 中间态是 FP32，只有最终 merged 输出转 BF16；GPU 视觉工作区/算子输出为 BF16，因此“BF16 weights/patches”并非同精度 oracle。active 与 CPU harness 的 28×28 merge-2 patch/control 顺序、y/x IDs、endpoint-aligned position interpolation、patch布局、merger gather次序、exact GELU公式及最终输出布局均核对一致；目前未证明这些路径有 mismatch。`NINFER_VISION_POST_LN` 本次比对环境未设置，条件 post-LN 分支不是本次 red/big_bl 差异的原因。CPU/GPU 相对差异 red `0.02266`、big_bl `0.15419` 需在精度匹配的逐阶段对拍下解释；数值不同本身不等于 GPU defect。
> - 当前下一步：找回/构建真正按每个算子输出 BF16 RNE 的 CPU 仿真，对 full-BF16 big_bl 的 `emb_pre` 到 merger 与设备逐阶段比较，并和 red 对照，判定内容相关差异是否仅由中间精度累计造成；若数值贴合仍需排查 LM 读出，否则定位首个分叉算子。最终仍需 9/9 且重新验证文档/测试后才可完成目标。

## 1. 唯一的量化判据

> 本节的 artifact 红/蓝差异与表格均来自 WMMA 修复前，只作为历史记录。当前修复后的九图/CPU oracle 结果见文件顶部 2026-11-12 更新与 §10，不可再据本节断言 attention 已排除。

不看回答对错，先看同一场景换一张图后**塔输出有多大差别**：

```
跑到 build\vd_red、build\vd_blue 两个目录
p = NINFER_VISION_DUMP=<dir>
rel = mean|a-b| / mean|a|      # 只看 merged_out（27 层塔 + 合并层之后的最终特征）
```

- 基线（artifact 自带视觉权重）：`merged_out` 红/蓝相对差 **0.0081**，两张互补色图的特征余弦相似度 **1.0000**
- 这就是"塔把图像抹平了"的定量证据：输入差 1.10，走完 27 层只剩 0.008

逐阶段 dump（同一批图）：

| 阶段 | 均值 | 红/蓝相对差 |
|---|---|---|
| `patches_in` | -0.294 | 1.099 |
| `emb_pre` | -0.103 | 0.845 |
| `l0_qkv` | -0.031 | 0.203 |
| `l0_attn_out` | -0.004 | 0.031 |
| `l0_x_attn` | -0.107 | 0.622 |
| `l0_up` | -2.400 | 0.136 |
| `tower_out` | 1.616 (std 48.9) | 0.019 |
| `merged_out` | -0.099 | 0.008 |

## 2. 已经被排除的因素

| 假设 | 结论 | 证据 |
|---|---|---|
| 分辨率 | 排除 | 448 与 768 两档同样错 |
| 解码 / 通道 / 归一化 | 排除 | `[vision] grid=28x28 resized=448x448 rgb_mean=127.5/127.5/63.8`，与四象限理论均值逐位吻合 |
| 位置插值 | 排除 | `config.cpp:198` 的 `side=sqrt(2304)=48`；`vision_control.cpp:173-188` 的 `i*(side-1)/(size-1)` 与四角权重等于 HF `fast_pos_embed_interpolate` |
| 注意力 | 旧结论作废；已定位并修复实现 bug | WMMA 修复前 `NINFER_VISION_SEG1=1` 的量化 tower 对照不覆盖 gfx12 B-fragment 丢维问题；修复后 identity 和独立 dense attention oracle 均通过，见顶部 2026-11-12 更新 |
| 通道顺序 | 非主因 | 六种置换（`NINFER_MEDIA_CHANNEL=rgb/rbg/grb/gbr/brg/bgr`）中 `rbg`、`brg` 能让三色竖条答对，但定位仍错 |
| 采样 / 温度 | 不能单独解释根因 | 当前 mini probe 请求使用 `temperature=0`、`seed=42`；重复运行能稳定复现同一错误答案，说明问题不能归因于随机采样，但也不能据此证明视觉通过 |

## 3. 四个实验开关（默认全关）

| 环境变量 | 作用 | 单独效果（`merged_out` 红蓝相对差） |
|---|---|---|
| `NINFER_VISION_WEIGHTS=build/vision-bf16` | 用官方 mmproj 的线性层权重覆盖 artifact | 与下一项合用才有意义 |
| `NINFER_VISION_NORM_FIX=1` | norm2 强制 weight=1、bias=0 | 0.0081 → 0.0336 |
| `NINFER_VISION_FC1_BIAS_ZERO=1` | 去掉 `fc1_bias`（artifact 里是 −2.42，mmproj 是 0.011） | 单独无改善 |
| `NINFER_VISION_MERGER_NORM_UNIT=1` | 合并前那层 LayerNorm 单位化 | 与上面几项合用 |

**四项同时打开**：0.0081 → **0.3102（38×）**，`red` 首次答对（"Red"）。但九图整体并未改善（仍是 4/9，只是错法不同），所以这些开关保持默认关闭。

## 4. artifact 与官方 mmproj 的权重不一致（待复核）

| 张量 | artifact | mmproj |
|---|---|---|
| `norm1_weight` / `ln1.weight` | mean 0.4663, std 0.79 | mean 0.002, std 0.015 |
| `norm2_weight`（layer 23） | mean 11.498 | 同量级噪声 |
| `mlp/fc1_bias` / `ffn_up.bias` | mean −2.42, [−9.44, 0.73] | mean 0.000, std 0.011 |

**但这组比对目前不可信**：artifact 的视觉张量是量化存储，例如
`vision/patch_embedding` → `{"shape":[1152,1536],"format":"q6_g64_fp16","layout":"row_split_k128_v1","bytes":1382400}`，
而 `build\_vcmp.py` 的 `Artifact.bf16()` 一律按 bf16 读，会读出 NaN、元素数还翻倍。
另外形状本身就不同（artifact 1152×1536 = 3 通道 × 2 帧 × 16×16；mmproj 1152×768 = 3 通道 × 16×16），
说明两者对 `temporal_patch_size` 的口径可能不同。

## 5. 下一步（2026-11-11 历史记录；当前状态以顶部 2026-11-12 更新为准）

1. 以已重链接 mini probe 和固定 temperature/seed 九图结果为当前基线；不能再引用 WMMA 修复前“attention 排除”或 tower 红蓝差异作为当前状态。
2. 已通过 PNG 像素确认 TL/TR/BL fixture 真值；默认、canonical upstream 坐标、单独 `NINFER_VISION_HW_SWAP=0` 和单独 `NINFER_VISION_POS_ZERO=0` 均不修复 big_tr 的 upper-left 回答，不再盲扫这些坐标默认值。
3. `ops::scatter` 的数值拷贝已用临时运行时比较验证为 bit-exact，big_tr 196 个 token 写入文本槽 `39..234`；诊断代码已移除。文本侧 X/Y 网格翻转对照也已完成：X 无改善，Y 只改变 big_tl 自由回答、未修复 big_tr，故不改 positions 或 embedding 顺序。
4. 修复 WMMA 后的 scale（0.03/0.3/1.0）与 `NINFER_VISION_NORM_FIX=1` 控制均没有修好九图；后者会使 quad4 全白。保持 scale 0.1 和默认 norm2。
5. runner 已强制 `temperature=0.0`、`seed=42`、`max_tokens=60`。patch-only 九图结果仍失败：纯红 White、四个位置题都偏答左上、half/tb 失败；quad4 和 bars 正确。不可宣称九图通过。
6. 已建立 upstream host-only full-tower CPU 参照，并做逐层 BF16-round 仿真；BF16 舍入仍不能解释最终 merged 差异，且 CPU/device 两条路尚未完全数值等价。下一步用完全相同的 GPU `l0_x_attn` 输入单独对拍 layer-0 norm2，再按层找误差突增。
7. 九图门禁通过前目标不完成；服务端/多模态不发布，除非用户另行明确授权。

## 4.1 post_ln 旧假设（已否证）

早期曾把 `mmproj` 的 `v.post_ln.weight` / `v.post_ln.bias` 解释成 artifact 缺失的额外层，并据此怀疑塔末端没有收尾 LayerNorm。后续对照确认：artifact 的 `vision/merger/norm_*` 就是该 merger norm 对应的权重语义，额外启用 `v.post_ln` 会重复归一化，默认必须关闭。

当前代码仍保留可选 post-LN 导出/加载能力，但只有同时存在权重和 bias 且显式设置 `NINFER_VISION_POST_LN=1` 才走该分支；默认路径使用 artifact merger norm。此前关于“单补 post_ln 即可修复纯色”的结论作废。

## 4.2 覆盖加载器现在支持的张量（manifest 名）

| 名字 | 来源 | 说明 |
|---|---|---|
| `vision/patch_embedding` | `v.patch_embd.weight` | 单帧 `[1152,768]` 按通道平面复制为双时间帧 `[1152,1536]` |
| `vision/layers/<i>/attention/qkv` | `v.blk.<i>.attn_qkv.weight` | 线性层 |
| `vision/layers/<i>/attention/output` | `v.blk.<i>.attn_out.weight` | 线性层 |
| `vision/layers/<i>/mlp/fc1` / `mlp/fc2` | `v.blk.<i>.ffn_up/down.weight` | 线性层 |
| `vision/layers/<i>/norm1_weight` / `norm1_bias` / `norm2_weight` / `norm2_bias` | `v.blk.<i>.ln1/ln2.*` | weight 自动 `+1` |
| `vision/layers/<i>/attention/qkv_bias` / `output_bias` | `v.blk.<i>.attn_qkv/attn_out.bias` | 原样拷贝 |
| `vision/layers/<i>/mlp/fc1_bias` / `fc2_bias` | `v.blk.<i>.ffn_up/down.bias` | 原样拷贝 |
| `vision/merger/fc1` / `fc2` | `mm.0.weight` / `mm.2.weight` | 线性层 |
| `vision/post_ln_weight` / `post_ln_bias` | `v.post_ln.*` | weight 自动 `+1`，需 `NINFER_VISION_POST_LN=1` 才生效 |

导出脚本：`python build\_mkvision_bf16.py`（329 个张量、约 0.916 GB，落到 `build/vision-bf16`），
启用：`NINFER_VISION_WEIGHTS=D:\ninfer-hip-migrate\build\vision-bf16`（**必须带 n 的路径**）。

## 4.3 关键教训：`NINFER_VISION_SCALE` 默认 0.1，写成 1.0 会毁掉所有实验

`vision.cpp` 里 `factor = scale_env ? strtof(...) : 0.1F`。我此前多轮把 `NINFER_VISION_SCALE=1.0`
当作"用真实权重"，实际上是把特征放大十倍灌进语言模型，答案退化成 "White"/"Black"，
导致一整天里"开了某开关更差"的结论有相当一部分是假阴性。**以后所有 A/B 都用默认 0.1。**

## 4.4 dump 里 patch 的顺序（分析 dump 必读）

`patches_in` 的 784 个 patch **不是行主序**，而是 `(block_y, block_x, merge_y, merge_x)`：
`idx = ((block_y*14 + block_x)*2 + merge_y)*2 + merge_x`，其中 `grid_y = block_y*2 + merge_y`、
`grid_x = block_x*2 + merge_x`。按行主序去解读会得出"整行同色"的错觉（我踩过）。
按正确顺序抽样后，`quad4` 四象限颜色与 `half`（左红右蓝）**全部与真值一致**，
证明解码/缩放/patchify 链路完全正确。

## 4.5 参照实现（oracle）现在可随时启动

```powershell
$env:PATH="D:\Qwen3-27B-Deploy-migrate\kvmem-rocm-beta2\kvmem-rocm-windows\bin;"+$env:PATH
llama-kvmem-server.exe -m E:\ninfer-models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf `
  -mm D:\ninfer-hip-migrate\build\mmproj-qwen3.8-27b-bf16.gguf `
  --device ROCm0 -ngl 99 --load-mode none --port 18200 -c 16384 -b 1024 -n 64
```
`mmproj` 路径必须是 ASCII（`S:\NLP自然语言\...` 会被 llama 读失败）。
实测参照结果：`red→Red`、`blue→Blue`、`quad4→Red Green Blue Yellow`、`bars→red, green, blue`、
`big_tl→top-left`、`big_tr→top-right`、`corner_tr→top-right`（7/7 全对）。
用 `build\_req_<case>.json` 直接 POST 到 `http://127.0.0.1:18200/v1/chat/completions`。

## 6. 追加实验（WMMA 修复前历史记录；结论以文件顶部更新为准）

| 实验 | 结果 |
|---|---|
| `NINFER_VISION_EMB_SCALE=0.5`（把两帧卷积和平均回来） | 更差：`quad4` 四色全乱、`big_tl`/`big_tr` 描述成"黑底白字"、`red`→Blue |
| `NINFER_MEDIA_CHANNEL=rbg` / `brg` 等六种置换 | 三色竖条能答对，但 `big_tr`/`corner_tr` 报"看不到红方块"，`quad4` 左列红蓝互换 |
| `NINFER_VISION_SEG1=1`（关跨 patch 注意力） | 塌缩幅度不变，排除注意力 |
| `NINFER_VISION_NORM_PLUS_ONE=1`（LN 用 1+w） | 0.0108，无效 |

结论：默认权重仍是九图最好的一档（4/9），上述开关全部保持默认关闭。

## 8. 早期计划（2026-11-11；历史记录，当前状态以顶部更新及 §10 为准）

1. **先把测法固定**：`build\_req_half.json` 等请求体没写采样参数，同一配置连跑三次得到
   white / black / black。所有对照实验前必须先设 `temperature=0`（或固定 seed），否则
   "连跑两次对"不能算通过。
2. **离群值来源**：`merged_out` 在 `post_ln` 打开后 min 反而到 −752。下一步按张量逐个量：
   `position_embedding`（[1152,2304]，量级）→ patch+pos 相加后的 `emb_pre` 分布 →
   最后一块的 `tower_out`（当前 std 19.9）。若 `tower_out` 已经爆，问题在塔内；
   若 `tower_out` 正常而合并后爆，问题在 merger 的 LayerNorm 输入口径。
3. **等价性验证的替代路径**：llama 能答对但拿不到中间张量；可以考虑用 `build\_mkvision_bf16.py`
   导出的 328 个张量在 numpy 里把 27 层塔重算一遍（每层 196 token、1152 维，成本很低），
   与引擎 dump 的 `l0_qkv`、`tower_out` 逐层对拍，就能定位到哪一层开始分叉。
4. 视觉修好之前，服务端/多模态**不做发布**（发布需用户单独授权）。

## 9. 相关提交（续）

- `a157b32`/`fedf218` `docs/VISION_STATE.md` 建立并补实验记录
- `26cf917`/`304f1bf` 导出并加载 post_ln（`NINFER_VISION_POST_LN` 门控，默认关）


- `27ccfd5` 保留 `tower_out` dump 与三个 norm/bias 开关
- `18bdd19` `NINFER_MEDIA_CHANNEL` 支持六种通道置换
- `46f131d` `docs/VISION.md`（用法 / 请求体 / 环境变量表 / 已知限制）

## 10. 2026-11-11 晚间：CPU 参照与整塔对拍（历史记录；Norm2 结论已于 2026-11-12 更新）

- 独立 host-only 编译并运行 upstream `encode_vision_on_cpu(...)`，未改 upstream 源文件；通过 build-only `cuda.h` 类型 shim 解决其公共头对 CUDA driver typedef 的引用。CPU 参照使用 448 图像几何：784 patches、196 merged tokens、patch width 1536、hidden 1152、27 层、16 heads、intermediate 4304、position grid 48、segment length 784。
- 输入来自 `build/vd_l0_layer0_fullbf16_20261111/patches_in.f32` 并重新舍入 BF16；全 BF16 塔矩阵来自 `build/vision-bf16`。按设备加载合同给 tower norm gamma 做 `raw+1`，patch 权重使用已数值验证的特殊 transpose。position/bias/post-LN/merger biases 从 GGUF 补齐。CPU 输出 `196×5120` BF16，与设备同一 full-BF16 red dump 的 `merged_out.f32` 对比，初始 FP32-CPU vs BF16-device 得到 rel-L2 `0.9668552`、corr `0.82469816`、max abs `180`；这个结果**不是根因证据**，因为 CPU encoder 在 FP32 激活中运行，额外 F32 张量还未按设备 BF16 存储舍入。
- build-only CPU 调试副本记录各层 residual。CPU FP32 与设备 BF16 的 per-layer rel-L2 从 layer 0 `0.01851` 增到 layer 1 `0.04062`，缓升至 layer 25 `0.09500`、layer 26 `0.11494`（最终 tower corr `0.99550`）；std 轨迹总体接近。layer-0 对照：`emb_pre` rel `0.002678` / corr `0.999996`；attention output rel `0.01881` / corr `0.999836`；norm2 rel `0.06955` / corr `0.997579`；FC1 rel `0.01474` / corr `0.999750`；FC2 rel `0.02983` / corr `0.999561`。此处仍受 CPU FP32 舍入差异影响，不能据此改矩阵布局或 attention。
- per-layer capture used a temporary build-only diagnostic in active `ninfer/src/models/qwen3_5/execution/vision.cpp`; it has now been removed, and clean mini was recompiled/rearchived/relinked. CPU debug copy remains under `build/vision_cpu_debug.cpp`; no `third_party/upstream-ninfer` production source was modified.
- 后续另建 CPU BF16 仿真副本：把来源 F32 的 position/bias/norm/merger tensors 先舍入到 BF16，tower gamma 的 `raw+1` 也舍入；并在 patch/position、LN、QKV、RoPE、attention、projection、residual、MLP、merger 每个输出点做 BF16 RNE。结果仍与 device 轨迹接近原 FP32 对照：`emb_pre` rel `0.002324`；tower layer26 rel `0.11331` / corr `0.99551`；最终 merged rel `0.97131` / corr `0.82469`。因此“只差每层 BF16 round”不足以解释剩余差异；CPU oracle 尚未复现某个 device-specific accumulation/layout 细节，仍需谨慎，不能因此改层权重布局。
- 修复 attention 后重测实验开关 `NINFER_VISION_NORM_FIX=1`（tower norm2 改单位 gamma、零 bias），patch-only 固定 seed 九图仍未过：`red=White`、所有单红块位置仍偏左上、`quad4` 退化为四块白、`bars` 对、`half=左白右黑`、`tb=上白下黑`。开关不作为修复，保持默认关闭。
- 2026-11-11 同输入 layer-0 norm2 对拍曾得到 rel-L2 `0.0672833`；该差异后来定位到 paired D=1152 LayerNorm 内单 lane/full-mask shuffle UB，2026-11-12 修复后实际 full-BF16 red 同输入 rel-L2 降至 `1.521055e-5`，详见顶部当前更新。旧数值仅保留为排查历史，不再是待查权重或 gamma/beta 假设。
- 当前调查和待办以文档顶部 2026-11-12 更新为准；本节仅保存当晚 CPU 参照的历史数据。
- 2026-11-12 追加：vision 归一化 `raw+1` 约定已用行为实验确认正确。BF16 覆盖关掉 `+1`（临时 `NINFER_VISION_NORM_NO_PLUS_ONE`，已删除）后 `l0_norm1` std 从 `1.4629` 降到 `0.7274`，但 `quad4` 从全对退化为 top-right 蓝 / bottom-left 绿，纯红从 White 变 Gray。保持 `+1`（GGUF 侧存的是 delta）。
- 2026-11-12 更正：此前记录的「artefact 与 BF16 的 merger 输出量级差 12.6x」来自 `build/vd_art_red` 这个 **LayerNorm 修复前的旧 dump**，不可用。用修复后二进制重新采集（`build/vd_merger_artifact`、`build/vd_merger_bf16`，同为 red）：`tower_out` std `60.82` vs `38.31`、absmax `4608` vs `1672`；`merger_normed` std `0.1573` vs `0.1391`；`merger_fc1` std `0.2597` vs `0.3362`；`merged_out` std `0.5813` vs `0.8074`。两条路径的 merger 输出量级相当，**12.6x 差异不存在**，注入尺度不构成根因。教训：视觉阶段对比必须用当前二进制重新采集，旧 dump 目录会静默混入历史数据。
- 2026-11-12 追加：参考实现的图像 MRoPE 位置已逐行核对 `third_party/upstream-ninfer/src/models/qwen3_5/frontend/processor.cpp:1158-1178`：`gh = grid.h/kMerge`、`set(0)=current`、`set(1)=current+y`、`set(2)=current+x`、`current += max(gh,gw)`。这与本引擎 canonical 环境（`POS_ZERO=0`/`HW_SWAP=0`/`TSTEP=grid`）完全一致，而已测该 canonical 组合仍不能修正空间问答，故剩余空间错误不再优先归因于图像 MRoPE 坐标。
- 2026-11-12 追加：参考实现的图像 MRoPE 位置已逐行核对 `third_party/upstream-ninfer/src/models/qwen3_5/frontend/processor.cpp:1158-1178`：`gh = grid.h/kMerge`、`set(0)=current`、`set(1)=current+y`、`set(2)=current+x`、`current += max(gh,gw)`。这与本引擎 canonical 环境（`POS_ZERO=0`/`HW_SWAP=0`/`TSTEP=grid`）完全一致，而已测该 canonical 组合仍不能修正空间问答，故剩余空间错误不再优先归因于图像 MRoPE 坐标。
