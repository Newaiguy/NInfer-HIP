# NInfer-HIP `0.1.3`

在 **单张 16 GB AMD Radeon RX 9070 XT（RDNA4 / gfx1201）** 上跑 **27B 量化模型**：
262K 上下文可用、8K 档 decode **35.7 t/s**、开 MTP 草稿 **105 t/s**，并支持 **图像输入** 与
**OpenAI 兼容 HTTP 服务**。全部内核用 HIP 手写，不依赖 CUDA。

这是一个**功能完整、性能定型**的版本：视觉链路修好（九图确定性回归 9/9）、KV 记忆可跨 192K
召回、服务端可直接对接 OpenAI SDK。

---

## 1. 能力总览

| 能力 | 状态 | 说明 |
|---|---|---|
| 文本推理（CLI） | ✅ | `infer-mini.exe`，交互式 chat / 批量任务 |
| OpenAI 兼容服务 | ✅ | `infer-serve.exe`，`/v1/chat/completions`、SSE 流式 |
| 图像输入（多模态） | ✅ | PNG/JPEG/GIF/BMP/TIFF，单图 / 多图 |
| 长上下文 + KV 记忆 | ✅ | 最多 262,144 token；常驻窗口 32K + 主机归档 + 按需召回 |
| MTP 投机解码 | ✅ | 草稿深度自适应（≤64K 用 5，>64K 用 4） |
| 连续多轮对话 | ✅ | KV 复用，无需重算历史 |
| 批处理 / 多请求并发 | ❌ | 单请求串行（见 §8） |

---

## 2. 性能实测

测试机：RX 9070 XT 16 GB、ROCm 7.2、Windows 11；模型 `Qwen3.8-27B-GSQ-RCO-IQ3_S`
（`.ninfer` 工件 15.0 GB，decode 驻留权重 11.76 GB）。数值为端到端 token/s。

### 2.1 短上下文

| 档位 | prefill t/s | decode t/s | 备注 |
|---|---|---|---|
| 8K（`--ctx 16384`） | **948.4** | **35.69** | AR，单请求 |
| 8K + MTP（draft 5） | **848.2** | **105.53** | 投机解码，3.0× |
| 96K（`--ctx 100000`） | **423.3** | **25.26** | AR |

### 2.2 长上下文（KV 记忆：常驻 32K + 归档 4 GiB + 召回）

| 上下文 | prefill t/s | decode t/s | 命中 |
|---|---|---|---|
| 196K | **540.4** | **29.77** | ✅ |
| 250K | **534.9** | **30.50** | ✅ |
| 261K（上限 262,144） | **534.1** | **29.79** | ✅ |

> 262,144 是模型自身位置编码上限，超过会启动时直接拒绝并提示。

### 2.3 MTP 草稿深度的影响

| 上下文 | AR | MTP d2 | MTP d5 | 说明 |
|---|---|---|---|---|
| 8K | 35.7 | 79.5 | **105.4** | 峰值在 d5–d6 |
| 32K（常驻） | 32.3 | — | **47.3** | |
| 32K（KV 记忆） | — | — | **52.8** | 每 token 扫的 KV 更少 |
| 96K（KV 记忆） | 25.2 | 55.1 | 61.9（d4） | >64K 自动降到 d4 |

`--draft-tokens 0`（默认）按上下文自动选择：**≤65536 用 5，>65536 用 4**。

### 2.4 与 llama.cpp（llama-kvmem）同机同权重对比

| 项目 | 本引擎 | llama-kvmem |
|---|---|---|
| 长上下文准确率（8 例：25/50/75/92% 深度 needle + 多文档 + 196K/250K） | **8/8** | 8/8 |
| prefill（196K / 250K） | 537.9 / 533.5 | 331.9 / 323.1 |
| decode（8K，同开 MTP2） | **41.05** | 32.85 |
| decode（196K，同开 MTP2） | **40.01** | 32.93 |
| decode（250K，同开 MTP2） | 32.42 | 32.93 |

结论：**准确率相同，prefill 快 1.4–1.6×，decode 除 250K 档持平外全部更快。**

### 2.5 视觉

| 项目 | 结果 |
|---|---|
| 九图确定性回归（量化档 / patch-only BF16 / 全 BF16） | **各 9/9** |
| 真实照片交叉验证（16 问：颜色/计数/OCR/形状/地标/朝向等） | **16/16**（llama.cpp 同题 16/16） |

---

## 3. 部署

### 3.1 硬件与系统

- GPU：AMD Radeon RX 9070 XT（gfx1201），16 GB 显存
- 运行时：ROCm 7.2（需把 `C:\Program Files\AMD\ROCm\7.2\bin` 加进 `PATH`）
- 系统：Windows 11 x64

### 3.2 需要的文件

| 文件 | 说明 |
|---|---|
| `bin\infer-mini.exe` | CLI 推理引擎（191,981,568 B） |
| `bin\infer-serve.exe` | OpenAI 兼容服务（192,317,952 B） |
| 模型工件 `*.ninfer` | 由转换工具从 GGUF 生成，约 15 GB，不随仓库分发 |

校验和见 `MANIFEST.sha256` 与 `VERSION.txt`。

### 3.3 快速开始（CLI）

```powershell
$env:PATH = "C:\Program Files\AMD\ROCm\7.2\bin;$env:PATH"
.\bin\infer-mini.exe "D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer" `
  --kv rk4v4-e8 --ctx 16384
```

进入交互对话；`--tasks <file>` 可批量跑文件里的问题。

### 3.4 启动服务

```powershell
$env:PATH = "C:\Program Files\AMD\ROCm\7.2\bin;$env:PATH"
$env:NINFER_KV_ARCHIVE_MIB = "4096"
.\bin\infer-serve.exe "D:\models\Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer" `
  --kv rk4v4-e8 --ctx 262144 --resident-tokens 32768 --sink-tokens 2048 --kv-retrieval 1
```

默认监听 `http://127.0.0.1:8099`：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | 对话（支持 `stream`、`image_url`） |

---

## 4. 使用示例

### 4.1 最简请求

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ninfer","messages":[{"role":"user","content":"用一句话解释什么是 RoPE"}],"max_tokens":128}'
```

### 4.2 流式

```bash
curl -N http://127.0.0.1:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ninfer","stream":true,"messages":[{"role":"user","content":"从 1 数到 20"}]}'
```

### 4.3 图像输入

```json
{
  "model": "ninfer",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "这张图里有什么？"},
      {"type": "image_url", "image_url": {"url": "file:///D:/pics/bus.jpg"}}
    ]
  }],
  "max_tokens": 256
}
```

> `content` 的每个元素**必须带 `type`**；缺少 `type` 的部分会被当作纯文本，图片被忽略。

### 4.4 Python（OpenAI SDK）

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8099/v1", api_key="x")
print(client.chat.completions.create(
    model="ninfer",
    messages=[{"role": "user", "content": "你好"}],
).choices[0].message.content)
```

---

## 5. 参数与推荐配置

### 5.1 常用参数

| 参数 | 说明 | 推荐 |
|---|---|---|
| `--kv <profile>` | KV 量化档 | `rk4v4-e8` |
| `--ctx <n>` | 上下文窗口上限 | 短任务 16384 / 长任务 262144 |
| `--resident-tokens <n>` | 常驻窗口大小 | 32768 |
| `--sink-tokens <n>` | 头部固定保留 token | 2048 |
| `--kv-retrieval <0/1>` | 开启 KV 记忆与召回 | 1（长上下文） |
| `--spec mtp` | 开启 MTP 投机解码 | 长生成任务 |
| `--draft-tokens <n>` | 草稿深度，0 = 自适应 | 0 |
| `--max-new <n>` | 单次最大生成 token | 按需 |

### 5.2 环境变量

| 变量 | 说明 | 推荐 |
|---|---|---|
| `NINFER_KV_ARCHIVE_MIB` | 主机归档池大小（MiB） | 4096 |
| `NINFER_KV_RETRIEVE` | 召回开关（0 关） | 1 |
| `NINFER_KV_RECALL_PAGES` | 每次召回页数（64 token/页） | 112 |
| `NINFER_VISION_WEIGHTS` | 视觉权重覆盖目录（调试用） | 不设 |
| `NINFER_DECODE_GRAPH` | 设 0 关闭 CUDA Graph（仅调试） | 不设 |

### 5.3 三套推荐配置

```powershell
# A. 低延迟短对话（8K，最快）
--ctx 16384 --kv rk4v4-e8 --spec mtp --draft-tokens 0

# B. 长文档问答（196K–262K，KV 记忆）
--ctx 262144 --resident-tokens 32768 --sink-tokens 2048 --kv-retrieval 1 --kv rk4v4-e8
#   配合 $env:NINFER_KV_ARCHIVE_MIB=4096

# C. 长文档 + 长生成（KV 记忆 + MTP）
--ctx 262144 --resident-tokens 32768 --sink-tokens 2048 --kv-retrieval 1 `
            --kv rk4v4-e8 --spec mtp --draft-tokens 4
```

---

## 6. 构建

需要 Visual Studio 2022 Build Tools + ROCm 7.2（HIP）。

```powershell
python probe_rebuilt\full_rebuild.py          # 编译全部 TU
python build\_rearchive_current.py            # 归档（响应文件，避免命令行过长）
python probe_rebuilt\link_exe.py              # 链接 infer-mini.exe
```

服务端额外一步：`python build\_compile_serve.py` + `python build\_link_serve.py`。
细节见 `docs/BUILD.md`。

自检：

```powershell
python build\_gate.py --arms 8k,96k,mtp       # 性能门禁（对基线做回归判定）
```

---

## 7. 已知限制

- **单流 decode 约 35–36 t/s 已接近本机瓶颈**：整步 27.3 ms 里 MLP 15.65 ms（573 GB/s，
  超过独立微基准）、GDN 投影 4.22 ms（590 GB/s）、lm_head 1.31 ms（549 GB/s）都已顶到
  实测带宽；余下是 GDN 递推/状态传递的串行延迟（约 7.7 ms），需要重构内核才能再降。
  更高吞吐请用 MTP（105 t/s）或后续的多请求批处理。
- **单请求串行**：服务端目前一次处理一个请求，没有 continuous batching。
- **KV 记忆是启发式的**：按页均值打分选段，精排只重排前 256 页；单文档/多文档各有验证，
  但没有成套长文基准。
- **262,144 token 是硬上限**（模型位置编码）。
- **图像输入上限**：单图按 448×448 归一化，多图按请求顺序拼接。

---

## 8. 文档索引

| 文件 | 内容 |
|---|---|
| `VERSION.txt` | 版本、二进制校验和、实测数据 |
| `RELEASE_NOTES_0.1.3.md` | 本版变更说明 |
| `docs/BUILD.md` | 构建与工具链 |
| `docs/VALIDATION.md` | 门禁、回归与验收记录 |
| `docs/PERF-CAPACITY.md` | 性能与容量实测 |
| `docs/KVMEM.md` | 长上下文 / KV 记忆设计与参数 |
| `docs/VISION.md` | 视觉链路用法与环境变量 |
| `docs/CHANGELOG.md` | 历史变更 |
| `docs/PATCHLIST.md` | 相对上游的改动清单 |
| `MANIFEST.sha256` | 发布文件校验和 |

---

## 9. 许可

见 `LICENSE` 与 `NOTICE`。