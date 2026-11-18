# 使用教程（0.1.3r2）

面向使用者。全部命令与数字都在 RX 9070 XT 16 GB + ROCm 7.2 + Windows 11 上实测过。

---

## 1. 准备

| 项目 | 要求 |
|---|---|
| GPU | AMD RDNA4（验证卡：RX 9070 XT 16 GB） |
| 驱动/工具链 | ROCm 7.2 for Windows（HIP SDK） |
| PATH | `C:\Program Files\AMD\ROCm\7.2\bin` 必须在 PATH 里，否则 exe 起不来（退出码 `0xC0000135`） |
| 内存 | ≥16 GB（长上下文归档会占用主机内存） |
| 模型 | `Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer`（约 15 GB） |

模型默认放在 `E:\ninfer-models\`；换路径用启动器菜单 **5) 改模型路径**，或直接命令行传参。

---

## 2. 用启动器（最简单）

双击 `启动.bat`，菜单：

```
1) CLI 聊天 8K+MTB自动        ← 日常对话，最快
2) API 服务 262K+KV记忆 端口8099  ← 给程序/客户端调用
3) CLI 聊天 262K+KV记忆        ← 超长文档问答
4) 思考等级  xhigh（模板默认）
5) 改模型路径
0) 退出
```

- 选项 **1** 会以 `--ctx 16384 --spec mtp --draft-tokens 0 --max-new 8192` 启动聊天，MTP 自动挑草稿深度。
- 选项 **2** 后台常驻 HTTP 服务，按 `Ctrl+C` 停止。
- 选项 **4** 写 `effort.txt`：`xhigh`（默认）/ `medium` / `low` / `none`，对 CLI 和 API 同时生效。

> `启动.bat` 是 GBK + CRLF 编码。用编辑器改它时要保持这两点，否则 cmd 会报 `'leExtensions' is not recognized`。

---

## 3. CLI 聊天

### 3.1 直接命令行

```bat
bin\infer-mini.exe <模型.ninfer> --chat --ctx 16384 --kv rk4v4-e8 ^
    --spec mtp --draft-tokens 0 --think --max-new 8192
```

启动后：

```
NInfer-HIP chat  ctx=16384  thinking=on  think-budget=unlimited
commands: /help /clear /stats /think on|off /effort low|medium|xhigh /system <text> /exit
you > _
```

### 3.2 聊天内命令

| 命令 | 作用 |
|---|---|
| `/exit` | 退出 |
| `/clear` | 清空对话历史（重新开始，KV 复用） |
| `/stats` | 打印上一轮的 token 数与速度 |
| `/think on` / `/think off` | 开/关思考（立即生效，下一轮起） |
| `/effort low|medium|xhigh|none` | 思考强度；`xhigh` 即模板默认（最深层） |
| `/system <文本>` | 设置 system prompt |
| `/help` | 命令列表 |

实测回显：

```
you > /effort low
[effort low]
you > /effort bogus
[error] unknown effort level: bogus (template supports low|medium|xhigh, plus none to disable)
```

### 3.3 参数说明

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ctx N` | 8192 | 上下文容量；给大了会多占显存 |
| `--max-new N` | 8192 | 每轮生成预算（思考 + 正文合计） |
| `--think` / `--no-think` | 开 | 是否思考 |
| `--think-budget N` | **0 = 不限** | 思考段上限；>0 时到顶会强制收尾并提示 |
| `--effort LEVEL` | xhigh（默认） | low / medium / xhigh / none |
| `--temp T` | 0.7 | 采样温度 |
| `--spec mtp --draft-tokens N` | 关闭 / 0=自动 | 投机解码；0 时 ≤64K 用 5、>64K 用 1 |
| `--resident-tokens N` | 按 ctx | 常驻 KV 窗口（长上下文必配） |
| `--sink-tokens N` | 2048 | 开头固定的 sink 段 |
| `--kv-retrieval 1` | 关 | 开启 KV 记忆召回 |
| `--archive-mib N` | 0 | 主机归档容量（MiB），长上下文建议 4096 |
| `--kv rk4v4-e8` | 默认 | KV 量化档 |
| `--no-graph` | — | 关闭 CUDA Graph（调试用，会慢 ~20%） |
| `--ids` | — | 打印生成 token id（对拍用） |

### 3.4 思考长度

- 默认 **不设上限**：模型想多久就想多久，正文不会因为预算被切。
- 想要安全感就设上限：`--think-budget 6144`（旧版 2048 + 4096），到顶会打印
  `[note] reasoning budget (6144 tokens) reached, forcing the answer`，然后强制进入正文。
- 完全不要思考：`/think off` 或启动加 `--no-think`，答案更快但更短。

实测（同一句 "Explain entropy in one sentence."，`--max-new 512`）：

| 等级 | 生成 token | 耗时 |
|---|---|---|
| none | 29 | 0.90 s |
| low | 102 | 2.95 s |
| medium | 219 | 5.98 s |
| xhigh（默认） | 60 | 1.92 s |

> 等级改的是模板里的推理强度提示，不是硬配额，所以长度会随内容波动。

---

## 4. API 服务

### 4.1 启动

```bat
bin\infer-serve.exe <模型.ninfer> --ctx 262144 --kv rk4v4-e8 ^
    --kv-retrieval 1 --resident-tokens 32768 --archive-mib 4096 ^
    --port 8099 --think
```

设置 `--api-key KEY` 后，请求需带 `Authorization: Bearer KEY`。

就绪检查：

```bash
curl http://127.0.0.1:8099/health
```

### 4.2 基本对话

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ninfer-qwen3","messages":[{"role":"user","content":"1+1 等于几？"}]}'
```

Python（OpenAI SDK）：

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8099/v1", api_key="none")
r = c.chat.completions.create(
    model="ninfer-qwen3",
    messages=[{"role": "user", "content": "用一句话解释熵。"}],
    max_tokens=512,
)
print(r.choices[0].message.content)
```

### 4.3 流式

```python
stream = c.chat.completions.create(
    model="ninfer-qwen3",
    messages=[{"role": "user", "content": "写一首关于秋天的短诗"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

首字节实测 1–14 ms（取决于是否思考）。

### 4.4 思考设置（服务端 / 客户端）

优先级：**请求 > 服务端启动参数 > 模板默认（xhigh）**

服务端：

```bat
infer-serve.exe model.ninfer --think          :: 默认开（现在就是默认）
infer-serve.exe model.ninfer --no-think       :: 全局关
infer-serve.exe model.ninfer --effort low     :: 全局 low|medium|xhigh|none
```

客户端按请求覆盖：

```json
{"reasoning_effort": "low"}
{"chat_template_kwargs": {"reasoning_effort": "medium"}}
{"enable_thinking": false}
```

非法值 → `HTTP 400`，`{"error":{"message":"unsupported reasoning_effort: bogus (this artifact accepts low|medium|xhigh, or none)","type":"invalid_request_error"}}`。

### 4.5 视觉（看图）

```bash
curl http://127.0.0.1:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"ninfer-qwen3",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"图中是什么？"},
      {"type":"image_url","image_url":{"url":"file:///D:/pics/stop.jpg"}}
    ]}],
    "max_tokens":256
  }'
```

要点：

- `image_url` 支持本地 `file:///` 或 `data:image/...;base64,...`
- **每个 content part 必须带 `type`**，否则会被当成纯文本
- 一图多问、一请求多图都可以；九图回归与真实照片 16 问均通过

### 4.6 缺省值

| 字段 | 缺省行为 |
|---|---|
| `max_tokens` / `max_completion_tokens` | 不传 = 8192（不会返回空） |
| `temperature` | 1.0；建议显式给 0 做确定性测试 |
| `top_p` | 1.0 |
| `seed` | 不传 = 随机 |
| `stop` | 支持字符串或数组 |

---

## 5. 三套推荐配置（2026-11-16 实测）

> 口径：同一语料、口令埋在 prompt 50% 处、`--max-new 384`、贪心解码、每格都校验答案命中（HIT）。
> 三个配置的区别只在「要不要 KV 记忆」；上下文越大，KV 记忆的收益越明显。

### A. 短文 / 日常对话（≤16K，纯 MTP，不占主机归档内存）

```bat
bin\infer-mini.exe model.ninfer --chat --ctx 24576 --spec mtp --draft-tokens 5 --think --max-new 8192
```

| prompt | prefill | decode（think 开） | decode（think 关） |
|---|---|---|---|
| 8K | 936 | **64.1** | ~106 |
| 16K | 891 | **57.4** | — |

### B. 中上下文（≤32K，纯 MTP）

```bat
bin\infer-mini.exe model.ninfer --tasks doc.txt --ctx 40960 --spec mtp --draft-tokens 5 --max-new 384
```

| prompt | prefill | decode（think 开） |
|---|---|---|
| 32K | 740 | **46.5** |
| 40K | 684 | 41.6 |

> 32K 以上不要再单用 MTP：64K 时纯 MTP 只有 35.5，而开 KV 记忆是 44.3（+25%）。

### C. 长上下文（32K–262K，KV 记忆 + MTP，深度自动：>64K 为 1）

```bat
bin\infer-mini.exe model.ninfer --tasks long.txt --ctx 262144 ^
    --resident-tokens 32768 --sink-tokens 2048 --kv-retrieval 1 ^
    --archive-mib 4096 --spec mtp --max-new 384
```

| prompt | prefill | decode（think 开） | 说明 |
|---|---|---|---|
| 64K | 583 | **44.3** | 稳态窗口速度 |
| 96K | 560 | **40.4** | |
| 250K | 532 | **40.4** | 稳态窗口的 91% |
| 196K（纯 AR） | 540 | 29.8 | 不开 MTP |
| 250K（纯 AR） | 535 | 30.5 | |

> 窗口模式的 decode 只扫 `sink 2048 + 窗口 32768` 个 key，所以 250K 与 96K 同速；比 32K prompt（尚未滑动）的 51.2 低 21%，那 21% 是「窗口装满后多扫 2K sink 页」的固定开销，不是长度惩罚。

### 关于思考开关（重要）

同一个 2,845-token prompt、同一个 `--draft-tokens 5`：

| 模式 | decode |
|---|---|
| `--think`（默认） | **70.8** |
| `--no-think` | **106.5** |

思维链文本比正文难预测，草稿命中率低，所以 MTP 收益从 105 掉到 71。**引用速度时必须同时说明是否开思考。**

### 自动档规则

`--draft-tokens 0`（或不传）时：`--ctx ≤ 16384 → 5`；`16384 < --ctx ≤ 65536 → 2`；`--ctx > 65536 → 1`。
长上下文用 1 是因为 verify 批次（T=深度+1）会按 token 重扫整个窗口，窗口一旦装满这个成本就恒定了：实测 32K 44.27/44.77（平）、63K 43.77/42.36、95K 43.96/36.10、195K 41.57/33.69、251K 36.77/32.64/25.94（深度 1/2/3）。没有 KV 记忆、上下文又短时窗口等于 prompt 本身，深度 5 仍然最快（8K 106、16K 57.4）。
自动值按 `--ctx` 而不是单次 prompt 长度决定，所以短请求跑在 262K 部署里仍取 2；想要深草稿就显式传 `--draft-tokens 5`。

---

## 6. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 双击后闪退，退出码 `-1073741515` / `0xC0000135` | PATH 里没有 ROCm bin | 把 `C:\Program Files\AMD\ROCm\7.2\bin` 加进 PATH |
| `Error: invalid UTF-8 in Jinja string` | 中文走命令行参数被按 ANSI 转码 | 用 `--chat` 交互输入，别把中文放在 `-p`/argv |
| 聊天第一行输入被当成问题 | `--chat` 会把首行当开场问题 | 正常交互无影响；脚本测试要先发一行空/问题 |
| 答案写到一半被截 | `--max-new` 太小或 `--think-budget` 设了小值 | 用默认 8192 + 不设 think-budget |
| API 返回空文本 `completion=0` | 旧版 bug（缺省 max_tokens） | 0.1.3r2 已修；若仍见，显式传 `max_tokens` |
| 长上下文越跑越慢 | `--ctx` 大于可用显存 → 静默半速路径 | 降到能装下的 ctx（98K 用 100000） |
| `.bat` 报 `'leExtensions' is not recognized` | 文件被存成 LF | 用编辑器另存为 CRLF + GBK |
| 卡住不动、GPU 0% | 旧版 MTP draft 5/6 + ctx>65536 的越界 | 0.1.3 起已修；升级到 0.1.3r2 |
| 启动报 `MTP draft window 5/6 is not safe above 65536` | 触发了 0.1.2 的安全闸 | 用 `--draft-tokens 4`（0.1.3 起已移除该闸） |

---

## 7. 性能调优速查

| 想更快 | 做法 |
|---|---|
| 短对话 | 开 MTP：`--spec mtp --draft-tokens 5`（ctx≤16K） |
| 已知固定深度 | `--draft-tokens 5`（≤64K）、`1`（>64K，长上下文） |
| 不要太长思考 | `/effort low` 或 `/think off` |
| 少占显存 | 降 `--ctx`、降 `--resident-tokens`、`--archive-mib 2048` |
| 确定性复现 | `--temp 0 --ids`，API 侧传 `"temperature":0,"seed":42` |

---

## 8. 目录结构

```
ninfer-hip/
├─ 启动.bat / start.bat      一键启动器（菜单）
├─ bin\
│  ├─ infer-mini.exe         CLI / 批处理
│  └─ infer-serve.exe        OpenAI 兼容服务
├─ docs\                     文档
├─ harness\                  构建/调试脚本（可选）
├─ src\infer\                源码
└─ tasks\                    示例任务文件
```
