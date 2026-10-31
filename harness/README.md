# harness —— 构建与跑测脚本

这套脚本是这套移植工程真正"每天在用"的部分。它们最初是为单卡调试写的，
所以风格很直接：一次只重编一个翻译单元（TU），换进静态库，再链接 exe。

## 环境

- HIP SDK 7.2.3（`C:\Program Files\AMD\ROCm\7.2`，hipcc 在 `bin\`）
- MSVC 2022 BuildTools（`vcvars64.bat`，用于 `cl.exe` 编 `mini_main.cpp` 这类 host 代码）
- Python 3（`run_capture.py` 用）

路径写死在脚本里，换机器时改 `compile_one.py` / `run_capture.py` 顶部的常量即可。

## 增量构建三步（最常用）

```bat
python harness\compile_one.py ops/softmax_attention/dense/causal_cache/decode_t1_simt.cuh
python harness\swap_lib.py   ops/softmax_attention/dense/causal_cache/decode_t1_simt.cuh
set NINFER_NORDC=1
python harness\link_exe.py
```

注意事项（踩过的坑，别重复）：

1. `compile_one.py` **永远返回 0**，真实判据是输出里的 `rc=N <obj>` 行和 obj 的时间戳。
2. `swap_lib.py` 用 `llvm-ar r` 换库成员；引擎进程占着 `.lib` 时会静默失败，所以链接前先：
   `Get-Process ninfer_mini_nordc | Stop-Process -Force`
3. GEMV 存根 TU 重编必须加 `-DNINFER_VEC_DEBUG_OFF`（`compile_one.py` 的 `NINFER_EXTRA_DEFS`）。

## 全量构建

```bat
python harness\full_rebuild.py
```

耗时很长（attention / prompt 的 TU 单个 2-5 分钟，GEMV 约 110 秒），建议后台跑。

## 跑测

```bat
set PYTHONIOENCODING=utf-8
python harness\run_capture.py build\infer_mini_nordc.exe <artifact.ninfer> ^
       --tasks tasks\tasks_cp8km.txt --ctx 16384 --kv rk4v4-e8 --max-new 32
```

**必须用 `run_capture.py` 包一层**：exe 依赖运行时 DLL，直接双击/直接跑会 `0xC0000135`。
日志重定向用 `cmd /c "... > out.txt 2>&1"`；不要用 PowerShell 的 `>` / `Out-File`（默认 UTF-16LE，
中文会变成乱码，曾经让我误判"引擎输出了垃圾"）。

## 启动器（给最终用户，不需要命令行）

`launchers/chat.bat` 是双击即用的聊天入口：自动定位 `ninfer_mini_nordc.exe`（同目录或 `bin\`）、
把 ROCm `bin` 加进 PATH、记住模型路径（写 `model.txt`，参考 `model.txt.example`）、
四档场景菜单（ctx 16384/32768/65536/131072，前三档带 `--spec mtp --draft-tokens 2`）+ 第 5 项手输参数。

排障要点（都是真踩过的）：

1. 启动器会优先找 `ninfer_mini_nordc.exe`（发布名），再找 `infer_mini_nordc.exe`。
   **同一个目录里同时放两个名字的不同构建会静默跑到旧的那个** —— 现在启动器会先打印
   `engine: <文件名> <字节数> <时间>`，对不上就是选错了文件。
2. 不要在 bat 里写 `chcp 65001`：它会让 `set /p` 从重定向 stdin 读不到值（菜单空转）。
   UTF-8 由引擎自己 `SetConsoleOutputCP(CP_UTF8)` 负责。
3. 自动化测试 bat 用 `subprocess.run(['cmd','/c',bat], stdin=open(f,'rb'))`；
   PowerShell 的 `cmd /c "bat < file"` 嵌套引号不可靠。输入文件必须 CRLF。

## 正确性探针

- `attn_int4_probe.cu`：把 int4 KV 的 attention 与 host 参考实现逐值对拍（T=1..522，含非恒等块表）。
- `append_int4_probe.cu`：逐字节对拍 KV append 内核（K/V 的 code 与 scale 全部 0 差异）。

这两个探针是这套移植里最有用的东西：它们能在怀疑数值问题时把问题从"整条链路"缩到"一个 kernel"。
编译：`hipcc --offload-arch=gfx1201 -mno-wavefrontsize64 -std=c++20 -O2 -DNINFER_HW_WMMMA`。
