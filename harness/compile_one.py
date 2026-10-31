import os, subprocess, sys

# Compile a single .cu / .cpp for the D:\ninfer-hip-migrate tree (adaptation of the
# pre-migration compile_one.py; OUT/SRC pointed at the migrated copy).
ROCM_BIN = r"C:\Program Files\AMD\ROCm\7.2\bin"
VCVARS = r"E:\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
OUT = r"D:\ninfer-hip-migrate\build_nordc"
SRC = r"D:\ninfer-hip-migrate\ninfer\src"

src_rel = sys.argv[1]
is_cu = src_rel.endswith(".cu")
ext = ".o" if is_cu else ".obj"
obj_name = os.path.relpath(os.path.join(SRC, src_rel), SRC).replace("\\", "_") + ext
obj = os.path.join(OUT, obj_name)

bat = os.path.join(OUT, "_env.bat")
with open(bat, "w", newline="\r\n") as f:
    f.write(f'@echo off\ncall "{VCVARS}" >nul 2>&1\nset PATH=%PATH%;{ROCM_BIN}\nset\n')
r = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, errors="replace")
env = {}
for line in r.stdout.splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        env[k.upper()] = v

if is_cu:
    cmd = [os.path.join(ROCM_BIN, "hipcc.exe"), "--offload-arch=gfx1201", "-mno-wavefrontsize64",
           "-std=c++20", "-O2",
           r"-ID:\ninfer-hip-migrate\shim", "-I" + SRC, r"-ID:\ninfer-hip-migrate\ninfer\include",
           r"-ID:\ninfer-hip-migrate\ninfer\third_party",
           r"-ID:\ninfer-hip-migrate\ninfer\third_party\ggml-quants\src",
           r"-ID:\ninfer-hip-migrate\ninfer\third_party\ggml-quants\include",
           r"-ID:\ninfer-hip-migrate\ninfer\third_party\cub",
           r"-ID:\ninfer-hip-migrate\ninfer\third_party\llama-jinja",
           r"-ID:\ninfer-hip-migrate\ninfer\third_party\spdlog\include",
           "-D__HIP_PLATFORM_AMD__", "-D__CUDACC__", "-DGGML_USE_HIP", "-DRDNA4", "-DNINFER_HW_WMMMA", "-DNINFER_K_TILE", "-DNINFER_ACCEPT_FORCE_SINGLE", "-c",
           os.path.join(SRC, src_rel), "-o", obj]
    extra = os.environ.get("NINFER_EXTRA_DEFS")
    if extra:
        for d in extra.split():
            cmd.insert(cmd.index("-c"), d)
else:
    cl = None
    for p in env.get("PATH", "").split(";"):
        if p and os.path.exists(os.path.join(p, "cl.exe")):
            cl = os.path.join(p, "cl.exe")
            break
    if cl is None:
        print("ERROR: cl.exe not found")
        sys.exit(1)
    cmd = [cl, "/std:c++20", "/O2", "/EHsc", "/Zc:__cplusplus", "/utf-8",
           r"/ID:\ninfer-hip-migrate\shim", "/I" + SRC, r"/ID:\ninfer-hip-migrate\ninfer\include",
           r"/ID:\ninfer-hip-migrate\ninfer\third_party",
           r"/ID:\ninfer-hip-migrate\ninfer\third_party\llama-jinja",
           r"/ID:\ninfer-hip-migrate\ninfer\third_party\spdlog\include",
           r"/IC:\Program Files\AMD\ROCm\7.2\include",
           "/D__HIP_PLATFORM_AMD__", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX", "/DUTF8PROC_STATIC",
           "/c",
           os.path.join(SRC, src_rel), "/Fo" + obj]

r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=3000)
print(f"rc={r.returncode} {obj_name}")
if r.returncode != 0:
    out = (r.stderr or "") + "\n" + (r.stdout or "")
    lines = out.splitlines()
    print(f"  total output lines: {len(lines)}")
    shown = 0
    for line in lines:
        if "error" in line.lower():
            print(line[:260])
            shown += 1
            if shown > 30:
                break
    if shown == 0:
        print(out[-3000:])
