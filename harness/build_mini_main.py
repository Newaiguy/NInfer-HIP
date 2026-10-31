# Compile the milestone probe's mini_main.cpp into build_nordc/mini_main.obj (cl / vcvars).
import os, subprocess, sys

ROCM_BIN = r"C:\Program Files\AMD\ROCm\7.2\bin"
VCVARS = r"E:\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
OUT = r"D:\ninfer-hip-migrate\build_nordc"
SRC = r"D:\ninfer-hip-migrate\ninfer\src"
SRC_FULL = r"D:\ninfer-hip-migrate\milestones\20260928_decode26_prefill431\probe\mini_main.cpp"
OBJ = os.path.join(OUT, "mini_main.obj")

bat = os.path.join(OUT, "_env.bat")
with open(bat, "w", newline="\r\n") as f:
    f.write(f'@echo off\ncall "{VCVARS}" >nul 2>&1\nset\n')
r = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, errors="replace")
env = {}
for line in r.stdout.splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        env[k.upper()] = v

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
       "/D__HIP_PLATFORM_AMD__", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX", "/c",
       SRC_FULL, "/Fo" + OBJ]
r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=3000)
print(f"rc={r.returncode} {OBJ}")
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