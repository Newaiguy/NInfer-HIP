#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Link mini_main.obj + ninfer_hip.lib into the D-drive mini build (nordc)."""
import os
import subprocess
import sys

SRC   = r"D:\ninfer-hip-migrate\ninfer\src"
INC   = r"D:\ninfer-hip-migrate\ninfer\include"
SHIM  = r"D:\ninfer-hip-migrate\shim"
TP    = r"D:\ninfer-hip-migrate\ninfer\third_party"
JINJA = r"D:\ninfer-hip-migrate\ninfer\third_party\llama-jinja"
OUT   = r"D:\ninfer-hip-migrate\build_nordc"
NORDC = os.environ.get("NINFER_NORDC") == "1"
ROCM_BIN = r"C:\Program Files\AMD\ROCm\7.2\bin"
VCVARS = r"E:\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

def capture_env():
    bat = os.path.join(OUT, "_env.bat")
    with open(bat, "w", newline="\r\n") as f:
        f.write(f'@echo off\ncall "{VCVARS}" >nul 2>&1\nset PATH=%PATH%;{ROCM_BIN}\nset\n')
    r = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, errors="replace")
    env = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.upper()] = v
    return env

ENV = capture_env()
ENV["PATH"] = ENV.get("PATH", os.environ["PATH"])

mini_obj = os.path.join(OUT, "mini_main.obj")
if not os.path.exists(mini_obj):
    print("ERROR: missing", mini_obj)
    sys.exit(1)

exe = os.environ.get("NINFER_LINK_OUT") or (r"D:\ninfer-hip-migrate\build\infer_mini_nordc.exe" if NORDC else r"D:\ninfer-hip-migrate\build\infer_mini.exe")
LINK_FLAGS = ["--offload-arch=gfx1201", "-mno-wavefrontsize64", "-std=c++20", "-O2"]
if not NORDC:
    LINK_FLAGS += ["-fgpu-rdc", "--hip-link"]
LINK_FLAGS += ["-I" + SHIM, "-I" + SRC, "-I" + INC,
               "-D__HIP_PLATFORM_AMD__", "-D__CUDACC__",
               os.path.join(r"C:\Program Files\AMD\ROCm\7.2", "lib", "hipblas.lib"),
               os.path.join(r"C:\Program Files\AMD\ROCm\7.2", "lib", "rocblas.lib"),
               os.path.join(OUT, "ninfer_hip.lib"), mini_obj, "-o", exe]
r = subprocess.run([os.path.join(ROCM_BIN, "hipcc.exe")] + LINK_FLAGS,
                   capture_output=True, text=True, errors="replace", env=ENV, cwd=OUT,
                   timeout=5400)
err = (r.stderr or "") + (r.stdout or "")
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
with open(os.path.join(OUT, "logs", "link.log"), "w", encoding="utf8") as f:
    f.write(err)
if r.returncode != 0:
    lines = [l for l in err.splitlines() if "error" in l.lower()]
    print(f"link FAIL ({len(lines)} error lines); first 15:")
    for l in lines[:15]: print(" ", l[:200])
    sys.exit(1)
print("linked:", exe, "exists=", os.path.exists(exe), "bytes=", (os.path.getsize(exe) if os.path.exists(exe) else -1))