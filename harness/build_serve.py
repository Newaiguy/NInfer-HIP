#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the OpenAI-compatible server entry (serve_mini_main.cpp) and link it.

The engine archive is produced by harness/full_rebuild.py (or link_exe.py); this
script only compiles the entry translation unit and links it against that archive.

Usage (from anywhere):
    python harness/build_serve.py                       # -> build/ninfer_serve.exe
    python harness/build_serve.py <exe-output-path>

Environment:
    NINFER_BUILD_DIR   object/archive directory (default: <root>/build_nordc)
    NINFER_ROCM_BIN    ROCm bin directory      (default: C:/Program Files/AMD/ROCm/7.2/bin)
    NINFER_VCVARS      vcvars64.bat path       (default: E:/Microsoft Visual Studio/2022/BuildTools/VC_Auxiliary/Build/vcvars64.bat)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "infer", "src")
INC = os.path.join(ROOT, "src", "infer", "include")
TP = os.path.join(ROOT, "src", "infer", "third_party")
OUT = os.environ.get("NINFER_BUILD_DIR", os.path.join(ROOT, "build_nordc"))
ROCM = os.environ.get("NINFER_ROCM_BIN", r"C:/Program Files/AMD/ROCm/7.2/bin")
VCVARS = os.environ.get(
    "NINFER_VCVARS",
    r"E:/Microsoft Visual Studio/2022/BuildTools/VC_Auxiliary/Build/vcvars64.bat",
)
ENTRY = os.path.join(ROOT, "harness", "serve_mini_main.cpp")
OBJ = os.path.join(OUT, "serve_mini.obj")


def capture_env():
    os.makedirs(OUT, exist_ok=True)
    bat = os.path.join(OUT, "_env_serve.bat")
    with open(bat, "w", newline="\r\n") as handle:
        handle.write('@echo off\ncall "%s" >nul 2>&1\nset PATH=%%PATH%%;%s\nset\n' % (VCVARS, ROCM))
    result = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, errors="replace")
    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.upper()] = value
    env["PATH"] = env.get("PATH", "") + ";" + os.environ.get("PATH", "")
    return env


ENV = capture_env()
exe = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "ninfer_serve.exe")

cl = None
for part in ENV.get("PATH", "").split(";"):
    if part and os.path.exists(os.path.join(part, "cl.exe")):
        cl = os.path.join(part, "cl.exe")
        break
if cl is None:
    print("ERROR: cl.exe not found; set NINFER_VCVARS")
    sys.exit(1)

compile_cmd = [
    cl, "/std=c++20", "/O2", "/EHsc", "/Zc:__cplusplus", "/utf-8",
    "/I" + SRC, "/I" + INC, "/I" + TP,
    "/I" + os.path.join(TP, "cpp-httplib"),
    "/I" + os.path.join(TP, "llama-jinja"),
    "/I" + os.path.join(TP, "spdlog", "include"),
    "/I" + os.path.join(ROCM, "..", "include").replace("\\", "/"),
    "/D__HIP_PLATFORM_AMD__", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX", "/DUTF8PROC_STATIC",
    "/c", ENTRY, "/Fo" + OBJ,
]
result = subprocess.run(compile_cmd, capture_output=True, text=True, errors="replace", env=ENV, timeout=1800)
print("compile rc=%d %s" % (result.returncode, OBJ))
if result.returncode != 0:
    print(((result.stderr or "") + "\n" + (result.stdout or ""))[-4000:])
    sys.exit(1)

lib = os.path.join(OUT, "ninfer_hip.lib")
if not os.path.exists(lib):
    print("ERROR: missing %s (run harness/full_rebuild.py first)" % lib)
    sys.exit(1)

os.makedirs(os.path.dirname(os.path.abspath(exe)), exist_ok=True)
link_cmd = [
    os.path.join(ROCM, "hipcc.exe"), "--offload-arch=gfx1201", "-mno-wavefrontsize64",
    "-std=c++20", "-O2",
    "-I" + SRC, "-I" + INC, "-D__HIP_PLATFORM_AMD__", "-D__CUDACC__",
    os.path.join(ROCM, "..", "lib", "hipblas.lib"),
    os.path.join(ROCM, "..", "lib", "rocblas.lib"),
    lib, OBJ, "-o", exe,
]
result = subprocess.run(link_cmd, capture_output=True, text=True, errors="replace",
                        env=ENV, cwd=OUT, timeout=5400)
if result.returncode != 0:
    print("link FAIL; first error lines:")
    for line in ((result.stderr or "") + (result.stdout or "")).splitlines():
        if "error" in line.lower():
            print(" ", line[:200])
    sys.exit(1)
print("linked:", exe, "bytes=", os.path.getsize(exe))
