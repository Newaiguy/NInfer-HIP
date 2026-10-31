import os, subprocess, sys

ROOT = "D:/ninfer-hip-migrate"
SRC = ROOT + "/ninfer/src"
OUT = ROOT + "/build_nordc"
ROCM_BIN = "C:/Program Files/AMD/ROCm/7.2/bin"
VCVARS = "E:/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat"

src = sys.argv[1]
obj_name = sys.argv[2]

bat = os.path.join(OUT, "_env.bat")
with open(bat, "w", newline="\r\n") as f:
    f.write('@echo off\ncall "' + VCVARS + '" >nul 2>&1\nset PATH=%PATH%;' + ROCM_BIN + '\nset\n')
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

obj = os.path.join(OUT, obj_name)
cmd = [cl, "/std:c++20", "/O2", "/EHsc", "/Zc:__cplusplus", "/utf-8",
       "/I" + ROOT + "/shim", "/I" + SRC, "/I" + ROOT + "/ninfer/include",
       "/I" + ROOT + "/ninfer/third_party",
       "/I" + ROOT + "/ninfer/third_party/llama-jinja",
       "/I" + ROOT + "/ninfer/third_party/spdlog/include",
       "/IC:/Program Files/AMD/ROCm/7.2/include",
       "/D__HIP_PLATFORM_AMD__", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX", "/DUTF8PROC_STATIC",
       "/c", src, "/Fo" + obj]

r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=env, timeout=1200)
print("rc=%d %s" % (r.returncode, obj_name))
if r.returncode != 0:
    out = (r.stderr or "") + "\n" + (r.stdout or "")
    print(out[-4000:])
