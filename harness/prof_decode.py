import os, subprocess, re
ROOT=r"D:\ninfer-hip-migrate"; OUT=os.path.join(ROOT,"build_nordc")
ROCM_BIN=r"C:\Program Files\AMD\ROCm\7.2\bin"
VCVARS=r"E:\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
bat=os.path.join(OUT,"_env.bat")
with open(bat,"w",newline="\r\n") as f:
    f.write(f'@echo off\ncall "{VCVARS}" >nul 2>&1\nset PATH=%PATH%;{ROCM_BIN}\nset\n')
r=subprocess.run(["cmd","/c",bat],capture_output=True,text=True,errors="replace")
ENV={}
for line in r.stdout.splitlines():
    if "=" in line:
        k,v=line.split("=",1); ENV[k.upper()]=v
EXE=os.path.join(ROOT,"build","ninfer_mini_nordc.exe")
MODEL=r"E:\ninfer-models\qwen3_8_27b_gsq_iq3s.ninfer"
tf=os.path.join(ROOT,"tasks_4k.txt")
env=dict(ENV); env["NINFER_PROFILE_DECODE"]="1"
p=subprocess.run([EXE,MODEL,"--tasks",tf,"--ctx","16384","--max-new","24","--kv","int8","--temp","0","--seed","1","--no-graph"],capture_output=True,text=True,errors="replace",env=env,timeout=1800)
o=(p.stdout or "")+(p.stderr or "")
print("rc=",p.returncode)
for line in o.splitlines():
    if "prof" in line.lower() or "decode=" in line or "prefill=" in line:
        print(line[:300])
