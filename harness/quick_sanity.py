import os, subprocess, re, sys
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
MODEL=r"E:\infer-models\qwen3_8_27b_gsq_iq3s.ninfer"
for m in [MODEL, r"E:\ninfer-models\qwen3_8_27b_gsq_iq3s.ninfer"]:
    if os.path.exists(m): MODEL=m
p=subprocess.run([EXE,MODEL,"The capital of France is","--ctx","512","--max-new","8","--kv","int8","--temp","0","--seed","1"],capture_output=True,text=True,errors="replace",env=ENV,timeout=900)
out=(p.stdout or "")+(p.stderr or "")
print("rc=",p.returncode)
print(out[-1200:])
