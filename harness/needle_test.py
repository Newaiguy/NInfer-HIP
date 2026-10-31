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
MODEL=r"E:\infer-models\qwen3_8_27b_gsq_iq3s.ninfer"
for c in [MODEL, r"E:\ninfer-models\qwen3_8_27b_gsq_iq3s.ninfer"]:
    if os.path.exists(c): MODEL=c
raw=open(os.path.join(ROOT,"HANDOFF.md"),encoding="utf8",errors="replace").read()
FACT="Benchmark note: the Kokkos prototype reached 412 GB/s on the reference card in October."
body=FACT+"\n\n"+raw[:26000]+"\n\nQuestion: According to the benchmark note near the top of this document, what throughput figure in GB/s did the Kokkos prototype reach? Reply with the number only."
tf=os.path.join(ROOT,"tasks_needle.txt")
with open(tf,"w",encoding="utf8") as f:
    f.write("###ID needle\n"+body+"\n###END\n")
def run(label,kv,extra=None):
    env=dict(ENV); 
    if extra: env.update(extra)
    p=subprocess.run([EXE,MODEL,"--tasks",tf,"--ctx","32768","--max-new","16","--kv",kv,"--temp","0","--seed","1"],capture_output=True,text=True,errors="replace",env=env,timeout=1800)
    o=(p.stdout or "")
    m=re.search(r"prompt=(\d+) gen=(\d+) prefill=([0-9.]+) decode=([0-9.]+)",o)
    b=o.split("===TASK needle===")[-1].split("===IDS")[0].strip().replace("\n"," ")
    print("=== %-16s rc=%d prompt=%s decode=%-7s FIND412=%s" % (label,p.returncode,m.group(1) if m else "?",m.group(4) if m else "?", "412" in b))
    print("   ",b[:160]); print()
run("int8","int8")
run("bf16","bf16")
