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

# build a long real-text tasks file
srcs=[os.path.join(ROOT,"HANDOFF.md"),
      os.path.join(ROOT,"build","HANDOFF_20261027.md"),
      os.path.join(ROOT,"build","HANDOFF_20261016.md"),
      os.path.join(ROOT,"HANDOFF_20260930.md")]
text=""
for s in srcs:
    if os.path.exists(s):
        text += open(s,encoding="utf8",errors="replace").read() + "\n\n"
print("source chars:", len(text))
tasks=os.path.join(ROOT,"tasks_long.txt")
with open(tasks,"w",encoding="utf8",newline="\n") as f:
    f.write("###ID long\n")
    f.write(text)
    f.write("\nPlease write a very detailed, section-by-section technical summary of the document above. "
            "Be thorough and do not stop early.\n")
    f.write("###END\n")
print("tasks file bytes:", os.path.getsize(tasks))

EXE=os.path.join(ROOT,"build","ninfer_mini_nordc.exe")
MODEL=r"E:\ninfer-models\qwen3_8_27b_gsq_iq3s.ninfer"
def run(label, extra, ctx, gen):
    env=dict(ENV); env.update(extra)
    cmd=[EXE,MODEL,"--tasks",tasks,"--ctx",str(ctx),"--max-new",str(gen),"--kv","int8","--temp","0.9","--seed","7"]
    p=subprocess.run(cmd,capture_output=True,text=True,errors="replace",env=env,timeout=3600)
    out=(p.stdout or "")+"\n"+(p.stderr or "")
    m=re.search(r"finish=(\d+) prompt=(\d+) gen=(\d+) prefill=([0-9.]+) decode=([0-9.]+)", out)
    if not m:
        tail=[l for l in out.splitlines() if l.strip()][-4:]
        print(label,"NO MATCH rc=",p.returncode,"|"," || ".join(t[:160] for t in tail), flush=True)
        return None,None
    fin,pr,gen_,pf,dc = int(m.group(1)),int(m.group(2)),int(m.group(3)),float(m.group(4)),float(m.group(5))
    ms = 1000.0/dc if dc>0 else None
    print("%-40s prompt=%d gen=%d finish=%d decode=%7.2f t/s  %s ms/step  prefill=%.1f"
          % (label,pr,gen_,fin,dc,("%7.2f"%ms) if ms else "  n/a  ",pf), flush=True)
    return dc,pr

print("=== long context (real text) ===")
a,_=run("SIMT(default)",{},49152,24)
b,_=run("WMMA+SKIP_ATTN",{"NINFER_ATTN_SIMT":"0","NINFER_SKIP_ATTN":"1"},49152,24)
c,_=run("WMMA",{"NINFER_ATTN_SIMT":"0"},49152,24)
if a and b: print("\nattn(SIMT) = %.2f ms" % (1000.0/b - 1000.0/a if False else (1000.0/b)-(1000.0/a)))
if b: print("no-attn floor  = %.2f ms (=%.1f t/s)" % (1000.0/b, b))
