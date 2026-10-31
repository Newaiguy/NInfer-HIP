# -*- coding: utf-8 -*-
"""Full rebuild of ninfer_hip.lib for the D-drive tree, then relink.

Compiles every .cu/.cpp under ninfer/src (the gguf/ggml kernels get the HIP defines;
everything else uses the same flags as compile_one.py). Runs compilers in parallel,
writes objects into build_nordc, then rebuilds ninfer_hip.lib and relinks the exe.
"""
import os
import subprocess
import sys
import glob
import concurrent.futures

ROCM_BIN = r"C:\Program Files\AMD\ROCm\7.2\bin"
VCVARS = r"E:\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
OUT = r"D:\ninfer-hip-migrate\build_nordc"
SRC = r"D:\ninfer-hip-migrate\ninfer\src"
SHIM = r"D:\ninfer-hip-migrate\shim"
INC = r"D:\ninfer-hip-migrate\ninfer\include"
TP = r"D:\ninfer-hip-migrate\ninfer\third_party"
QUANTS = [os.path.join(TP, "ggml-quants", "include"), os.path.join(TP, "ggml-quants", "src")]

# ---- capture env once ----
bat = os.path.join(OUT, "_env_full.bat")
with open(bat, "w", newline="\r\n") as f:
    f.write(f'@echo off\ncall "{VCVARS}" >nul 2>&1\nset PATH=%PATH%;{ROCM_BIN}\nset\n')
r = subprocess.run(["cmd", "/c", bat], capture_output=True, text=True, errors="replace")
ENV = {}
for line in r.stdout.splitlines():
    if "=" in line:
        k, v = line.split("=", 1)
        ENV[k.upper()] = v
ENV["PATH"] = ENV.get("PATH", "") + ";" + os.environ["PATH"]

COMMON_INCS = ["-I" + SHIM, "-I" + SRC, "-I" + INC,
               # cub/ is the include root's directory contents; <cub/...> needs TP itself.
               "-I" + TP,
               "-I" + os.path.join(TP, "cub"),
               "-I" + os.path.join(TP, "ggml-quants", "src"),
               "-I" + os.path.join(TP, "ggml-quants", "include"),
               "-I" + os.path.join(TP, "llama-jinja"),
               "-I" + os.path.join(TP, "spdlog", "include")]

def build_one(rel):
    src = os.path.join(SRC, rel)
    is_cu = rel.endswith(".cu")
    stem = rel.replace("\\", "_").replace("/", "_")
    ext = ".o" if is_cu else ".obj"
    obj = os.path.join(OUT, stem + ext)
    ggml_kernel = rel.startswith(("ops/linear/gguf/ggml_", "ops\\linear\\gguf\\ggml_"))
    gguf_linear = rel.startswith(("ops/linear/gguf/gguf_linear", "ops\\linear\\gguf\\gguf_linear"))
    if is_cu:
        cmd = [os.path.join(ROCM_BIN, "hipcc.exe"), "--offload-arch=gfx1201",
               "-mno-wavefrontsize64", "-std=c++20", "-O2",
               "-D__HIP_PLATFORM_AMD__", "-D__CUDACC__", "-DGGML_USE_HIP", "-DRDNA4",
               "-DNINFER_HW_WMMMA", "-DNINFER_K_TILE", "-DNINFER_ACCEPT_FORCE_SINGLE"] + COMMON_INCS
        if ggml_kernel:
            cmd += ["-I" + os.path.join(os.path.dirname(src)),
                    "-I" + QUANTS[0], "-I" + QUANTS[1]]
        if "ggml_vec_" in rel:
            # The vec kernels' g_vec_rdbg/g_vec_wdbg debug device symbols collide under
            # the single fatbin link; disable them (see ggml_bridge_vec.cuh guards).
            cmd += ["-DNINFER_VEC_DEBUG_OFF"]
        cmd += ["-c", src, "-o", obj]
    else:
        if gguf_linear:
            # hipcc so the bridge ABI (__hip_bfloat16) matches ggml_bridge.cu
            cmd = [os.path.join(ROCM_BIN, "hipcc.exe"), "--offload-arch=gfx1201",
                   "-mno-wavefrontsize64", "-std=c++20", "-O2",
                   "-D__HIP_PLATFORM_AMD__", "-D__CUDACC__",
                   "-DNINFER_HW_WMMMA", "-DNINFER_K_TILE", "-DNINFER_ACCEPT_FORCE_SINGLE"] + COMMON_INCS
            cmd += ["-c", src, "-o", obj]
        else:
            cl = None
            for p in ENV.get("PATH", "").split(";"):
                if p and os.path.exists(os.path.join(p, "cl.exe")):
                    cl = os.path.join(p, "cl.exe")
                    break
            if cl is None:
                return (rel, "no cl.exe")
            cmd = [cl, "/std:c++20", "/O2", "/EHsc", "/Zc:__cplusplus", "/utf-8",
                   "/D__HIP_PLATFORM_AMD__", "/DWIN32_LEAN_AND_MEAN", "/DNOMINMAX",
                   "/DUTF8PROC_STATIC"]
            cmd += ["/I" + i[2:] for i in COMMON_INCS]
            cmd += ["/IC:\\Program Files\\AMD\\ROCm\\7.2\\include",
                    "/c", src, "/Fo" + obj]
    rr = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=ENV,
                        timeout=900)
    if rr.returncode != 0:
        errs = [l for l in (rr.stderr or "").splitlines() if "error" in l.lower()][:6]
        return (rel, errs)
    return (rel, None)

def main():
    files = []
    for ext in (".cu", ".cpp"):
        for p in glob.glob(os.path.join(SRC, "**", "*" + ext), recursive=True):
            rel = os.path.relpath(p, SRC)
            if rel.startswith(("ops/linear/gguf/ggml_", "ops\\linear\\gguf\\ggml_")):
                continue  # added below with the HIP includes
            files.append(rel)
    # add ggml kernels explicitly (they are .cu under ops/linear/gguf/)
    files += [os.path.relpath(p, SRC) for p in
              glob.glob(os.path.join(SRC, "ops", "linear", "gguf", "ggml_*.cu"))]
    files = sorted(set(files))
    print(f"total translation units: {len(files)}")
    fails = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(build_one, rel): rel for rel in files}
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            rel, err = fut.result()
            done += 1
            if err:
                fails.append((rel, err))
                print(f"[{done}/{len(files)}] FAIL {rel}")
            else:
                if done % 50 == 0:
                    print(f"[{done}/{len(files)}] ok")
    print(f"compiled {done - len(fails)}/{len(files)}; fails {len(fails)}")
    for rel, err in fails[:15]:
        print(" ", rel)
        for e in err:
            print("     ", e[:200])
    # Proceed even with known-failing TU lists (NVIDIA TMA exclusive paths, one chunked debug
    # kernel) -- their legacy objects remain in build_nordc from earlier builds. The lib is
    # rebuilt from every surviving object.
    if os.path.exists(os.path.join(OUT, "ninfer_hip.lib.pre_rcs.bak")) and len(fails) > 0:
        # Restrict to the authoritative 421-member set so we don't pull in probe/test objects.
        r0 = subprocess.run([ar := os.path.join(ROCM_BIN, "llvm-ar.exe"), "t", os.path.join(OUT, "ninfer_hip.lib.pre_rcs.bak")],
                            capture_output=True, text=True, errors="replace")
        members = [m for m in r0.stdout.split() if m]
        missing = [m for m in members if not os.path.exists(os.path.join(OUT, m))]
        if missing:
            print("missing member objects:", missing[:10])
            return 1
        lib = os.path.join(OUT, "ninfer_hip.lib")
        if os.path.exists(lib):
            os.remove(lib)
        r = subprocess.run([ar, "rcs", lib] + [os.path.join(OUT, m) for m in members],
                           capture_output=True, text=True, errors="replace", env=ENV, timeout=1200)
        if r.returncode != 0:
            print("ar failed:", r.stderr[:1000])
            return 1
        print("lib rebuilt (421 members):", lib, os.path.getsize(lib))
        r = subprocess.run([sys.executable, r"D:\ninfer-hip-migrate\probe_rebuilt\link_exe.py"],
                           capture_output=True, text=True, errors="replace", env=ENV, timeout=1800)
        print(r.stdout[-500:])
        print(r.stderr[-500:] if r.returncode != 0 else "")
        return r.returncode
    # rebuild the lib from every object
    objs = glob.glob(os.path.join(OUT, "*.o")) + glob.glob(os.path.join(OUT, "*.obj"))
    # exclude ggml kernel objects? They are needed. Include everything in build_nordc except
    # mini_main.obj (linked separately) and _*.obj probes.
    objs = [o for o in objs if not os.path.basename(o).startswith("mini_main")]
    ar = os.path.join(ROCM_BIN, "llvm-ar.exe")
    lib = os.path.join(OUT, "ninfer_hip.lib")
    if os.path.exists(lib):
        os.remove(lib)
    r = subprocess.run([ar, "rcs", lib] + objs, capture_output=True, text=True,
                       errors="replace", env=ENV, timeout=1200)
    if r.returncode != 0:
        print("ar failed:", r.stderr[:1000])
        return 1
    print("lib rebuilt:", lib, os.path.getsize(lib))
    # relink
    r = subprocess.run([sys.executable, r"D:\ninfer-hip-migrate\probe_rebuilt\link_exe.py"],
                       capture_output=True, text=True, errors="replace", env=ENV, timeout=1200)
    print(r.stdout[-500:])
    print(r.stderr[-500:] if r.returncode != 0 else "")
    return r.returncode

if __name__ == "__main__":
    sys.exit(main())