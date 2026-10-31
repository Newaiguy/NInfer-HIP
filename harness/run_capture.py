import os, subprocess, sys
# Like run_in_vcenv.py but forwards BOTH streams (the engine prints its
# '--- finish=' timings and '[simt-splits]' diagnostics to stderr).
ROOT = r"D:\ninfer-hip-migrate"
OUT = os.path.join(ROOT, "build_nordc")
ROCM_BIN = r"C:\Program Files\AMD\ROCm\7.2\bin"
VCVARS = r"E:\Microsoft Visual Studio\2022\BuildTools\VC_Auxiliary\Build\vcvars64.bat".replace("VC_Auxiliary", "VC\\Auxiliary")

def vc_env():
    bat = os.path.join(OUT, "_env.bat")
    with open(bat, "w", newline="\r\n") as f:
        f.write('@echo off\ncall "%s" >nul 2>&1\nset PATH=%%PATH%%;%s\nset\n' % (VCVARS, ROCM_BIN))
    r = subprocess.run(["cmd", "/c", bat], capture_output=True)
    env = {}
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1); env[k.upper()] = v
    return env

if __name__ == "__main__":
    env = vc_env()
    for k in ("SystemRoot", "SYSTEMROOT"):
        env.setdefault(k, os.environ.get("SystemRoot", r"C:\WINDOWS"))
    # TOOLING BUG (2026-10-29): an *exact* path containing a backslash separator sometimes
    # fails to resolve on this box (GetFileAttributesW -> ERROR_FILE_NOT_FOUND) even though the
    # same file resolves via '/' or via a wildcard. Normalising the executable to '/' costs
    # nothing and makes every launcher deterministic.
    if len(sys.argv) > 1:
        sys.argv[1] = sys.argv[1].replace("\\", "/")
    # AMD FIX (2026-10-29): text=True decodes the child's bytes with the ANSI code page
    # (cp936 here), so every byte of UTF-8 Chinese the engine writes came back as mojibake and was
    # re-encoded as UTF-8 into the log. That made valid Chinese prompts/output look corrupted and
    # sent us chasing a non-existent "the corpus is mojibake" theory. Capture raw bytes and decode
    # as UTF-8 explicitly.
    r = subprocess.run(sys.argv[1:], capture_output=True, env=env)
    sys.stdout.write(r.stdout.decode("utf-8", "replace") if r.stdout else "")
    sys.stdout.write(r.stderr.decode("utf-8", "replace") if r.stderr else "")
    sys.stdout.write("\nrc=%d\n" % r.returncode)
