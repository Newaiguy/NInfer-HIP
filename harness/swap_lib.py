#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Swap one recompiled translation unit into build_nordc/ninfer_hip.lib (D-drive copy)."""
import os
import subprocess
import sys

ROCM_BIN = r"C:\Program Files\AMD\ROCm\7.2\bin"
AR = os.path.join(ROCM_BIN, "llvm-ar.exe")
OUT = r"D:\ninfer-hip-migrate\build_nordc"
LIB = os.path.join(OUT, "ninfer_hip.lib")
SRC = r"D:\ninfer-hip-migrate\ninfer\src"


def main():
    rel = sys.argv[1].replace("/", os.sep)
    stem = rel.replace("\\", "_")
    if rel.endswith(".cu"):
        obj = os.path.join(OUT, stem + ".o")
        canonical = stem + ".o"
    else:
        obj = os.path.join(OUT, stem + ".obj")
        canonical = stem + ".obj"
    if not os.path.exists(obj):
        print("ERROR: missing object", obj)
        return 1

    members = subprocess.run([AR, "t", LIB], capture_output=True, text=True,
                             errors="replace").stdout.split()
    doomed = [m for m in members if m.startswith(stem) and m != canonical]
    if rel.endswith(".cu"):
        # Legacy .cu members omit the .cu suffix (ggml_bridge.o) while the fresh one keeps
        # it (ggml_bridge.cu.o); remove the stale legacy member too. Match the legacy name
        # EXACTLY: a prefix match here would also delete sibling translation units, because
        # small_t.cu's legacy stem ("..._small_t") is a prefix of small_t_fp8.cu.o,
        # small_t_k8v4.cu.o and small_t_nvfp4.cu.o.
        legacy_stem = stem[:-3]
        legacy_names = {legacy_stem + ".o", legacy_stem + ".obj"}
        doomed += [m for m in members if m in legacy_names and m != canonical]
    if doomed:
        print("removing stale members:", doomed)
        for m in doomed:
            subprocess.run([AR, "d", LIB, m], check=True)
    r = subprocess.run([AR, "r", LIB, obj], capture_output=True, text=True, errors="replace")
    print(f"swapped {canonical}")
    if r.returncode != 0:
        print(r.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())