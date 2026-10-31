#pragma once
// NInfer -> HIP port shim: MSVC host-compat neutralizers.
// Include BEFORE any ROCm header in shim wrappers. Pure-host MSVC .cpp builds
// include ROCm headers only for declarations; these GNU-only constructs in
// amd_detail headers are never used on the host path.
//   - __attribute__(x): ROCm device decls use const/pure GNU attributes
//   - _Float16: device_library_decls.h declares _Float16-returning __ockl/__ocml
//     device functions; MSVC has no _Float16 and host code never calls them
#if defined(_MSC_VER) && !defined(__clang__)
#ifndef __shim_msvc_host_compat
#define __shim_msvc_host_compat
#include <bit>
#define __attribute__(x)
typedef float _Float16;
#endif
#endif
