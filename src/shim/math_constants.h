#pragma once
// NInfer -> HIP port shim: math_constants.h
#define CUDART_PI_F 3.14159265358979323846f
#define CUDART_PI 3.14159265358979323846
#define CUDART_INF_F ((float)(__builtin_huge_valf()))
#define CUDART_INF ((double)(__builtin_huge_val()))
#define CUDART_NAN_F ((float)(__builtin_nanf("")))
#define CUDART_NAN ((double)(__builtin_nan("")))
