// The vector products for one block type, in their own unit so they build in parallel with the others.
#include "ggml_bridge_vec.cuh"

#if !defined(NINFER_VEC_DEBUG_OFF)
__device__ int g_vec_rdbg[16];
__device__ float g_vec_wdbg[32];
#endif

NINFER_GGUF_VECTOR_INSTANCE(GGML_TYPE_IQ4_XS);
