#include <stdint.h>

template <typename T> __device__ void arange(T *buf, T start, T step, const size_t numel) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        buf[i] = static_cast<T>(i) * step + start;
    }
}

extern "C" __global__ void arange_u8(uint8_t *buf, uint8_t start, uint8_t step,
                                     const size_t numel) {
    arange(buf, start, step, numel);
}
extern "C" __global__ void arange_u32(uint32_t *buf, uint32_t start, uint32_t step,
                                      const size_t numel) {
    arange(buf, start, step, numel);
}
extern "C" __global__ void arange_i64(int64_t *buf, int64_t start, int64_t step,
                                      const size_t numel) {
    arange(buf, start, step, numel);
}
extern "C" __global__ void arange_f32(float *buf, float start, float step, const size_t numel) {
    arange(buf, start, step, numel);
}
extern "C" __global__ void arange_f64(double *buf, double start, double step, const size_t numel) {
    arange(buf, start, step, numel);
}

#ifdef HALF
#include "cuda_fp16.h"
extern "C" __global__ void arange_f16(__half *buf, __half start, __half step, const size_t numel) {
    arange(buf, start, step, numel);
}
#endif

#ifdef BFLOAT
#include "cuda_bf16.h"
extern "C" __global__ void arange_bf16(__nv_bfloat16 *buf, __nv_bfloat16 start, __nv_bfloat16 step,
                                       const size_t numel) {
    arange(buf, start, step, numel);
}
#endif