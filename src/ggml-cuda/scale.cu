#include "scale.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

template<typename T>
static __device__ __forceinline__ float scale_to_float(T value);

template<>
__device__ __forceinline__ float scale_to_float<float>(float value) {
    return value;
}

template<>
__device__ __forceinline__ float scale_to_float<half>(half value) {
    return __half2float(value);
}

template<>
__device__ __forceinline__ float scale_to_float<nv_bfloat16>(nv_bfloat16 value) {
    return __bfloat162float(value);
}

template<typename T>
static __device__ __forceinline__ T scale_from_float(float value);

template<>
__device__ __forceinline__ float scale_from_float<float>(float value) {
    return value;
}

template<>
__device__ __forceinline__ half scale_from_float<half>(float value) {
    return __float2half(value);
}

template<>
__device__ __forceinline__ nv_bfloat16 scale_from_float<nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template<typename T>
static __global__ void scale_cuda_kernel(const T * x, T * dst, const float scale, const float bias, const int64_t nelements) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = scale_from_float<T>(scale * scale_to_float<T>(x[i]) + bias);
    }
}

template<typename T>
static void scale_cuda(const T * x, T * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_cuda_kernel<<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == dst->type);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    switch (src0->type) {
        case GGML_TYPE_F32:
            scale_cuda((const float *) src0->data, (float *) dst->data, scale, bias, ggml_nelements(src0), stream);
            break;
        case GGML_TYPE_F16:
            scale_cuda((const half *) src0->data, (half *) dst->data, scale, bias, ggml_nelements(src0), stream);
            break;
        case GGML_TYPE_BF16:
            scale_cuda((const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data, scale, bias, ggml_nelements(src0), stream);
            break;
        default:
            GGML_ABORT("unsupported scale type: %s", ggml_type_name(src0->type));
    }
}
