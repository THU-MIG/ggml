#include "pad.cuh"

#include <stdint.h>
#include <stdlib.h>

static bool temporal_copy2d_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("ED_CUDA_PAD_TEMPORAL_COPY2D");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

__device__ __forceinline__ int64_t wrap_around(int64_t coord, int64_t size) {
    // + size ensures negatives are handled properly
    return (coord + size) % size;
}

static __global__ void pad_f32(const float * src, size_t s00, size_t s01, size_t s02, size_t s03, float * dst,
                               const int lp0, const int rp0, const int lp1, const int rp1,
                               const int lp2, const int rp2, const int lp3, const int rp3,
                               const int ne0, const int ne1, const int ne2, const int ne3,
                               const int pad_mode) {
    // blockIdx.z: i3*ne2+i2
    // blockIdx.y: i1
    // blockIDx.x: i0 / CUDA_PAD_BLOCK_SIZE
    // gridDim.y:  ne1
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int i1 = blockIdx.y;
    int i2 = blockIdx.z % ne2;
    int i3 = blockIdx.z / ne2;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;

    if (pad_mode == 0) {
        if ((i0 >= lp0 && i0 < ne0 - rp0) && (i1 >= lp1 && i1 < ne1 - rp1) && (i2 >= lp2 && i2 < ne2 - rp2) &&
            (i3 >= lp3 && i3 < ne3 - rp3)) {
            const int64_t i00  = i0 - lp0;
            const int64_t i01  = i1 - lp1;
            const int64_t i02  = i2 - lp2;
            const int64_t i03  = i3 - lp3;

            const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

            dst[dst_idx] = src[src_idx];
        } else {
            dst[dst_idx] = 0.0f;
        }
    }
    // circular means on a torus, so x and y wrap around
    else if (pad_mode == 1) {
        const int64_t ne00 = ne0 - lp0 - rp0;
        const int64_t ne01 = ne1 - lp1 - rp1;
        const int64_t ne02 = ne2 - lp2 - rp2;
        const int64_t ne03 = ne3 - lp3 - rp3;

        const int64_t i00 = wrap_around(i0 - lp0, ne00);
        const int64_t i01 = wrap_around(i1 - lp1, ne01);
        const int64_t i02 = wrap_around(i2 - lp2, ne02);
        const int64_t i03 = wrap_around(i3 - lp3, ne03);

        const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    }
    else {
        const int64_t ne00 = ne0 - lp0 - rp0;
        const int64_t ne01 = ne1 - lp1 - rp1;
        const int64_t ne02 = ne2 - lp2 - rp2;
        const int64_t ne03 = ne3 - lp3 - rp3;
        const int64_t rel0 = i0 - lp0;
        const int64_t rel1 = i1 - lp1;
        const int64_t rel2 = i2 - lp2;
        const int64_t rel3 = i3 - lp3;
        const int64_t i00 = rel0 < 0 ? -rel0 : (rel0 >= ne00 ? 2 * ne00 - 2 - rel0 : rel0);
        const int64_t i01 = rel1 < 0 ? -rel1 : (rel1 >= ne01 ? 2 * ne01 - 2 - rel1 : rel1);
        const int64_t i02 = rel2 < 0 ? -rel2 : (rel2 >= ne02 ? 2 * ne02 - 2 - rel2 : rel2);
        const int64_t i03 = rel3 < 0 ? -rel3 : (rel3 >= ne03 ? 2 * ne03 - 2 - rel3 : rel3);
        const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;
        dst[dst_idx] = src[src_idx];
    }
}


static void pad_f32_cuda(const float * src, size_t s00, size_t s01, size_t s02, size_t s03, float * dst,
    const int lp0, const int rp0, const int lp1, const int rp1,
    const int lp2, const int rp2, const int lp3, const int rp3,
    const int ne0, const int ne1, const int ne2, const int ne3,
    const int pad_mode, cudaStream_t stream) {
    int  num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2 * ne3);
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, s00, s01, s02, s03, dst,
                                                         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                                                         ne0, ne1, ne2, ne3, pad_mode);
}

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t lp0      = ((const int32_t *) (dst->op_params))[0];
    const int32_t rp0      = ((const int32_t *) (dst->op_params))[1];
    const int32_t lp1      = ((const int32_t *) (dst->op_params))[2];
    const int32_t rp1      = ((const int32_t *) (dst->op_params))[3];
    const int32_t lp2      = ((const int32_t *) (dst->op_params))[4];
    const int32_t rp2      = ((const int32_t *) (dst->op_params))[5];
    const int32_t lp3      = ((const int32_t *) (dst->op_params))[6];
    const int32_t rp3      = ((const int32_t *) (dst->op_params))[7];
    const int32_t pad_mode = ((const int32_t *) (dst->op_params))[8];

    const size_t s00 = nb00 / ggml_type_size(src0->type);
    const size_t s01 = nb01 / ggml_type_size(src0->type);
    const size_t s02 = nb02 / ggml_type_size(src0->type);
    const size_t s03 = nb03 / ggml_type_size(src0->type);

    const bool temporal_left_zero_pad =
        pad_mode == 0 &&
        lp0 == 0 && rp0 == 0 && lp1 == 0 && rp1 == 0 &&
        lp2 > 0 && rp2 == 0 && lp3 == 0 && rp3 == 0 &&
        s00 == 1 &&
        s01 == (size_t) src0->ne[0] &&
        s02 == (size_t) src0->ne[0] * src0->ne[1] &&
        s03 == (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] &&
        dst->ne[0] == src0->ne[0] &&
        dst->ne[1] == src0->ne[1] &&
        dst->ne[2] == src0->ne[2] + lp2 &&
        dst->ne[3] == src0->ne[3];
    if (temporal_left_zero_pad && temporal_copy2d_enabled()) {
        const size_t frame_bytes = (size_t) src0->ne[0] * src0->ne[1] * sizeof(float);
        const size_t prefix_bytes = (size_t) lp2 * frame_bytes;
        const size_t src_pitch = (size_t) src0->ne[2] * frame_bytes;
        const size_t dst_pitch = (size_t) dst->ne[2] * frame_bytes;
        CUDA_CHECK(cudaMemset2DAsync(dst_d, dst_pitch, 0, prefix_bytes, (size_t) dst->ne[3], stream));
        CUDA_CHECK(cudaMemcpy2DAsync(reinterpret_cast<char *>(dst_d) + prefix_bytes,
                                     dst_pitch,
                                     src0_d,
                                     src_pitch,
                                     src_pitch,
                                     (size_t) src0->ne[3],
                                     cudaMemcpyDeviceToDevice,
                                     stream));
        return;
    }

    pad_f32_cuda(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                 pad_mode, stream);
}
