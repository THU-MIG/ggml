#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <cstdlib>
#include <cstring>
#include <inttypes.h>

static bool ggml_cuda_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool ggml_cuda_env_flag_enabled_or_default(const char * name, bool default_enabled) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_enabled;
    }
    return value[0] != '0';
}

static bool ggml_cuda_env_step_list_contains(const char * name, const int step) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    const char * cursor = value;
    while (*cursor != '\0') {
        char * end = nullptr;
        long parsed = std::strtol(cursor, &end, 10);
        if (end != cursor && parsed == step) {
            return true;
        }
        cursor = end != cursor ? end : cursor + 1;
        while (*cursor == ',' || *cursor == ';' || *cursor == ' ') {
            ++cursor;
        }
    }
    return false;
}

static bool ggml_cuda_env_list_present(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

static int ggml_cuda_h3_layer_from_name(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return -1;
    }
    const char * blocks = std::strstr(name, "blocks.");
    if (blocks == nullptr) {
        return -1;
    }
    const char * cursor = blocks + 7;
    char * end = nullptr;
    const long layer = std::strtol(cursor, &end, 10);
    if (end == cursor || layer < 0 || layer > INT_MAX) {
        return -1;
    }
    return static_cast<int>(layer);
}

static bool ggml_cuda_h3_main_projection(const char * name, const char * suffix) {
    const char * current_step = std::getenv("ED_MINIMAX_H3_CURRENT_STEP");
    if (current_step == nullptr || current_step[0] == '\0' ||
        name == nullptr || suffix == nullptr ||
        std::strstr(name, "model.diffusion_model.blocks.") == nullptr) {
        return false;
    }
    const size_t name_length = std::strlen(name);
    const size_t suffix_length = std::strlen(suffix);
    return name_length >= suffix_length &&
           std::strcmp(name + name_length - suffix_length, suffix) == 0;
}

static bool ggml_cuda_h3_scoped_enabled(const char * env_name, const char * dst_name) {
    if (ggml_cuda_env_flag_enabled(env_name)) {
        return true;
    }

    const std::string step_list_name = std::string(env_name) + "_STEPS";
    const std::string layer_list_name = std::string(env_name) + "_LAYERS";
    const bool has_step_scope = ggml_cuda_env_list_present(step_list_name.c_str());
    const bool has_layer_scope = ggml_cuda_env_list_present(layer_list_name.c_str());
    if (!has_step_scope && !has_layer_scope) {
        return false;
    }

    if (has_step_scope) {
        const char * current_step = std::getenv("ED_MINIMAX_H3_CURRENT_STEP");
        if (current_step == nullptr || current_step[0] == '\0') {
            return false;
        }
        char * end = nullptr;
        const long step = std::strtol(current_step, &end, 10);
        if (end == current_step || step < 0 || step > INT_MAX ||
            !ggml_cuda_env_step_list_contains(step_list_name.c_str(), static_cast<int>(step))) {
            return false;
        }
    }

    if (has_layer_scope) {
        const int layer = ggml_cuda_h3_layer_from_name(dst_name);
        if (layer < 0 || !ggml_cuda_env_step_list_contains(layer_list_name.c_str(), layer)) {
            return false;
        }
    }

    return true;
}

static int64_t ggml_cuda_h3_q4k_cublas_min_n() {
    const char * value = std::getenv("ED_MINIMAX_H3_Q4K_CUBLAS_MIN_N");
    if (value == nullptr || value[0] == '\0') {
        return 15000;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && parsed > 0 ? parsed : 15000;
}

static bool ggml_cuda_mmq_phase_profile_enabled() {
    static const bool enabled = ggml_cuda_env_flag_enabled("ED_PROFILE_MMQ_PHASES");
    return enabled;
}

static cudaEvent_t ggml_cuda_mmq_phase_profile_start(cudaStream_t stream) {
    if (!ggml_cuda_mmq_phase_profile_enabled()) {
        return nullptr;
    }
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDefault));
    CUDA_CHECK(cudaEventRecord(event, stream));
    return event;
}

static void ggml_cuda_mmq_phase_profile_stop(cudaEvent_t start, cudaStream_t stream, const char * phase,
                                             const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst,
                                             int64_t m, int64_t n, int64_t k) {
    if (start == nullptr) {
        return;
    }
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&stop, cudaEventDefault));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    GGML_LOG_INFO("ED_MMQ_PHASE phase=%s elapsed_ms=%.3f m=%" PRId64 " n=%" PRId64 " k=%" PRId64
                  " src0_type=%s src1_type=%s dst_type=%s src0_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]"
                  " src1_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] dst_ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]\n",
                  phase, elapsed_ms, m, n, k,
                  ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(dst->type),
                  src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                  src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                  dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_fp4 = blackwell_mma_available(cc) && (src0->type == GGML_TYPE_MXFP4 || src0->type == GGML_TYPE_NVFP4);

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            cudaEvent_t profile_start = ggml_cuda_mmq_phase_profile_start(stream);
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
            ggml_cuda_mmq_phase_profile_stop(profile_start, stream, "quantize_q8_1", src0, src1, dst, ne01, ne1, ne10);
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :  // block_fp4_mmq holds 256 values
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1};
        cudaEvent_t profile_start = ggml_cuda_mmq_phase_profile_start(stream);
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        ggml_cuda_mmq_phase_profile_stop(profile_start, stream, "mmq_kernel", src0, src1, dst, ne01, ne1, ne10);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    static_assert(QK_K == 8 * QK_MXFP4, "QK_K needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_K * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne12};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, const ggml_tensor * fused_bias, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols};

    cudaEvent_t profile_start = ggml_cuda_mmq_phase_profile_start(stream);
    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    ggml_cuda_mmq_phase_profile_stop(profile_start, stream, "mmq_kernel_prequantized", src0, src1, dst, row_diff, src1_ncols, ne10);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size, fused_bias);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts, int64_t m, int64_t k, const char * dst_name) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
#if !defined(GGML_CUDA_FORCE_MMQ)
        const bool sm90_q4k_cublas_enabled = ggml_cuda_env_flag_enabled_or_default("GGML_CUDA_SM90_Q4K_CUBLAS", true);
        const bool h3_qkv_projection = ggml_cuda_h3_main_projection(dst_name, ".attn.qkv_proj.weight");
        const bool h3_fc1_projection = ggml_cuda_h3_main_projection(dst_name, ".mlp.fc1.weight");
        const bool h3_fc2_projection = ggml_cuda_h3_main_projection(dst_name, ".mlp.fc2.weight");
        const bool h3_mid_seq_range_enabled = ggml_cuda_h3_scoped_enabled("GGML_CUDA_SM90_Q4K_CUBLAS_MID_QKV_FC1_RANGE", dst_name);
        const bool h3_mid_qkv_range_enabled = h3_mid_seq_range_enabled ||
                                              ggml_cuda_h3_scoped_enabled("GGML_CUDA_SM90_Q4K_CUBLAS_MID_QKV_RANGE", dst_name);
        const bool h3_mid_fc1_range_enabled = h3_mid_seq_range_enabled ||
                                              ggml_cuda_h3_scoped_enabled("GGML_CUDA_SM90_Q4K_CUBLAS_MID_FC1_RANGE", dst_name);
        const bool h3_mid_qkv_shape = h3_qkv_projection &&
                                      (ne11 == 7919 || (h3_mid_qkv_range_enabled && ne11 >= 7800 && ne11 <= 8200)) &&
                                      k == 5376 && m == 21504;
        const bool h3_mid_fc1_shape = h3_fc1_projection &&
                                      (ne11 == 7919 || (h3_mid_fc1_range_enabled && ne11 >= 7800 && ne11 <= 8200)) &&
                                      k == 5376 && m == 28672;
        const bool h3_mid_fc2_range_enabled = ggml_cuda_h3_scoped_enabled("GGML_CUDA_SM90_Q4K_CUBLAS_MID_FC2_RANGE", dst_name);
        const bool h3_mid_fc2_shape = h3_fc2_projection &&
                                      (ne11 == 7919 || (h3_mid_fc2_range_enabled && ne11 >= 7800 && ne11 <= 8200)) &&
                                      m == 5376 && (k == 14336 || k == 7168);
        const int64_t h3_long_min_n = ggml_cuda_h3_q4k_cublas_min_n();
        const bool h3_long_qkv_shape = h3_qkv_projection && ne11 >= h3_long_min_n && m == 21504 && k == 5376;
        const bool h3_long_fc1_shape = h3_fc1_projection && ne11 >= h3_long_min_n && m == 28672 && k == 5376;
        const bool h3_long_fc2_shape = h3_fc2_projection && ne11 >= h3_long_min_n && m == 5376 && (k == 14336 || k == 7168);
        const bool h3_q8_cublas_shape = type == GGML_TYPE_Q8_0 && ne11 >= h3_long_min_n &&
                                        ((h3_qkv_projection && m == 21504 && k == 5376) ||
                                         (h3_fc1_projection && m == 28672 && k == 5376));
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= 900 && h3_q8_cublas_shape) {
            return false;
        }
        if (sm90_q4k_cublas_enabled && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= 900 && type == GGML_TYPE_Q4_K &&
            (h3_mid_qkv_shape ||
             h3_mid_fc1_shape ||
             (ggml_cuda_h3_scoped_enabled("GGML_CUDA_SM90_Q4K_CUBLAS_MID_FC2", dst_name) && h3_mid_fc2_shape) ||
             h3_long_qkv_shape ||
             h3_long_fc1_shape ||
             (!ggml_cuda_env_flag_enabled("GGML_CUDA_SM90_Q4K_CUBLAS_DISABLE_LONG_FC2") && h3_long_fc2_shape))) {
            if (ggml_cuda_env_flag_enabled("ED_H3_CUBLAS_ROUTE_DEBUG")) {
                GGML_LOG_INFO("ED_H3_CUBLAS_ROUTE step=%s name=%s ne11=%" PRId64 " m=%" PRId64 " k=%" PRId64
                              " mid_qkv=%d mid_fc1=%d mid_fc2=%d long_qkv=%d long_fc1=%d long_fc2=%d\n",
                              std::getenv("ED_MINIMAX_H3_CURRENT_STEP") != nullptr ? std::getenv("ED_MINIMAX_H3_CURRENT_STEP") : "-",
                              dst_name != nullptr ? dst_name : "", ne11, m, k,
                              h3_mid_qkv_shape, h3_mid_fc1_shape, h3_mid_fc2_shape,
                              h3_long_qkv_shape, h3_long_fc1_shape, h3_long_fc2_shape);
            }
            return false;
        }
#endif
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}
