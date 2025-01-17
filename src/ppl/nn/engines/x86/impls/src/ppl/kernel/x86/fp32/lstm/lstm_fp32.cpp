// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <math.h>
#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/lstm.h"
#include "ppl/kernel/x86/fp32/gemm.h"

namespace ppl { namespace kernel { namespace x86 {

static inline float sigmoidf(const float x)
{
    return 1.0f / (1.0f + expf(-x));
}

uint64_t lstm_fp32_get_buffer_bytes(
    const ppl::nn::TensorShape *X_shape,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    const bool has_sequence_lens,
    const bool has_Y,
    const bool has_Y_h,
    const bool has_Y_c)
{
    if (!has_Y && !has_Y_h && !has_Y_c)
        return 64u;

    const int64_t seq_len       = X_shape->GetDim(0);
    const int64_t batch         = X_shape->GetDim(1);
    const int64_t input_size    = X_shape->GetDim(1);
    const int64_t num_direction = direction == rnn_direction::BIDIRECTIONAL ? 2 : 1;
    const bool    has_reverse   = direction == rnn_direction::BIDIRECTIONAL || direction == rnn_direction::REVERSE;

    const uint64_t gate_buff_size = seq_len * batch * rnn_num_gate::LSTM * hidden_size;
    const uint64_t yh_size        = has_Y_h ? num_direction * batch * hidden_size : 0;
    const uint64_t yc_size        = has_Y_c ? num_direction * batch * hidden_size : 0;
    const uint64_t rev_seq_size   = has_sequence_lens && has_reverse ? seq_len * batch * input_size : 0;

    return (gate_buff_size + yh_size + yc_size + rev_seq_size) * sizeof(float);
}

ppl::common::RetCode lstm_fp32_ref(
    const ppl::nn::TensorShape *X_shape,
    const float *X,
    const float **X_weight,
    const float **R_weight,
    const float *P_weight,
    const float *bias,
    const int32_t *sequence_lens,
    const float *initial_h,
    const float *initial_c,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    bool has_packed_w,
    bool has_packed_r,
    void *temp_buffer,
    float *Y,
    float *Y_h,
    float *Y_c)
{
    if (!Y && !Y_h && !Y_c) {
        return ppl::common::RC_SUCCESS;
    }

    const int64_t num_direction = direction == rnn_direction::BIDIRECTIONAL ? 2 : 1;
    const bool    has_reverse   = direction == rnn_direction::BIDIRECTIONAL || direction == rnn_direction::REVERSE;
    const int64_t seq_len       = X_shape->GetDim(0);
    const int64_t batch         = X_shape->GetDim(1);
    const int64_t input_size    = X_shape->GetDim(2);

    float *Yh_buf = Y_h;
    float *Yc_buf = Y_c;
    float *rX = nullptr;

    // set temp buffer
    float *temp_buffer_fp32 = reinterpret_cast<float *>(temp_buffer);
    if (!Yh_buf) {
        Yh_buf = temp_buffer_fp32;
        temp_buffer_fp32 += num_direction * batch * hidden_size;
    }
    if (!Yc_buf) {
        Yc_buf = temp_buffer_fp32;
        temp_buffer_fp32 += num_direction * batch * hidden_size;
    }
    if (sequence_lens && has_reverse) {
        // need reverse X if seq_len of each batch is different
        rX = temp_buffer_fp32;
        temp_buffer_fp32 += seq_len * batch * input_size;
    }
    float *gate_buf = temp_buffer_fp32;

    for (int64_t nd = 0; nd < num_direction; ++nd) {
        const bool is_reverse = nd || (direction == rnn_direction::REVERSE);

        float *nd_Yh = Yh_buf + nd * batch * hidden_size;
        float *nd_Yc = Yc_buf + nd * batch * hidden_size;
        float *nd_Y  = Y + nd * batch * hidden_size;

        // const float *nd_W = X_weight + nd * rnn_num_gate::LSTM * hidden_size * input_size;
        // const float *nd_R = R_weight + nd * rnn_num_gate::LSTM * hidden_size * hidden_size;
        const float *nd_W = X_weight[nd];
        const float *nd_R = R_weight[nd];

        const float *nd_P  = P_weight ? P_weight + nd * (rnn_num_gate::LSTM - 1) * hidden_size : nullptr;
        const float *nd_Wb = bias ? bias + nd * 2 * rnn_num_gate::LSTM * hidden_size : nullptr;
        const float *nd_Rb = bias ? nd_Wb + rnn_num_gate::LSTM * hidden_size : nullptr;

        const float *nd_init_h = initial_h ? initial_h + nd * batch * hidden_size : nullptr;
        const float *nd_init_c = initial_c ? initial_c + nd * batch * hidden_size : nullptr;

        const float *nd_X = X;
        if (sequence_lens && is_reverse) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t work = 0; work < seq_len * batch; ++work) {
                const int64_t seq_idx = work / batch;
                const int64_t b       = work % batch;
                const int64_t seq_end = sequence_lens[b];
                auto src = X + ((seq_idx < seq_end) ? (seq_end - seq_idx - 1) : seq_idx) * batch * input_size + b * input_size;
                auto dst = rX + seq_idx * batch * input_size + b * input_size;
                memcpy(dst, src, input_size * sizeof(float));
            }
            nd_X = rX;
        }

        gemm_fp32_ref( // X[nd]*W[nd]_{iofc}^T+Wb_{iofc}
            nd_X,
            nd_W,
            nd_Wb,
            nullptr,
            gemm_m_type::NOTRANS,
            has_packed_w ? gemm_m_type::PACKED : gemm_m_type::TRANS,
            nd_Wb ? gemm_v_type::ROW_VEC : gemm_v_type::EMPTY,
            gemm_m_type::EMPTY,
            seq_len * batch,
            rnn_num_gate::LSTM * hidden_size,
            input_size,
            input_size,
            input_size,
            rnn_num_gate::LSTM * hidden_size,
            0,
            1.0f,
            0.0f,
            1.0f,
            0.0f,
            gemm_post::NONE,
            gate_buf);

        for (int64_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            // X (seq_len, batch, input_size)
            // gate_buf (seq_len, batch, 4xhidden_size)
            // h_0 (num_direction, batch, hidden_size)
            // c_0 (num_direction, batch, hidden_size)
            // Y (seq_len, num_direction, batch, hidden_size)
            // h_n (num_direction, batch, hidden_size)
            // c_n (num_direction, batch, hidden_size)

            auto seq_gate = gate_buf + ((!sequence_lens && is_reverse) ? (seq_len - seq_idx - 1) : seq_idx) * batch * rnn_num_gate::LSTM * hidden_size;
            auto Y_h_prev = seq_idx == 0 ? nd_init_h : nd_Yh;
            auto Y_c_prev = seq_idx == 0 ? nd_init_c : nd_Yc;

            gemm_fp32_ref( // h[nd]_{t-1}*R[nd]_{iofc}^T+Rb_{iofc}
                Y_h_prev,
                nd_R,
                nd_Rb,
                nullptr,
                gemm_m_type::NOTRANS,
                has_packed_r ? gemm_m_type::PACKED : gemm_m_type::TRANS,
                nd_Rb ? gemm_v_type::ROW_VEC : gemm_v_type::EMPTY,
                gemm_m_type::EMPTY,
                batch,
                rnn_num_gate::LSTM * hidden_size,
                hidden_size,
                hidden_size,
                hidden_size,
                rnn_num_gate::LSTM * hidden_size,
                rnn_num_gate::LSTM * hidden_size,
                !Y_h_prev ? 0.0f : 1.0f, // some hack, gemm will skip aAxB if alpha is 0
                1.0f,
                1.0f,
                0.0f,
                gemm_post::NONE,
                seq_gate);

            if (seq_idx == 0 && !Y_h_prev) {
                Y_h_prev = nd_Yh; // preprocess Y_h_prev
                memset(nd_Yh, 0, batch * hidden_size * sizeof(float));
            }
            if (seq_idx == 0 && !Y_c_prev) {
                Y_c_prev = nd_Yc; // preprocess Y_c_prev
                memset(nd_Yc, 0, batch * hidden_size * sizeof(float));
            }

            if (!P_weight) {
                PRAGMA_OMP_PARALLEL_FOR()
                for (int64_t b = 0; b < batch; ++b) {
                    const int64_t seq_end = sequence_lens ? sequence_lens[b] : seq_len;
                    if (seq_idx < seq_end) {
                        const float *gI    = seq_gate + b * rnn_num_gate::LSTM * hidden_size;
                        const float *gO    = gI + hidden_size;
                        const float *gF    = gO + hidden_size;
                        const float *gC    = gF + hidden_size;
                        const float *Cprev = Y_c_prev + b * hidden_size;
                        float *Ct          = nd_Yc + b * hidden_size;
                        float *Ht          = nd_Yh + b * hidden_size;
                        float *Yt          = nd_Y + (is_reverse ? (seq_end - seq_idx - 1) : seq_idx) * num_direction * batch * hidden_size + b * hidden_size;
                        for (int64_t h = 0; h < hidden_size; ++h) {
                            const float it = sigmoidf(gI[h]);
                            const float ft = sigmoidf(gF[h]);
                            const float ct = ::tanhf(gC[h]);
                            Ct[h]          = ft * Cprev[h] + it * ct;
                            const float ot = sigmoidf(gO[h]);
                            Ht[h]          = ot * ::tanhf(Ct[h]);
                        }
                        if (Y) memcpy(Yt, Ht, hidden_size * sizeof(float));
                    } else { // pass through the initial_h, initial_c
                        const float *Hprev = Y_h_prev + b * hidden_size;
                        const float *Cprev = Y_c_prev + b * hidden_size;
                        float *Ct          = nd_Yc + b * hidden_size;
                        float *Ht          = nd_Yh + b * hidden_size;
                        float *Yt          = nd_Y + seq_idx * num_direction * batch * hidden_size + b * hidden_size;
                        if (Cprev != Ct) memcpy(Ct, Cprev, hidden_size * sizeof(float));
                        if (Hprev != Ht) memcpy(Ht, Hprev, hidden_size * sizeof(float));
                        if (Y) memset(Yt, 0, hidden_size * sizeof(float));
                    }
                }
            } else {
                const float *pI = nd_P;
                const float *pO = pI + hidden_size;
                const float *pF = pO + hidden_size;
                PRAGMA_OMP_PARALLEL_FOR()
                for (int64_t b = 0; b < batch; ++b) {
                    const int64_t seq_end = sequence_lens ? sequence_lens[b] : seq_len;
                    if (seq_idx < seq_end) {
                        const float *gI    = seq_gate + b * rnn_num_gate::LSTM * hidden_size;
                        const float *gO    = gI + hidden_size;
                        const float *gF    = gO + hidden_size;
                        const float *gC    = gF + hidden_size;
                        const float *Cprev = Y_c_prev + b * hidden_size;
                        float *Ct          = nd_Yc + b * hidden_size;
                        float *Ht          = nd_Yh + b * hidden_size;
                        float *Yt          = nd_Y + (is_reverse ? (seq_end - seq_idx - 1) : seq_idx) * num_direction * batch * hidden_size + b * hidden_size;
                        for (int64_t h = 0; h < hidden_size; ++h) {
                            const float it = sigmoidf(gI[h] + pI[h] * Cprev[h]);
                            const float ft = sigmoidf(gF[h] + pF[h] * Cprev[h]);
                            const float ct = ::tanhf(gC[h]);
                            Ct[h]          = ft * Cprev[h] + it * ct;
                            const float ot = sigmoidf(gO[h] + pO[h] * Ct[h]);
                            Ht[h]          = ot * ::tanhf(Ct[h]);
                        }
                        if (Y) memcpy(Yt, Ht, hidden_size * sizeof(float));
                    } else { // pass through the initial_h, initial_c
                        const float *Hprev = Y_h_prev + b * hidden_size;
                        const float *Cprev = Y_c_prev + b * hidden_size;
                        float *Ct          = nd_Yc + b * hidden_size;
                        float *Ht          = nd_Yh + b * hidden_size;
                        float *Yt          = nd_Y + seq_idx * num_direction * batch * hidden_size + b * hidden_size;
                        if (Cprev != Ct) memcpy(Ct, Cprev, hidden_size * sizeof(float));
                        if (Hprev != Ht) memcpy(Ht, Hprev, hidden_size * sizeof(float));
                        if (Y) memset(Yt, 0, hidden_size * sizeof(float));
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode lstm_fp32(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *X_shape,
    const float *X,
    const float **X_weight,
    const float **R_weight,
    const float *P_weight,
    const float *bias,
    const int32_t *sequence_lens,
    const float *initial_h,
    const float *initial_c,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    bool has_packed_w,
    bool has_packed_r,
    void *temp_buffer,
    float *Y,
    float *Y_h,
    float *Y_c)
{
#ifdef PPL_USE_X86_AVX512
    if (isa & ppl::common::ISA_X86_AVX512) {
        return kernel::x86::lstm_fp32_avx512(X_shape, X, X_weight, R_weight, P_weight, bias, sequence_lens, initial_h, initial_c, direction, hidden_size, has_packed_w, has_packed_r, temp_buffer, Y, Y_h, Y_c);
    }
#endif
    if (isa & ppl::common::ISA_X86_FMA) {
        return kernel::x86::lstm_fp32_fma(X_shape, X, X_weight, R_weight, P_weight, bias, sequence_lens, initial_h, initial_c, direction, hidden_size, has_packed_w, has_packed_r, temp_buffer, Y, Y_h, Y_c);
    }
    return kernel::x86::lstm_fp32_ref(X_shape, X, X_weight, R_weight, P_weight, bias, sequence_lens, initial_h, initial_c, direction, hidden_size, has_packed_w, has_packed_r, temp_buffer, Y, Y_h, Y_c);
}

}}}; // namespace ppl::kernel::x86
