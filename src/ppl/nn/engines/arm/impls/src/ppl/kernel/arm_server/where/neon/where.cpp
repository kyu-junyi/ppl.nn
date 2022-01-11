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


#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/where/neon/where_common.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT>
ppl::common::RetCode where_common(
    const ppl::nn::TensorShape *cond_shape,
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const uint8_t *cond,
    const eT *src0,
    const eT *src1,
    eT *dst)
{
    const bool is_eltwise = src0_shape->GetElementsExcludingPadding() == dst_shape->GetElementsExcludingPadding() &&
                            src1_shape->GetElementsExcludingPadding() == dst_shape->GetElementsExcludingPadding();

    if (is_eltwise) {
        return where_eltwise_common<eT>(dst_shape, cond, src0, src1, dst);
    }

    const auto data_format = src0_shape->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return where_ndarray_common<eT>(cond_shape, src0_shape, src1_shape, dst_shape, cond, src0, src1, dst);
    }

    return ppl::common::RC_UNSUPPORTED; 
}


ppl::common::RetCode where(
    const ppl::nn::TensorShape *cond_shape,
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const void *cond,
    const void *src0,
    const void *src1,
    void *dst)
{
    const auto data_type = src0_shape->GetDataType();
    switch (data_type)
    {
        case ppl::common::DATATYPE_BOOL:
            return where_common<bool>(cond_shape, src0_shape, src1_shape, dst_shape, (const uint8_t*)cond, (const bool*)src0, (const bool*)src1, (bool*)dst);
        case ppl::common::DATATYPE_FLOAT32:
            return where_common<float>(cond_shape, src0_shape, src1_shape, dst_shape, (const uint8_t*)cond, (const float*)src0, (const float*)src1, (float*)dst);
#ifdef PPL_USE_ARM_SERVER_FP16
        case ppl::common::DATATYPE_FLOAT16:
            return where_common<__fp16>(cond_shape, src0_shape, src1_shape, dst_shape, (const uint8_t*)cond, (const __fp16*)src0, (const __fp16*)src1, (__fp16*)dst);
#endif            
        default: break;
    }
    return ppl::common::RC_UNSUPPORTED; 
}

}}}}; // namespace ppl::kernel::arm_server::neon
