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
#include "ppl/kernel/arm_server/expand/neon/expand_common.h"


namespace ppl { namespace kernel { namespace arm_server { namespace neon {

ppl::common::RetCode expand(
    const ppl::nn::TensorShape *src_shape,
    const ppl::nn::TensorShape *dst_shape,
    const void *src,
    void *dst)
{
    const auto data_type = src_shape->GetDataType();

    switch (data_type)
    {
    case ppl::common::DATATYPE_FLOAT32: return expand_ndarray_common<float>(src_shape, dst_shape, (const float*)src, (float *)dst);
    case ppl::common::DATATYPE_INT64:   return expand_ndarray_common<int64_t>(src_shape, dst_shape, (const int64_t*)src, (int64_t *)dst);
#ifdef PPL_USE_ARM_SERVER_FP16
    case ppl::common::DATATYPE_FLOAT16: return expand_ndarray_common<__fp16>(src_shape, dst_shape, (const __fp16*)src, (__fp16 *)dst);
#endif
    default: break;
    }
    
    return ppl::common::RC_UNSUPPORTED;
}

}}}}; // namespace ppl::kernel::arm_server::neon
