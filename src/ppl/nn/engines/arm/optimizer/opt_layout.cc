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

#include "ppl/nn/engines/arm/optimizer/opt_layout.h"
#include "ppl/kernel/arm_server/reorder/neon/reorder.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

    DECLARE_TYPE(f32,   DATATYPE_FLOAT32);
    DECLARE_FORMAT(nda, DATAFORMAT_NDARRAY);
    DECLARE_FORMAT(c4,  DATAFORMAT_N4CX);

#ifdef PPLNN_USE_ARMV8_2_FP16
    DECLARE_TYPE(f16,  DATATYPE_FLOAT16);
    DECLARE_FORMAT(c8, DATAFORMAT_N8CX);

    DECLARE_LAYOUT_PAIR(f16,  c8,  f16,  nda);  // trans
    DECLARE_LAYOUT_PAIR(f32,  nda, f16,  nda);  // cast
    DECLARE_LAYOUT_PAIR(f32,  nda, f16,  c8);   // reorder
    DECLARE_LAYOUT_PAIR(f32,  c4,  f16,  c8);   // reorder

    DECLARE_LAYOUT_PAIR(f32,  c4,  f32,  nda);  // trans
#endif

#define LAYOUT_CONVERSION_WRAPPER(srcT, dstT, convert_func) \
    [](const void* src, const int64_t shape[4], void* dst) -> ppl::common::RetCode { \
        return ppl::kernel::arm_server::neon::convert_func((const srcT*)src, shape, (dstT*)dst); \
    }

std::unordered_map<uint64_t, OptLayoutManager::layout_convert_func_t> OptLayoutManager::layout_conversion_table_ = {
    { VAR_LAYOUT_PAIR(f32,  c4,  f32,  nda), LAYOUT_CONVERSION_WRAPPER(float , float , trans_f32c4_to_f32nda) },
    { VAR_LAYOUT_PAIR(f32,  nda, f32,  c4 ), LAYOUT_CONVERSION_WRAPPER(float , float , trans_f32nda_to_f32c4) },

#ifdef PPLNN_USE_ARMV8_2_FP16
    { VAR_LAYOUT_PAIR(f16,  c8,  f16,  nda), LAYOUT_CONVERSION_WRAPPER(__fp16, __fp16, trans_f16c8_to_f16nda) },
    { VAR_LAYOUT_PAIR(f16,  nda, f16,  c8 ), LAYOUT_CONVERSION_WRAPPER(__fp16, __fp16, trans_f16nda_to_f16c8) },

    { VAR_LAYOUT_PAIR(f32,  nda, f16,  nda), LAYOUT_CONVERSION_WRAPPER(float , __fp16, cast_f32_to_f16) },
    { VAR_LAYOUT_PAIR(f16,  nda, f32,  nda), LAYOUT_CONVERSION_WRAPPER(__fp16, float , cast_f16_to_f32) },

    { VAR_LAYOUT_PAIR(f32,  nda, f16,  c8 ), LAYOUT_CONVERSION_WRAPPER(float , __fp16, reorder_f32nda_to_f16c8) },
    { VAR_LAYOUT_PAIR(f16,  c8,  f32,  nda), LAYOUT_CONVERSION_WRAPPER(__fp16, float , reorder_f16c8_to_f32nda) },

    { VAR_LAYOUT_PAIR(f32,  c4,  f16,  c8 ), LAYOUT_CONVERSION_WRAPPER(float , __fp16, reorder_f32c4_to_f16c8) },
    { VAR_LAYOUT_PAIR(f16,  c8,  f32,  c4 ), LAYOUT_CONVERSION_WRAPPER(__fp16, float , reorder_f16c8_to_f32c4) },
#endif
};

#undef LAYOUT_CONVERSION_WRAPPER

bool OptLayoutManager::Check(datatype_t idt, dataformat_t idf, datatype_t odt, dataformat_t odf) {
    auto tag = ((uint64_t)idt << 48) | ((uint64_t)idf << 32) | ((uint64_t)odt << 16) | ((uint64_t)odf);
    return Instance()->layout_conversion_table_.find(tag) != Instance()->layout_conversion_table_.end();
}

RetCode OptLayoutManager::ConvertLayout(const void *src, const int64_t shape[4], datatype_t idt, dataformat_t idf, datatype_t odt, dataformat_t odf, void *dst) {
    auto tag = ((uint64_t)idt << 48) | ((uint64_t)idf << 32) | ((uint64_t)odt << 16) | ((uint64_t)odf);
    auto converter_it = Instance()->layout_conversion_table_.find(tag);
    if (converter_it == Instance()->layout_conversion_table_.end()) {
        return RC_NOT_FOUND;
    }
    auto convert_func = converter_it->second;
    return convert_func(src, shape, dst);
}

void OptLayoutManager::Register(uint64_t tag, OptLayoutManager::layout_convert_func_t f) {
    layout_conversion_table_[tag] = f;
}

}}} // namespace ppl::nn::arm
