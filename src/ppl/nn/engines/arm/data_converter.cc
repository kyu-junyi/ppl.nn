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

#include "ppl/nn/engines/arm/data_converter.h"
#include <cstring>
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/arm/optimizer/opt_layout.h"
// #include "ppl/kernel/arm_server/reorder/neon/reorder.h"
#include "ppl/kernel/arm_server/common/memory.h"
#include "ppl/kernel/arm_server/cast/neon/cast.h"
using namespace ppl::common;
using namespace ppl::kernel::arm_server;
using namespace ppl::kernel::arm_server::neon;

namespace ppl { namespace nn { namespace arm {

ppl::common::RetCode ArmDataConverter::Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                               const TensorShape& src_desc, const void*, const void*) const {
    const auto src_dtype = src_desc.GetDataType();
    const auto dst_dtype = dst_desc.GetDataType();
    const auto src_dformat = src_desc.GetDataFormat();
    const auto dst_dformat = dst_desc.GetDataFormat();

    LOG(DEBUG) << "ARM Data Converter from data format " << src_dformat << "(" << GetDataFormatStr(src_dformat) << ") to " << dst_dformat << "(" << GetDataFormatStr(dst_dformat) << ")";
    LOG(DEBUG) << "ARM Data Converter from data type " << src_dtype << "(" << GetDataTypeStr(src_dtype) << ") to " << dst_dtype << "(" << GetDataTypeStr(dst_dtype) << ")";

    if (dst_dformat == src_dformat && dst_dtype == src_dtype) {
        auto status = memory_copy(src.addr, src_desc.CalcBytesIncludingPadding(), dst->addr);
        return status;
    }

    const auto num_dims = src_desc.GetDimCount();
    const int64_t shape_desc[4] = {                 src_desc.GetDim(0),    (num_dims > 1) ? src_desc.GetDim(1) : 1,
                                   (num_dims > 2) ? src_desc.GetDim(2) : 1, (num_dims > 3) ? src_desc.GetDim(3) : 1 };
    auto rc = OptLayoutManager::ConvertLayout(src.addr, shape_desc, src_dtype, src_dformat, dst_dtype, dst_dformat, dst->addr);
    if (rc == RC_SUCCESS) {
        return RC_SUCCESS;
    }

    if (dst_dformat == src_dformat && dst_dtype != src_dtype) {
        return cast(&src_desc, &dst_desc, src.addr, dst->addr);
    }

    LOG(ERROR) << "Invalid data type conversion";
    LOG(ERROR) << "ARM Data Converter from data format " << src_dformat << "(" << GetDataFormatStr(src_dformat) << ") to " << dst_dformat << "(" << GetDataFormatStr(dst_dformat) << ")";
    LOG(ERROR) << "ARM Data Converter from data type " << src_dtype << "(" << GetDataTypeStr(src_dtype) << ") to " << dst_dtype << "(" << GetDataTypeStr(dst_dtype) << ")";
    return RC_UNSUPPORTED;
}

ppl::common::RetCode ArmDataConverter::ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                                     const TensorShape& src_desc, const void*) const {
    BufferDesc dst_wrapper(dst);
    return Convert(&dst_wrapper, dst_desc, src, src_desc);
}

ppl::common::RetCode ArmDataConverter::ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                                       const TensorShape& src_desc, const void*) const {
    return Convert(dst, dst_desc, BufferDesc(const_cast<void*>(src)), src_desc);
}

}}} // namespace ppl::nn::arm
