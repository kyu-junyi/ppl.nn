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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/range_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/range_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_range.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

RangeOp::RangeOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeRange(info, nullptr);
    };

    infer_layout_func_ = GenericInferLayout;
}

RetCode RangeOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

RetCode RangeOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    GenericSelectInputLayout(options.io_info, common_param_);
    datatype_t common_dtype = common_param_.input_types[0];
    for (int i = 1; i < options.io_info->GetInputCount(); i++) {
        common_dtype = (common_dtype == common_param_.input_types[i]) ? common_param_.input_types[0] : DATATYPE_UNKNOWN;
    }

    if (common_dtype == DATATYPE_UNKNOWN) {
        LOG(ERROR) << "Unsupported different input types for Range Op";
        return RC_UNSUPPORTED;
    }
    if (!CheckDTypes<DATATYPE_FLOAT32, DATATYPE_FLOAT16, DATATYPE_INT64>(common_dtype)) {
        LOG(ERROR) << "Unsupported input type for Range Op: " << GetDataTypeStr(common_dtype);
        return RC_UNSUPPORTED;
    }
    // Skip format - scalars to 1-d array

    GenericSelectOutputLayout(options.io_info, common_param_);
    return RC_SUCCESS;
}

KernelImpl* RangeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<RangeKernel>();
}

}}} // namespace ppl::nn::arm
