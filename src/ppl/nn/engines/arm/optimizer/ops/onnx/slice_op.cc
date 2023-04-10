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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/slice_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/slice_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_slice.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

SliceOp::SliceOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        // check parameters
        if (info->GetInputCount() < 3 || info->GetInputCount() > 5 || info->GetOutputCount() != 1) {
            return RC_INVALID_VALUE;
        }
        for (size_t i = 1; i < info->GetInputCount(); i++) {
            if (info->GetInput<TensorImpl>(i)->GetShape()->GetDimCount() !=
                1) { // starts, ends, axes, steps must be 1-D tensor
                return RC_INVALID_VALUE;
            }
        }
        const uint32_t axes_num = info->GetInput<TensorImpl>(1)->GetShape()->GetDim(0);
        for (size_t i = 2; i < info->GetInputCount(); i++) {
            if (info->GetInput<TensorImpl>(i)->GetShape()->GetDim(0) !=
                axes_num) { // starts, end, axes, steps must have same length except for not defined
                return RC_INVALID_VALUE;
            }
        }
        // check support
        for (size_t i = 1; i < info->GetInputCount(); i++) {
            if (info->GetInput<TensorImpl>(i)->GetShape()->GetDataType() != DATATYPE_INT64) {
                LOG(ERROR) << "starts, ends, axes & steps only support int64 now.";
                return RC_UNSUPPORTED;
            }
        }
        for (size_t i = 0; i < info->GetInputCount(); i++) {
            if (info->GetInput<TensorImpl>(i)->GetShape()->GetDataFormat() != DATAFORMAT_NDARRAY) {
                LOG(ERROR) << "unsupported data format: "
                           << GetDataFormatStr(info->GetInput<TensorImpl>(i)->GetShape()->GetDataFormat());
                return RC_UNSUPPORTED;
            }
        }

        return onnx::ReshapeSlice(info);
    };

    infer_layout_func_ = GenericInferLayout;
}

RetCode SliceOp::Init(const OptKernelOptions& options) {
    return RC_SUCCESS;
}

RetCode SliceOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    GenericSelectInputLayout(options.io_info, common_param_);
    common_param_.input_formats[0] = DATAFORMAT_NDARRAY;
    for (int64_t i = 1; i < options.io_info->GetInputCount(); i++) {
        if (!CheckDTypes<DATATYPE_INT64, DATATYPE_INT32>(common_param_.input_types[i])) {
            LOG(ERROR) << "Unsupported input[" << i << "] type for Split Op: " << GetDataTypeStr(common_param_.input_types[i]);
        }
        common_param_.input_formats[i] = DATAFORMAT_NDARRAY;
    }

    GenericSelectOutputLayout(options.io_info, common_param_);
    return RC_SUCCESS;
}

KernelImpl* SliceOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<SliceKernel>();
}

}}} // namespace ppl::nn::arm
