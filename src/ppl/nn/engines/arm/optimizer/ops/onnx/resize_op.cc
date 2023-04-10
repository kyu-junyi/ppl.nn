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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/resize_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/resize_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_resize.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/resize.h"
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ResizeOp::ResizeOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeResize(info, param_.get());
    };

    infer_layout_func_ = GenericInferLayout;
}

RetCode ResizeOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode ResizeOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    auto& in0_shape = *options.io_info->GetInput<TensorImpl>(0)->GetShape();
    common_param_.input_types[0] = in0_shape.GetDataType();
    common_param_.input_formats[0] = in0_shape.GetDataFormat();
    if (common_param_.input_types[0] == DATATYPE_BFLOAT16) {
        common_param_.input_types[0] = options.engine_options->forward_precision;
        common_param_.input_formats[0] = GetMajorFormat_(common_param_.input_types[0], common_param_.input_formats[0]);
    }
    if (!CheckMajorFloat_(common_param_.input_types[0])) {
        LOG(ERROR) << "Unsupported input[0] type for Resize Op: " << GetDataTypeStr(common_param_.input_types[0]);
        return RC_UNSUPPORTED;
    }
    if (param_->mode == param_->RESIZE_MODE_CUBIC) {
        const int64_t channels = in0_shape.GetDimCount() == 4 ? in0_shape.GetDim(1) : 0;
        if (common_param_.input_types[0] == DATATYPE_FLOAT32 && channels >= 2) {
            common_param_.input_formats[0] = DATAFORMAT_N4CX;
        } else if (common_param_.input_types[0] == DATATYPE_FLOAT16 && channels >= 4) {
            common_param_.input_formats[0] = DATAFORMAT_N8CX;
        }
    }
    const int64_t input_count = options.io_info->GetInputCount();
    if (input_count >= 2) {
        // common_param_.input_types[1] = DATATYPE_FLOAT32;
        if (!CheckMajorFloat_(common_param_.input_types[1])) {
            LOG(ERROR) << "Unsupported input[1] type for Resize Op: " << GetDataTypeStr(common_param_.input_types[1]);
        }
        common_param_.input_formats[1] = DATAFORMAT_NDARRAY;
    }
    if (input_count >= 3) {
        // common_param_.input_types[2] = DATATYPE_FLOAT32;
        if (common_param_.input_types[2] != DATATYPE_FLOAT32) {
            LOG(ERROR) << "Unsupported input[2] type for Resize Op: " << GetDataTypeStr(common_param_.input_types[2]);
        }
        common_param_.input_formats[2] = DATAFORMAT_NDARRAY;
    }
    if (input_count >= 4) {
        // common_param_.input_types[3] = DATATYPE_INT64;
        if (common_param_.input_types[3] != DATATYPE_INT64) {
            LOG(ERROR) << "Unsupported input[3] type for Resize Op: " << GetDataTypeStr(common_param_.input_types[3]);
        }
        common_param_.input_formats[3] = DATAFORMAT_NDARRAY;
    }

    GenericSelectOutputLayout(options.io_info, common_param_);
    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode ResizeOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder output_builder;
    auto fb_output_data = ppl::nn::pmx::arm::CreateOutputDataDirect(output_builder, &common_param_.output_types, &common_param_.output_formats);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(output_builder, ppl::nn::pmx::arm::PrivateDataType_OutputData, fb_output_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(output_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_param = ppl::nn::pmx::onnx::SerializeResizeParam(*param_.get(), &op_builder);
    auto fb_data = op_builder.CreateVector(output_builder.GetBufferPointer(), output_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_ResizeParam, fb_param.Union(), fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode ResizeOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);

    param_ = std::make_shared<ppl::nn::onnx::ResizeParam>();
    ppl::nn::pmx::onnx::DeserializeResizeParam(*fb_op_param->value_as_ResizeParam(), param_.get());

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    auto arm_output_data = arm_op_data->value_as_OutputData();
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dformat(), &common_param_.output_formats);
    return RC_SUCCESS;
}

#endif

KernelImpl* ResizeOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<ResizeKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
