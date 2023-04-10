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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/pad_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/pad_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_pad.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/pad.h"
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

PadOp::PadOp(const ir::Node* node) : ArmOptKernel(node) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto ret = onnx::ReshapePad(info, param_.get());
        if (ret != RC_SUCCESS) {
            return ret;
        }
        return RC_SUCCESS;
    };

    infer_layout_func_ = GenericInferLayout;
}

RetCode PadOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode PadOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    GenericSelectInputLayout(options.io_info, common_param_);

    auto selected_dataformat = common_param_.input_formats[0];
    if (selected_dataformat == ppl::common::DATAFORMAT_N4CX ||
        selected_dataformat == ppl::common::DATAFORMAT_N8CX) { // for nbcx pad, if pad on channel dim, will fall back to ndarray implement
        const auto pads_data = options.io_info->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();
        if (pads_data == nullptr) { // pads not sure on compiler time, fall back to ndarray implement
            selected_dataformat = ppl::common::DATAFORMAT_NDARRAY;
        } else {
            const auto start_pads = pads_data;
            const auto end_pads = pads_data + options.io_info->GetInput<TensorImpl>(0)->GetShape()->GetDimCount();
            const int64_t c_dim_idx = 1;
            if (start_pads[c_dim_idx] != 0 || end_pads[c_dim_idx] != 0) {
                selected_dataformat = ppl::common::DATAFORMAT_NDARRAY;
            }
        }
    }
    common_param_.input_formats[0] = selected_dataformat;
    auto num_inputs = options.io_info->GetInputCount();
    if (num_inputs >= 2) {
        // common_param_.input_types[1] = ppl::common::DATATYPE_INT64;
        if (common_param_.input_types[1] != DATATYPE_INT64) {
            LOG(ERROR) << "Unsupported input[1] type for Pad Op: " << GetDataTypeStr(common_param_.input_types[1]);
        }
        common_param_.input_formats[1] = ppl::common::DATAFORMAT_NDARRAY;
    }
    if (num_inputs >= 3) {
        common_param_.input_types[2] = common_param_.input_types[0];
        common_param_.input_formats[2] = ppl::common::DATAFORMAT_NDARRAY;
    }
    if (num_inputs >= 4) {
        // common_param_.input_types[3] = ppl::common::DATATYPE_INT64;
        if (!CheckDTypes<DATATYPE_INT64, DATATYPE_INT32>(common_param_.input_types[3])) {
            LOG(ERROR) << "Unsupported input[3] type for Pad Op: " << GetDataTypeStr(common_param_.input_types[3]);
        }
        common_param_.input_formats[3] = ppl::common::DATAFORMAT_NDARRAY;
    }

    GenericSelectOutputLayout(options.io_info, common_param_);
    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode PadOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    flatbuffers::FlatBufferBuilder output_builder;
    auto fb_output_data = ppl::nn::pmx::arm::CreateOutputDataDirect(output_builder, &common_param_.output_types, &common_param_.output_formats);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(output_builder, ppl::nn::pmx::arm::PrivateDataType_OutputData, fb_output_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(output_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_param = ppl::nn::pmx::onnx::SerializePadParam(*param_.get(), &op_builder);
    auto fb_data = op_builder.CreateVector(output_builder.GetBufferPointer(), output_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_PadParam, fb_param.Union(), fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode PadOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);

    param_ = std::make_shared<ppl::nn::onnx::PadParam>();
    ppl::nn::pmx::onnx::DeserializePadParam(*fb_op_param->value_as_PadParam(), param_.get());

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    auto arm_output_data = arm_op_data->value_as_OutputData();
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_output_data->dformat(), &common_param_.output_formats);
    return RC_SUCCESS;
}

#endif

KernelImpl* PadOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<PadKernel>(param_.get());
}

}}} // namespace ppl::nn::arm
