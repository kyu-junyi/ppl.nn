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

#include "ppl/nn/engines/arm/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/arm/kernels/onnx/gemm_kernel.h"
#include "ppl/nn/engines/arm/kernels/onnx/fc_kernel.h"
#include "ppl/nn/engines/arm/utils/data_trans.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/oputils/onnx/reshape_gemm.h"
#include "ppl/nn/common/logger.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/oputils/onnx/gemm.h"
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

GemmOp::GemmOp(const ir::Node* node) : ArmOptKernel(node), fc_param_(nullptr) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (fc_param_) {
            if (info->GetInputCount() < 2) {
                LOG(DEBUG) << "ERROR: input count[" << info->GetInputCount() << "] < 2.";
                return RC_INVALID_VALUE;
            }

            auto A = info->GetInput<TensorImpl>(0)->GetShape();
            auto Y = info->GetOutput<TensorImpl>(0)->GetShape();

            int32_t AMdim = 0;
            if (param_->transA) {
                AMdim = 1;
            }

            Y->Reshape({A->GetDim(AMdim), fc_param_->param.num_output});
            return RC_SUCCESS;
        } else {
            return onnx::ReshapeGemm(info, param_.get());
        }  
    };

    infer_layout_func_ = GenericInferLayout;
}

GemmOp::~GemmOp() {
    if (fc_param_ != nullptr) {
        if (fc_param_->mgr != nullptr) {
            fc_param_->mgr->release_cvt_weights();
            delete fc_param_->mgr;
        }
        delete fc_param_;
    }
}

RetCode GemmOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode GemmOp::SelectAlgoDTypeDFormat(const OptKernelOptions options) {
    auto node = GetNode();
    auto graph_data = options.graph_data;

    auto input_type = options.io_info->GetInput<TensorImpl>(0)->GetShape()->GetDataType();
    if (!CheckMajorFloat_(input_type)) {
        LOG(ERROR) << "Unsupported datatype for Gemm Op: " << GetDataTypeStr(input_type);
        return RC_UNSUPPORTED;
    }
    for (uint32_t i = 0; i < options.io_info->GetInputCount(); i++) {
        common_param_.input_types[i] = input_type;
        common_param_.input_formats[i] = DATAFORMAT_NDARRAY;
    }

    auto weight_data_it = graph_data->constants.find(node->GetInput(1));
    void* weight_data = nullptr;
    if (weight_data_it != graph_data->constants.end()) {
        weight_data = weight_data_it->second.data.GetData();
    }

    void* bias_data = nullptr;
    int64_t bias_len = 0;
    if (node->GetInputCount() == 3) {
        auto bias_data_it = graph_data->constants.find(node->GetInput(2));
        if (bias_data_it != graph_data->constants.end()) {
            bias_len = bias_data_it->second.data.GetSize() / sizeof(float);
            bias_data = bias_data_it->second.data.GetData();
        }
    }

    if ((param_->transA || !param_->transB || weight_data == nullptr || (node->GetInputCount() == 3 && bias_data == nullptr))) {
        LOG(DEBUG) << "Gemm Op is not suitable for the fully-connected kernel.";
        return RC_SUCCESS;
    }

    if (!fc_param_) {
        fc_param_ = new FCParam;
    }
    if (!fc_param_) {
        return ppl::common::RC_OUT_OF_MEMORY;
    }

    const ir::Shape& weight_shape = graph_data->shapes.find(node->GetInput(1))->second;
    fc_param_->param.num_output = weight_shape.dims[0];
    fc_param_->param.channels = weight_shape.dims[1];
    fc_param_->param.fuse_flag = 0;

    fc_param_->algo_info = ppl::kernel::arm_server::neon::fc_algo_selector::select_algo(
        DATAFORMAT_NDARRAY, fc_param_->param, input_type, options.device->GetISA());

    if (fc_param_->algo_info.algo_type == ppl::kernel::arm_server::neon::fc_algo::unknown) {
        LOG(INFO) << "FC select algorithm failed, use fallback kernel";
    } else {
        fc_param_->mgr = ppl::kernel::arm_server::neon::fc_algo_selector::gen_algo(
            fc_param_->param, fc_param_->algo_info, options.device->GetAllocator());
    }

    GenericSelectInputLayout(options.io_info, common_param_);
    common_param_.input_formats[0] = DATAFORMAT_NDARRAY;

    GenericSelectOutputLayout(options.io_info, common_param_);
    return RC_SUCCESS;
}

RetCode GemmOp::ConvertConstants(const OptKernelOptions options) {
    if (!fc_param_) {
        return RC_SUCCESS;
    }

    auto node = GetNode();
    auto graph_data = options.graph_data;

    auto weight_data_it = graph_data->constants.find(node->GetInput(1));
    int64_t weight_len = weight_data_it->second.data.GetSize() / sizeof(float);
    void* weight_data = weight_data_it->second.data.GetData();

    void* bias_data = nullptr;
    int64_t bias_len = 0;
    if (node->GetInputCount() == 3) {
        auto bias_data_it = graph_data->constants.find(node->GetInput(2));
        bias_len = bias_data_it->second.data.GetSize() / sizeof(float);
        bias_data = bias_data_it->second.data.GetData();
    }

    // Note: If the filter is reused, check it in generate_cvt_weights and simply reuse it as they are the same.
    //!CAVEAT: change this if new algorithms with different cvt_filter are added.
    ppl::nn::TensorBufferInfo * new_filter = &options.info->constants[node->GetInput(1)];
    new_filter->SetDevice(options.device);

    // Note: If the bias is reused, check it in generate_cvt_weights and simply reuse it as they are the same.
    //!CAVEAT: change this if new algorithms with different cvt_bias are added.
    ppl::nn::TensorBufferInfo * new_bias = nullptr;
    if (bias_data != nullptr) {
        new_bias = &options.info->constants[node->GetInput(2)];
        new_bias->SetDevice(options.device);
    }

    auto dtype = fc_param_->algo_info.dtype;
    if (dtype == ppl::common::DATATYPE_FLOAT32) {
        ppl::common::TensorShape cvt_filter_shape, cvt_bias_shape;
        fc_param_->mgr->generate_cvt_weights_shapes(cvt_filter_shape, cvt_bias_shape, dtype);

        if (new_bias && new_bias->IsBufferOwner() && new_bias->GetBufferPtr()) {
            bias_data = nullptr;
        } else if (bias_data && new_bias) {
            new_bias->Reshape(cvt_bias_shape);
            new_bias->ReallocBuffer();
        }

        new_filter->Reshape(cvt_filter_shape);
        new_filter->ReallocBuffer();

        void *cvt_filter_buffer = (new_filter) ? new_filter->GetBufferPtr() : nullptr;
        void *cvt_bias_buffer   = (new_bias  ) ? new_bias->GetBufferPtr()   : nullptr;

        fc_param_->mgr->generate_cvt_weights(cvt_filter_buffer, cvt_bias_buffer, weight_data, bias_data, dtype);
#ifdef PPLNN_USE_ARMV8_2_FP16
    } else if (dtype == ppl::common::DATATYPE_FLOAT16) {
        ppl::common::TensorShape cvt_filter_shape, cvt_bias_shape;
        fc_param_->mgr->generate_cvt_weights_shapes(cvt_filter_shape, cvt_bias_shape, dtype);

        vector<__fp16> weight_data_fp16;
        weight_data_fp16.resize(weight_len * sizeof(__fp16));
        Fp32ToFp16((const float*)weight_data, weight_len, weight_data_fp16.data());

        vector<__fp16> bias_data_fp16;
        __fp16 * origin_bias_data_fp16 = nullptr;
        if (bias_data != nullptr) {
            bias_data_fp16.resize(bias_len * sizeof(__fp16));
            Fp32ToFp16((const float*)bias_data, bias_len, bias_data_fp16.data());
            origin_bias_data_fp16 = bias_data_fp16.data();
        }

        if (new_bias && new_bias->IsBufferOwner() && new_bias->GetBufferPtr()) {
            origin_bias_data_fp16 = nullptr;
        } else if (bias_data && new_bias) {
            new_bias->Reshape(cvt_bias_shape);
            new_bias->ReallocBuffer();
        }

        new_filter->Reshape(cvt_filter_shape);
        new_filter->ReallocBuffer();

        void *cvt_filter_buffer = (new_filter) ? new_filter->GetBufferPtr() : nullptr;
        void *cvt_bias_buffer   = (new_bias  ) ? new_bias->GetBufferPtr()   : nullptr;

        fc_param_->mgr->generate_cvt_weights(cvt_filter_buffer, cvt_bias_buffer, weight_data_fp16.data(), origin_bias_data_fp16, dtype);
#endif
    }

    return RC_SUCCESS;
}

bool GemmOp::TryFuseReLU() {
    gemm_fuse_relu_ = true;
    if (fc_param_ && fc_param_->algo_info.algo_type != ppl::kernel::arm_server::neon::fc_algo::unknown) {
        ppl::kernel::arm_server::neon::fc_param param = fc_param_->mgr->param();
        param.fuse_flag |= ppl::kernel::arm_server::neon::fc_fuse_flag::relu;
        fc_param_->mgr->set_param(param);
    }
    return true;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode GemmOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    if (!fc_param_) { 
        flatbuffers::FlatBufferBuilder builder;
        auto fb_fusion_data = ppl::nn::pmx::arm::CreateFusionDataDirect(builder, (gemm_fuse_relu_ ? 1 : 0), &common_param_.output_types, &common_param_.output_formats);
        auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(builder, ppl::nn::pmx::arm::PrivateDataType_FusionData, fb_fusion_data.Union());
        ppl::nn::pmx::arm::FinishOpDataBuffer(builder, fb_op_data);

        flatbuffers::FlatBufferBuilder op_builder;
        auto fb_param = ppl::nn::pmx::onnx::SerializeGemmParam(*param_.get(), &op_builder);
        auto fb_data = op_builder.CreateVector(builder.GetBufferPointer(), builder.GetSize());
        auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_GemmParam, fb_param.Union(), fb_data);
        ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
        return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
    }

    flatbuffers::FlatBufferBuilder fc_internal_builder;
    auto algo_info = fc_param_->algo_info;
    auto fb_algo_info = ppl::nn::pmx::arm::CreateFCAlgoInfo(fc_internal_builder,
                                                            algo_info.algo_type,
                                                            algo_info.dtype,
                                                            common_param_.output_formats[0],
                                                            algo_info.isa);
    auto fb_param_info = ppl::nn::pmx::arm::CreateFCParamInfo(fc_internal_builder, 
                                                              fc_param_->param.num_output, 
                                                              fc_param_->param.channels, 
                                                              fc_param_->param.fuse_flag,
                                                              fc_param_->mgr->has_bias_term() ? 1 : 0);
    auto fb_fc_data = ppl::nn::pmx::arm::CreateFullConnectData(fc_internal_builder, fb_algo_info, fb_param_info);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(fc_internal_builder, ppl::nn::pmx::arm::PrivateDataType_FullConnectData, fb_fc_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(fc_internal_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_param = ppl::nn::pmx::onnx::SerializeGemmParam(*param_.get(), &op_builder);
    auto fb_data = op_builder.CreateVector(fc_internal_builder.GetBufferPointer(), fc_internal_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_GemmParam, fb_param.Union(), fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
}

ppl::common::RetCode GemmOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);

    auto fb_gemm_param = fb_op_param->value_as_GemmParam();
    if (!fb_gemm_param) {
        return ppl::common::RC_INVALID_VALUE;
    }
    param_ = std::make_shared<ppl::nn::onnx::GemmParam>();
    DeserializeGemmParam(*fb_gemm_param, param_.get());

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    
    auto arm_fc_data = arm_op_data->value_as_FullConnectData();
    if (arm_fc_data) {
        if (fc_param_) {
            delete fc_param_;
        }
        fc_param_ = new FCParam;
        if (!fc_param_) {
            return ppl::common::RC_OUT_OF_MEMORY;
        }

        auto algo_info = arm_fc_data->algo_info();
        auto param_info = arm_fc_data->param_info();

        fc_param_->algo_info.algo_type = algo_info->algo_type();
        fc_param_->algo_info.dtype = algo_info->dtype();
        fc_param_->algo_info.isa = algo_info->isa();

        common_param_.output_types.resize(1);
        common_param_.output_formats.resize(1);
        common_param_.output_types[0] = algo_info->dtype();
        common_param_.output_formats[0] = algo_info->dformat();

        fc_param_->param.channels = param_info->channels();
        fc_param_->param.num_output = param_info->num_output();
        fc_param_->param.fuse_flag = param_info->fuse_flag();

        auto mgr = new ppl::kernel::arm_server::neon::fc_manager(fc_param_->param, allocator_);

        const auto & shapes = *ctx.shapes;
        const auto & constants = *ctx.constants;
        
        uint32_t cvt_filter_id = GetNode()->GetInput(1);
        uint64_t cvt_filter_size = shapes.at(cvt_filter_id).CalcBytesExcludingPadding();
        void *cvt_filter_ptr = constants.at(cvt_filter_id).GetBufferPtr<void>();
        mgr->set_cvt_filter(cvt_filter_ptr, cvt_filter_size);

        bool has_bias = (param_info->bias_term() == 1);
        void *cvt_bias_ptr;
        if (has_bias) {
            uint32_t cvt_bias_id = GetNode()->GetInput(2);
            uint64_t cvt_bias_size = shapes.at(cvt_bias_id).CalcBytesExcludingPadding();
            cvt_bias_ptr = constants.at(cvt_bias_id).GetBufferPtr<void>();
            mgr->set_cvt_bias(cvt_bias_ptr, cvt_bias_size);
        } else {
            uint64_t cvt_bias_size = param_info->num_output() * ppl::common::GetSizeOfDataType(algo_info->dtype());
            cvt_bias_size = (cvt_bias_size + 15) & (~15);
            cvt_bias_ptr = mgr->allocator()->Alloc(cvt_bias_size);
            memset(cvt_bias_ptr, 0, cvt_bias_size);
            mgr->set_cvt_bias(cvt_bias_ptr, cvt_bias_size, true);
        }

        fc_param_->mgr = mgr;
        return RC_SUCCESS;
    }
    else {
        auto arm_fusion_data = arm_op_data->value_as_FusionData();
        if (arm_fusion_data) {
            gemm_fuse_relu_ = (arm_fusion_data->fuse_relu() == 1);
            ppl::nn::pmx::utils::Fbvec2Stdvec(arm_fusion_data->dtype(), &common_param_.output_types);
            ppl::nn::pmx::utils::Fbvec2Stdvec(arm_fusion_data->dformat(), &common_param_.output_formats);
        }
        return RC_SUCCESS;
    }

    return RC_INVALID_VALUE;
}

#endif

KernelImpl* GemmOp::CreateKernelImpl() const {
    if (fc_param_ && fc_param_->algo_info.algo_type != ppl::kernel::arm_server::neon::fc_algo::unknown) {
        return CreateKernelImplWithParam<FCKernel>(fc_param_);
    } else {
        auto kernel = CreateKernelImplWithParam<GemmKernel>(param_.get());
        kernel->SetFuseReLU(gemm_fuse_relu_);
        return kernel;
    }
}

}}} // namespace ppl::nn::arm
