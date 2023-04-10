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

#ifndef _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_ARM_OPTIMIZER_OPT_KERNEL_H_

#include "ppl/nn/runtime/opt_kernel.h"

#include <functional>

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/engines/arm/arm_device.h"
#include "ppl/nn/engines/arm/arm_common_param.h"
#include "ppl/nn/engines/arm/utils/macros.h"
#include "ppl/nn/engines/arm/engine_options.h"
#include "ppl/nn/engines/arm/optimizer/opt_layout.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/generated/onnx_op_generated.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/arm/pmx/generated/arm_op_params_generated.h"
#endif

namespace ppl { namespace nn { namespace utils {
struct SharedResource;
}}} // namespace ppl::nn::utils

namespace ppl { namespace nn { namespace arm {

struct OptKernelOptions final {
    const utils::SharedResource* resource = nullptr;
    ir::GraphData* graph_data = nullptr;
    ir::GraphTopo* graph_topo = nullptr;
    ArmDevice* device = nullptr;
    RuntimePartitionInfo* info = nullptr;
    std::map<edgeid_t, std::unique_ptr<TensorImpl>>* tensors = nullptr;
    EngineOptions* engine_options = nullptr;
    InputOutputInfo* io_info = nullptr;
};

class ArmOptKernel : public OptKernel {
public:
    ArmOptKernel(const ir::Node* node);
    virtual ~ArmOptKernel() {}

    virtual ppl::common::RetCode Init(const OptKernelOptions&) = 0;

    ppl::common::RetCode InferDims(InputOutputInfo* info) const {
        if (infer_dims_func_) {
            return infer_dims_func_(info);
        }
        return ppl::common::RC_NOT_FOUND;
    }

    virtual ppl::common::RetCode SelectAlgoDTypeDFormat(const OptKernelOptions options) {
        return ppl::common::RC_NOT_FOUND;
    }

    virtual ppl::common::RetCode OmitConstantsData(std::map<edgeid_t, int64_t>* constants_data_refcount) {
        return ppl::common::RC_SUCCESS;
    }

    virtual ppl::common::RetCode ConvertConstants(const OptKernelOptions options) {
        return ppl::common::RC_SUCCESS;
    }

    const std::vector<ppl::common::dataformat_t>& GetInputDataFormats() {
        return common_param_.input_formats;
    }

    const std::vector<ppl::common::datatype_t>& GetInputDataTypes() {
        return common_param_.input_types;
    }

    void SetInputDataLayout(uint32_t idx, ppl::common::datatype_t type, ppl::common::dataformat_t format) {
        common_param_.input_types[idx] = type;
        common_param_.input_formats[idx] = format;
    }

    void SetOutputDataLayout(uint32_t idx, ppl::common::datatype_t type, ppl::common::dataformat_t format) {
        common_param_.output_types[idx] = type;
        common_param_.output_formats[idx] = format;
    }

    virtual void SetAllocator(ppl::common::Allocator*) { }

protected:
    template <typename T>
    ppl::common::RetCode GenericLoadParam(const OptKernelOptions& options, std::shared_ptr<T>* param) const {
        auto node = GetNode();
        auto graph_data = options.graph_data;

        auto param_ref = graph_data->attrs.find(node->GetId());
        if (param_ref == graph_data->attrs.end()) {
            return ppl::common::RC_NOT_FOUND;
        }

        *param = std::static_pointer_cast<T>(param_ref->second);
        return ppl::common::RC_SUCCESS;
    }

    template <typename KernelType, typename ParamType>
    KernelType* CreateKernelImplWithParam(const ParamType* param) const {
        auto kernel = new KernelType(GetNode());
        kernel->SetParam(param);
        kernel->SetCommonParam(&common_param_);
        kernel->SetReshapeFunc([this](InputOutputInfo* info) -> ppl::common::RetCode {
            auto input0_type = info->GetInput<TensorImpl>(0)->GetShape()->GetDataType();
            for (uint32_t i = 0; i < info->GetOutputCount(); i++) {
                if (i < common_param_.output_types.size()) {
                    info->GetOutput<TensorImpl>(i)->GetShape()->SetDataType(common_param_.output_types[i]);
                } else {
                    info->GetOutput<TensorImpl>(i)->GetShape()->SetDataType(input0_type);
                }
            }
            infer_layout_func_(info);
            for (uint32_t i = 0; i < info->GetOutputCount(); i++) {
                if (i < common_param_.output_types.size()) {
                    info->GetOutput<TensorImpl>(i)->GetShape()->SetDataFormat(common_param_.output_formats[i]);
                } else {
                    info->GetOutput<TensorImpl>(i)->GetShape()->SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
                }
            }
            return infer_dims_func_(info);
        });
        return kernel;
    }

    template <typename KernelType>
    KernelType* CreateKernelImplWithoutParam() const {
        auto kernel = new KernelType(GetNode());
        kernel->SetCommonParam(&common_param_);
        kernel->SetReshapeFunc([this](InputOutputInfo* info) -> ppl::common::RetCode {
            auto input0_type = info->GetInput<TensorImpl>(0)->GetShape()->GetDataType();
            for (uint32_t i = 0; i < info->GetOutputCount(); i++) {
                if (i < common_param_.output_types.size()) {
                    info->GetOutput<TensorImpl>(i)->GetShape()->SetDataType(common_param_.output_types[i]);
                } else {
                    info->GetOutput<TensorImpl>(i)->GetShape()->SetDataType(input0_type);
                }
            }
            infer_layout_func_(info);
            for (uint32_t i = 0; i < info->GetOutputCount(); i++) {
                if (i < common_param_.output_types.size()) {
                    info->GetOutput<TensorImpl>(i)->GetShape()->SetDataFormat(common_param_.output_formats[i]);
                } else {
                    info->GetOutput<TensorImpl>(i)->GetShape()->SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
                }
            }
            auto status = infer_dims_func_(info);
            return status;
        });
        return kernel;
    }

    static ppl::common::RetCode GenericInferDims(InputOutputInfo* info) {
        auto& in_shape0 = *info->GetInput<TensorImpl>(0)->GetShape();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto& out_shape = *info->GetOutput<TensorImpl>(i)->GetShape();
            if (in_shape0.IsScalar()) {
                out_shape.ReshapeAsScalar();
            } else {
                out_shape.Reshape(in_shape0.GetDims(), in_shape0.GetDimCount());
            }
        }
        return ppl::common::RC_SUCCESS;
    }

    static void GenericSelectInputLayout(InputOutputInfo* info, ArmCommonParam& common_param) {
        const uint32_t num_inputs = info->GetInputCount();
        for (uint32_t i = 0; i < num_inputs; ++i) {
            auto& in_shape = *info->GetInput<TensorImpl>(i)->GetShape();
            common_param.input_types[i] = in_shape.GetDataType();
            common_param.input_formats[i] = in_shape.GetDataFormat();
        }
    }

    static ppl::common::RetCode ArithmeticSelectTwoInputsLayout(const std::string& op_name, InputOutputInfo* info, ArmCommonParam &common_param);
    
    static ppl::common::RetCode RelationSelectTwoInputsLayout(const std::string& op_name, InputOutputInfo* info, ArmCommonParam &common_param);

    static ppl::common::RetCode ReduceSelectLayout(const std::string& op_name, InputOutputInfo* info,
                                                   ppl::common::datatype_t major_float_type, bool keep_dims, const std::vector<int32_t>& axes,
                                                   ArmCommonParam &common_param);

    static void GenericSelectOutputLayout(InputOutputInfo* info, ArmCommonParam& common_param) {
        ppl::common::datatype_t in0_type = ppl::common::DATATYPE_FLOAT32;
        ppl::common::dataformat_t in0_format = ppl::common::DATAFORMAT_NDARRAY;
        if (common_param.input_types.size() > 0) {
            in0_type = common_param.input_types[0];
            in0_format = common_param.input_formats[0];
        }

        const uint32_t num_outputs = info->GetOutputCount();
        for (uint32_t i = 0; i < num_outputs; ++i) {
            common_param.output_types[i] = in0_type;
            common_param.output_formats[i] = in0_format;
        }
    }

    static void StaticInferLayout(InputOutputInfo* info) {
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            out_shape->SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
            out_shape->SetDataType(ppl::common::DATATYPE_FLOAT32);
        }
    }

    static void GenericInferLayout(InputOutputInfo* info) {
        auto& in_shape0 = *info->GetInput<TensorImpl>(0)->GetShape();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            out_shape->SetDataFormat(in_shape0.GetDataFormat());
            out_shape->SetDataType(in_shape0.GetDataType());
        }
    }

    static void DynamicInferLayout(InputOutputInfo* info, const ArmCommonParam* common_param = nullptr,
                                   std::function<ppl::common::dataformat_t(InputOutputInfo*, uint32_t)> dynamic_format_func = [](InputOutputInfo*, uint32_t)
                                        {return ppl::common::DATAFORMAT_NDARRAY;},
                                   std::function<ppl::common::datatype_t(InputOutputInfo*, uint32_t)> dynamic_type_func = [](InputOutputInfo*, uint32_t)
                                        {return ppl::common::DATATYPE_FLOAT32;}
                                   ) {
        if (common_param) {
            for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
                auto out_shape = info->GetOutput<TensorImpl>(i)->GetShape();
                out_shape->SetDataFormat( (i < common_param->output_formats.size()) ? common_param->output_formats[i] : dynamic_format_func(info, i) );
                out_shape->SetDataType( (i < common_param->output_types.size()) ? common_param->output_types[i] : dynamic_type_func(info, i) );
            }
        } else {
            for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
                auto out_shape = info->GetOutput<TensorImpl>(i)->GetShape();
                out_shape->SetDataFormat( dynamic_format_func(info, i) );
                out_shape->SetDataType( dynamic_type_func(info, i) );
            }
        }
    }

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode SerializeData(const pmx::SerializationContext&, utils::DataStream* ds) const override {
        return ppl::common::RC_SUCCESS;
    }

    ppl::common::RetCode DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t) override {
        return ppl::common::RC_SUCCESS;
    }
#endif

protected:
    std::function<void(InputOutputInfo*)> infer_layout_func_;
    std::function<ppl::common::RetCode(InputOutputInfo*)> infer_dims_func_;
public:
    ArmCommonParam common_param_;
};

}}} // namespace ppl::nn::arm

#endif
