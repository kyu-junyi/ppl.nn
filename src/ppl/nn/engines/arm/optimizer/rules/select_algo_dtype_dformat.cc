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

#include <vector>

#include "ppl/nn/engines/arm/optimizer/rules/select_algo_dtype_dformat.h"
#include "ppl/nn/engines/arm/optimizer/rules/utils.h"
#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

bool SelectAlgoDTypeDFormatRule::Apply(const OptKernelOptions& options) {

    auto graph_topo = options.graph_topo;
    auto graph_data = options.graph_data;
    auto info = options.info;
    auto& tensors = *options.tensors;
    auto& io = *options.io_info;

    std::vector<nodeid_t> sorted_nodes;
    graph_topo->TopologicalSort([&sorted_nodes](nodeid_t nid) -> void {
        sorted_nodes.push_back(nid);
    });

    for (auto node_id : sorted_nodes) {
        auto optor = dynamic_cast<ArmOptKernel*>(info->kernels[node_id].get());
        auto node = optor->GetNode();
        io.SetNode(node);

        // CheckInput
        const int num_inputs = node->GetInputCount();
        if (node->GetType().name == "Reshape") {
            LOG(DEBUG) << node->GetName() << " has " << num_inputs << " inputs";
            for (auto idx = 0; idx < num_inputs; idx++) {
                auto *input_tensor = io.GetInput<TensorImpl>(idx);

                auto cur_dformat = input_tensor->GetShape()->GetDataFormat();
                auto cur_dtype   = input_tensor->GetShape()->GetDataType();

                LOG(DEBUG) << " - Graph input[" << node->GetInput(idx) << "] has " << GetDataTypeStr(cur_dtype);
                LOG(DEBUG) << " - Graph input[" << node->GetInput(idx) << "] has " << GetDataFormatStr(cur_dformat);
            }
        }

        auto status = optor->SelectAlgoDTypeDFormat(options);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "Failed";
            status_ = status;
            return false;
        }

        // ApplyOutput
        auto const &output_dformats = optor->common_param_.output_formats;
        auto const &output_dtypes = optor->common_param_.output_types;
        LOG(DEBUG) << "Selected layout for " << node->GetName();
        
        const int num_outputs = node->GetOutputCount();
        for (auto idx = 0; idx < num_outputs; idx++) {
            auto *output_tensor = io.GetOutput<TensorImpl>(idx);
            LOG(DEBUG) << "\t" << GetDataTypeStr(output_dtypes[idx]) << ", " << GetDataFormatStr(output_dformats[idx]);
            output_tensor->GetShape()->SetDataFormat(output_dformats[idx]);
            output_tensor->GetShape()->SetDataType(output_dtypes[idx]);
        }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
        // LOG(INFO) << ".";
#endif
    }

    status_ = RC_SUCCESS;
    // one-time pass
    return false;
}

}}} // namespace ppl::nn::arm
