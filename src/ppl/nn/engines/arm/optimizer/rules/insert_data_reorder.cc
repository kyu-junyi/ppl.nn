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

#include "ppl/nn/engines/arm/optimizer/rules/insert_data_reorder.h"
#include "ppl/nn/engines/arm/optimizer/rules/utils.h"
#include "ppl/nn/engines/arm/optimizer/opt_kernel.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

RetCode InsertDataReorderRule::AddReorderOp(const OptKernelOptions& options, const bool is_normal_input,
                                        const edgeid_t& edge_id, const nodeid_t& node_id,
                                        const ppl::common::datatype_t reorder_in_type, const ppl::common::dataformat_t reorder_in_format,
                                        const ppl::common::datatype_t reorder_out_type, const ppl::common::dataformat_t reorder_out_format) {
    auto graph_topo = options.graph_topo;
    auto graph_data = options.graph_data;
    auto info = options.info;
    auto& tensors = *options.tensors;
    auto& io = *options.io_info;

    auto edge = graph_topo->GetEdge(edge_id);
    auto node = graph_topo->GetNode(node_id);

    std::string reorder_node_name = "";
    if (is_normal_input) {
        reorder_node_name = "ReorderInput_" + edge->GetName() + "_of_" + node->GetName();
    } else {
        reorder_node_name = "ReorderExtraInput_" + edge->GetName() + "_of_" + node->GetName();
    }

    auto node_ret_pair = graph_topo->AddNode(reorder_node_name);
    if (!node_ret_pair.second) {
        LOG(ERROR) << "node[" << reorder_node_name << "] already exists.";
        return RC_EXISTS;
    }
    ir::Node* reorder_node = node_ret_pair.first;
    reorder_node->SetType(ir::Node::Type("pmx", "Reorder", 1));

    std::string reorder_edge_name = reorder_node_name + "_edge";
    auto edge_ret_pair = graph_topo->AddEdge(reorder_edge_name);
    if (!edge_ret_pair.second) {
        LOG(ERROR) << "edge[" << reorder_edge_name << "] already exists.";
        return RC_EXISTS;
    }

    ir::Edge* reorder_edge = edge_ret_pair.first;
    reorder_node->AddInput(edge_id);
    reorder_node->AddOutput(reorder_edge->GetId());
    reorder_edge->SetProducer(reorder_node->GetId());
    reorder_edge->AddConsumer(node_id);

    edge->DelConsumer(node_id);
    edge->AddConsumer(reorder_node->GetId());
    if (is_normal_input) {
        node->ReplaceInput(edge_id, reorder_edge->GetId());
    } else {
        node->ReplaceExtraInput(edge_id, reorder_edge->GetId());
    }

    auto type = reorder_node->GetType();
    auto creator = OptKernelCreatorManager::GetInstance()->Find(type.domain, type.name, type.version);
    if (!creator) {
        LOG(ERROR) << "cannot find creator for ArmOptKernel[" << reorder_node->GetName() << "] type[" << type.domain
                   << ":" << type.name << "]";
        return RC_NOT_FOUND;
    }

    auto opt_kernel = std::unique_ptr<ArmOptKernel>((*creator)(reorder_node));
    if (!opt_kernel) {
        LOG(ERROR) << "create ArmOptKernel failed: oom";
        return RC_OUT_OF_MEMORY;
    }

    auto status = opt_kernel->Init(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "Init for kernel[" << opt_kernel->GetNode()->GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }
    opt_kernel->SetInputDataLayout(0, reorder_in_type, reorder_in_format);
    opt_kernel->SetOutputDataLayout(0, reorder_out_type, reorder_out_format);
    status = opt_kernel->SelectAlgoDTypeDFormat(options);
    if (status != RC_SUCCESS) {
        return status;
    }
    info->kernels.emplace(reorder_node->GetId(), std::move(opt_kernel));

    TensorImpl* tensor = new TensorImpl(reorder_edge, TENSORTYPE_NORMAL);
    tensor->GetShape()->SetDataFormat(reorder_out_format);
    tensor->GetShape()->SetDataType(reorder_out_type);

    tensors.emplace(reorder_edge->GetId(), std::unique_ptr<TensorImpl>(tensor));


// #ifdef PPLNN_ENABLE_KERNEL_PROFILING
    LOG(INFO) << "Successfully added reorder op " << reorder_node_name << ". [" << GetDataFormatStr(reorder_in_format)
              << ", " << GetDataTypeStr(reorder_in_type) << "] --> [" << GetDataFormatStr(reorder_out_format) << ", "
              << GetDataTypeStr(reorder_out_type) << "]";
// #endif
    return RC_SUCCESS;
}

bool InsertDataReorderRule::Apply(const OptKernelOptions& options) {

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

        auto num_inputs = node->GetInputCount();
        auto &input_dformats = optor->GetInputDataFormats();
        auto &input_dtypes = optor->GetInputDataTypes();

        for (auto idx = 0; idx < num_inputs; idx++) {
            auto input_edge_id = node->GetInput(idx);

            if (graph_data->constants.find(input_edge_id) != graph_data->constants.end()) {
                continue; // skip constants
            }

            auto *input_tensor = io.GetInput<TensorImpl>(idx);

            auto cur_dformat = input_tensor->GetShape()->GetDataFormat();
            auto cur_dtype   = input_tensor->GetShape()->GetDataType();

            auto req_dformat = input_dformats[idx];
            auto req_dtype   = input_dtypes[idx];

            if (cur_dformat != req_dformat || cur_dtype != req_dtype) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
                LOG(INFO) << "Reorder " << input_tensor->GetName()
                          << " from (" << GetDataTypeStr(cur_dtype) << " , " << GetDataFormatStr(cur_dformat) << ")"
                          << " to (" << GetDataTypeStr(req_dtype) << " , " << GetDataFormatStr(req_dformat) << ")";
#else 
                LOG(DEBUG) << "Reorder " << input_tensor->GetName()
                           << " from (" << GetDataTypeStr(cur_dtype) << " , " << GetDataFormatStr(cur_dformat) << ")"
                           << " to (" << GetDataTypeStr(req_dtype) << " , " << GetDataFormatStr(req_dformat) << ")";
#endif
                // if (IsGraphInput(graph_topo, graph_topo->GetEdge(input_edge_id))) {
                //     input_tensor->GetShape()->SetDataType(req_dtype);
                //     input_tensor->GetShape()->SetDataFormat(req_dformat);
                //     LOG(WARNING) << "Use data converter";
                //     continue;
                // }
                auto status = AddReorderOp(options, true, input_edge_id, node_id, cur_dtype, cur_dformat, req_dtype, req_dformat);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "Cannot reorder " << input_tensor->GetName()
                               << " from (" << GetDataTypeStr(cur_dtype) << " , " << GetDataFormatStr(cur_dformat) << ")"
                               << " to (" << GetDataTypeStr(req_dtype) << " , " << GetDataFormatStr(req_dformat) << ")";
                    status_ = status;
                    return false;
                }
            }
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
