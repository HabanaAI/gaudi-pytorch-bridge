/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include "hpu_ops/stack_getter.h"

namespace sh = synapse_helpers;

namespace habana {

class OptimizerFusedResourceApplyMomentumOperator : public OpBackend {
 public:
  OptimizerFusedResourceApplyMomentumOperator(
      int device_id,
      c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            NO_TPC + "optimizer_fused_ResourceApplyMomentumOperator_",
            scalar_type,
            {},
            {0}, // inplace ids
            {},
            false) {}

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

void OptimizerFusedResourceApplyMomentumOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(
      this, stack, "OptimizerFusedResourceApplyMomentumOperator::AddNode");
  auto params_momentum_buf_list =
      stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto dp_list = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto momentum = stackGetter.getNextInput<double>();
  auto dtype = params_momentum_buf_list[0].pt_t.scalar_type();

  if (params_momentum_buf_list.size() != 2 * dp_list.size()) {
    std::stringstream ss;
    ss << "params_momentum_buf_list must have twice as many elements as dp_list but they respectively have: "
       << params_momentum_buf_list.size() << ", " << dp_list.size();
    AT_ERROR(ss.str());
  }

  std::string add_node = get_guid_with_precision("add_fwd", dtype);
  std::string sub_node = get_guid_with_precision("sub_fwd", dtype);
  std::string mul_node = get_guid_with_precision("mult_fwd", dtype);

  int64_t scalar_shape[] = {1};
  auto momentum_t =
      ConstantHelper(graph, static_cast<double>(momentum), dtype, scalar_shape);

  size_t vec_size = dp_list.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto i2 = 2 * i;
    const auto i2p1 = i2 + 1;

    const auto& param = params_momentum_buf_list[i2];
    const auto& momentum_buffer = params_momentum_buf_list[i2p1];
    const auto& dp = dp_list[i];

    const auto& outshape = param.pt_t.sizes();

    auto mul = BuildOp(
        graph,
        mul_node,
        {momentum_buffer.syn_t, momentum_t.get()},
        {{outshape, dtype}});

    auto sub =
        BuildOp(graph, sub_node, {mul[0].get(), dp.syn_t}, {{outshape, dtype}});

    auto sub_out = IdentityHelper(graph, sub[0].get(), outshape, dtype, i2p1);

    auto add = BuildOp(
        graph, add_node, {param.syn_t, sub[0].get()}, {{outshape, dtype, i2}});

    syn_out(i2) = std::move(add[0]);
    syn_out(i2p1) = std::move(sub_out);
  }
}

} // namespace habana

static auto& OptimizerKernelsKernelRegistry = habana::KernelRegistry().add(
    "hpu::optimizer_resource_apply_momentum",
    KERNEL_FN(OptimizerFusedResourceApplyMomentumOperator));
