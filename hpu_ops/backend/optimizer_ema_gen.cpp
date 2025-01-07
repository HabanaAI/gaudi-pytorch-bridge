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
#include "hpu_ops/op_backend.h"
#include "hpu_ops/stack_getter.h"
#include "perf_lib_layer_params.h"

namespace sh = synapse_helpers;

namespace habana {

class OptimizerFusedEmaOperator : public OpBackend {
 public:
  OptimizerFusedEmaOperator(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            NO_TPC + "optimizer_fused_EmaOperator_",
            scalar_type,
            {},
            {1}, // inplace ids
            {},
            false) {}

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

void OptimizerFusedEmaOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "OptimizerFusedEmaOperator::AddNode");
  auto model_inputs = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto updated_ema = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto decay = stackGetter.getNextInput<TensorsPair>();

  if (model_inputs.size() != updated_ema.size()) {
    std::stringstream ss;
    ss << "model_inputs and updated_ema vector inputs must have the same number of elements but they respectively have: "
       << model_inputs.size() << ", " << updated_ema.size();
    AT_ERROR(ss.str());
  }

  std::string add_node = get_guid_with_precision("add_fwd", ScalarType());
  std::string sub_node = get_guid_with_precision("sub_fwd", ScalarType());
  std::string mul_node = get_guid_with_precision("mult_fwd", ScalarType());

  int64_t scalar_shape[] = {1};
  auto c_one = ConstantHelper(graph, 1.0f, ScalarType(), scalar_shape);

  const auto dtype = ScalarType();
  const auto& decay_shape = decay.pt_t.sizes();
  std::vector<NodeAttr::NodeOutputAttr> decay_attr = {{decay_shape, dtype}};

  auto one_minus_decay = BuildOp(
      graph, sub_node, {c_one.get(), decay.syn_t}, std::move(decay_attr));

  size_t vec_size = model_inputs.size();
  for (size_t i = 0; i < vec_size; ++i) {
    const auto& input = model_inputs[i];
    const auto& ema = updated_ema[i];

    const auto& outshape = input.pt_t.sizes();
    std::vector<NodeAttr::NodeOutputAttr> out_attr = {{outshape, dtype}};

    auto ema_times_decay =
        BuildOp(graph, mul_node, {ema.syn_t, decay.syn_t}, out_attr);

    auto one_minus_decay_times_input = BuildOp(
        graph, mul_node, {one_minus_decay[0].get(), input.syn_t}, out_attr);

    auto new_ema = BuildOp(
        graph,
        add_node,
        {ema_times_decay[0].get(), one_minus_decay_times_input[0].get()},
        {{outshape, dtype, i}});

    syn_out(i) = std::move(new_ema[0]);
  }
}

} // namespace habana

static auto& OptimizerKernelsKernelRegistry = habana::KernelRegistry().add(
    "hpu::optimizer_ema",
    KERNEL_FN(OptimizerFusedEmaOperator));
