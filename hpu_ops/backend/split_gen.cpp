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

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace habana {
sizes_vec SplitHpuOutputShape(const at::Stack& stack) {
  auto self = stack[0].toTensor();
  auto split_size = stack[1].toInt();
  auto dim = stack[2].toInt();
  // convert dim to positive value if required
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  auto orig_shape = self.sizes().vec();
  auto split_count = orig_shape[dim] / split_size;
  auto split_remainder = orig_shape[dim] - split_size * split_count;
  auto split_shape = orig_shape;
  split_shape[dim] = split_size;
  sizes_vec out_shapes(split_count, split_shape);
  if (split_remainder > 0) {
    split_shape[dim] = split_remainder;
    out_shapes.push_back(split_shape);
  }

  return out_shapes;
}

class SplitHpu : public OpBackend {
 public:
  SplitHpu(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "split", scalar_type, {0}, {}, {}, false) {
    SetComputeOutputShapes(SplitHpuOutputShape);
  }

  void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override;
};

void SplitHpu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim = stack.at(2).toInt();

  sizes_vec split_output_shapes;
  split_output_shapes = SplitHpuOutputShape(stack);

  std::vector<NodeAttr::NodeOutputAttr> node_output_attrs;
  node_output_attrs.reserve(split_output_shapes.size());
  for (unsigned i = 0; i < split_output_shapes.size(); ++i) {
    node_output_attrs.push_back(
        NodeAttr::NodeOutputAttr{split_output_shapes[i], ScalarType(), i});
  }

  synSplitParams params;
  params.axis = get_dim_in_tpc_order(dim, self.dim());

  auto split_op = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      std::move(node_output_attrs),
      &params,
      sizeof(params));

  for (unsigned i = 0; i < split_op.size(); ++i) {
    syn_out(i) = std::move(split_op[i]);
  }
}

} // namespace habana

static auto& SplitKernelRegistry =
    habana::KernelRegistry().add("aten::split.Tensor", KERNEL_FN(SplitHpu));
