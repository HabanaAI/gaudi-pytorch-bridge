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

#include "generated/backend/select.h"

namespace {

habana::sizes_vec SelectOutputShape(const at::Stack& stack) {
  auto self = stack[0].toTensor();
  auto dim = stack[1].toInt();
  auto index = stack[2].toInt();

  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  std::vector<int64_t> sizes = self.sizes().vec();

  dim = at::maybe_wrap_dim(dim, ndim);

  if (index < 0) {
    index += sizes[dim];
  }

  auto start_val = index;
  auto end_val = index + 1;

  if (start_val == INT64_MAX) {
    start_val = 0;
  } else if (start_val < 0) {
    start_val = 0;
  } else if (start_val > sizes[dim]) {
    start_val = sizes[dim];
  }

  if (end_val < 0) {
    end_val += sizes[dim];
  }

  if (end_val < start_val) {
    end_val = start_val;
  } else if (end_val > sizes[dim]) {
    end_val = sizes[dim];
  }

  sizes[dim] = end_val - start_val;

  return {sizes};
}
} // namespace

namespace habana {

sizes_vec SelectHpuOutputShape(const at::Stack& stack) {
  auto self = stack[0].toTensor();
  auto dim = stack[1].toInt();
  // convert dim to positive value if required
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  auto shape = self.sizes().vec();
  shape.erase(shape.begin() + dim);

  return {shape};
}

OutputMetaDataVector SelectHpuMeta(const at::Stack& stack) {
  auto self = stack[0].toTensor();
  OutputMetaData meta;
  meta.dtype = self.scalar_type();
  meta.shape = SelectHpuOutputShape(stack)[0];

  return {meta};
}

bool SelectDSSTMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  PT_BRIDGE_DEBUG("SelectDSSTMeta called");
  static_cast<void>(inputs);
  auto t_size = outputs[0].getTensorShape();
  PT_BRIDGE_DEBUG("SelectDSSTMeta ST shape ", t_size);
  habana_helpers::UpdateSTShapeInfo(t_size);

  return true;
}

class SelectHpu : public OpBackend {
 public:
  SelectHpu(int device_id, c10::ScalarType scalar_type)
      : OpBackend(device_id, "select", scalar_type, {0}, {}, {}, false) {
    SetOutputMetaFn(SelectHpuMeta);
    SetSTMetaFn(SelectDSSTMeta);
  }

  void AddNode(synapse_helpers::graph& graph, const at::Stack& stack) override;
};

void SelectHpu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toInt();
  auto index = stack.at(2).toInt();

  dim = at::maybe_wrap_dim(dim, self.dim());
  auto dim_tpc = get_dim_in_tpc_order(dim, self.dim());

  if (index < 0) {
    index += self.size(dim);
  }

  auto start = index;
  int64_t step = 1;
  auto end = start + step;

  std::vector<int64_t> slice_output_shape = SelectOutputShape(stack)[0];
  auto meta = SelectHpuMeta(stack)[0];

  synSliceParamsNDims params;
  std::fill_n(params.axes, HABANA_DIM_MAX, 0);
  std::fill_n(params.starts, HABANA_DIM_MAX, 0);
  std::fill_n(params.ends, HABANA_DIM_MAX, 0);
  std::fill_n(params.steps, HABANA_DIM_MAX, 1);

  params.axes[0] = dim_tpc;
  params.starts[0] = start;
  params.ends[0] = end;
  params.steps[0] = step;

  auto squeezeNeeded = self.dim() >= 2;

  NodeAttr::NodeOutputAttr node_output_attr = {
      slice_output_shape, meta.dtype, squeezeNeeded ? std::optional<int>{} : 0};

  auto slice_op = BuildOp(
      graph, "slice", {syn_in(0)}, {node_output_attr}, &params, sizeof(params));

  if (squeezeNeeded) {
    auto squeeze_output_shape = slice_output_shape;
    squeeze_output_shape.erase(squeeze_output_shape.begin() + dim);

    syn_out(0) = SqueezeHelper(
        graph, slice_op[0].get(), squeeze_output_shape, meta.dtype, dim_tpc, 0);
  } else {
    syn_out(0) = std::move(slice_op[0]);
  }
}
} // namespace habana

static auto& SelectKernelRegistry =
    habana::KernelRegistry().add("aten::select.int", KERNEL_FN(SelectHpu));
