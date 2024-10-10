/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "generated/backend/select.h"

namespace {

habana::sizes_vec SliceOutputShape(const at::Stack& stack) {
  auto self = stack[0].toTensor();
  auto dim = stack[1].toInt();
  auto index = stack[2].toInt();

  int64_t ndim = self.dim();
  if (ndim == 0) {
    TORCH_CHECK_INDEX(false, "slice() cannot be applied to a 0-dim tensor.");
  }
  std::vector<int64_t> sizes(self.sizes().begin(), self.sizes().end());

  if (index < 0) {
    index += sizes[dim];
  }

  auto start_val = index;
  auto end_val = index + 1;
  auto step = 1;

  dim = at::maybe_wrap_dim(dim, ndim);

  TORCH_CHECK(step > 0, "slice step must be positive");

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
  } else if (end_val > sizes[dim] + 1) {
    end_val = sizes[dim];
  }

  auto len = end_val - start_val;
  sizes[dim] = (len + step - 1) / step; // round-up

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

  if (index < 0) {
    index += self.size(dim);
  }

  auto start = index;
  auto end = index + 1;
  int64_t step = 1;

  std::vector<int64_t> slice_output_shape;
  slice_output_shape = SliceOutputShape(stack)[0];
  auto meta = SelectHpuMeta(stack)[0];
  synSliceParamsNDims params;

  std::fill_n(params.axes, HABANA_DIM_MAX, 0);
  std::fill_n(params.starts, HABANA_DIM_MAX, 0);
  std::fill_n(params.ends, HABANA_DIM_MAX, 0);
  std::fill_n(params.steps, HABANA_DIM_MAX, 1);

  params.axes[0] = get_dim_in_tpc_order(dim, self.dim());
  params.starts[0] = start;
  params.ends[0] = end;
  params.steps[0] = step;

  NodeAttr::NodeOutputAttr node_output_attr = {slice_output_shape, meta.dtype};

  auto slice_op = BuildOp(
      graph, "slice", {syn_in(0)}, {node_output_attr}, &params, sizeof(params));

  auto reshape_output_shape = slice_output_shape;
  auto dim_ = at::maybe_wrap_dim(dim, self.dim());
  reshape_output_shape.erase(reshape_output_shape.begin() + dim_);

  syn_out(0) = OpBackend::BuildReshape(
      this, graph, slice_op[0].get(), reshape_output_shape, meta.dtype, 0);
}
} // namespace habana

static auto& SelectKernelRegistry =
    habana::KernelRegistry().add("aten::select.int", KERNEL_FN(SelectHpu));
