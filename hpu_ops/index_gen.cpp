/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"

namespace habana {

// brodcast index tensor shape and get the correct shape and size
static std::vector<int64_t> broadcast_size(at::TensorList indices) {
  auto size = indices[0].sizes().vec();
  for (size_t i = 1; i < indices.size(); i++) {
    size = at::infer_size(size, indices[i].sizes());
  }
  return size;
}

sizes_vec IndexOutputShape(const at::Stack& stack, bool) {
  const at::Tensor input = stack_tensor(stack, 0);
  c10::ArrayRef<c10::IValue> indices_in = stack.at(1).toListRef();
  std::vector<at::Tensor> indices_vec_out{};

  std::vector<at::Tensor> indices_vec;
  for (auto input : indices_in) {
    auto o1 = input.toOptional<at::Tensor>();
    if (!(o1.has_value() && !o1->defined())) {
      indices_vec.push_back(o1.value());
    } else {
      HABANA_ASSERT(
          0 &&
          "None is not yet supported for c10::List<c10::optional<Tensor>>");
    }
  }

  /*for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].device().type() != c10::DeviceType::HPU) {
      indices_vec[i] = indices_vec[i].to(c10::kHPU);
    }
  }

  // handle views for tensorlist indices
  at::TensorList indices_in_list(indices_vec);
  indices_vec = habana_lazy::HandleViewsTensorList(indices_in_list);*/

  // for case where indices are Boolean tensor(s), convert these to integer
  // indices using nonzero operator before calling index
  if (indices_vec[0].scalar_type() == c10::ScalarType::Bool) {
    for (size_t i = 0; i < indices_vec.size(); i++) {
      auto list = torch::nonzero_numpy(indices_vec.at(i));
      indices_vec_out.insert(
          indices_vec_out.cend(), list.cbegin(), list.cend());
    }
  }

  at::TensorList indices =
      (indices_vec[0].scalar_type() == c10::ScalarType::Bool) ? indices_vec_out
                                                              : indices_vec;
  sizes_vec shape = std::vector<std::vector<int64_t>>{
      {IndexOperator::compute_output_shape(input, indices)}};
  return shape;
}

template <>
LazyIndex<at::Tensor>::LazyIndex(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  auto& sub_inputs = get_inputs();
  const at::Tensor self = sub_inputs.at(0).toTensor();
  c10::ArrayRef<c10::IValue> indices_in = sub_inputs.at(1).toListRef();

  std::vector<at::Tensor> indices_vec_out{};

  std::vector<at::Tensor> indices_vec;
  for (auto input : indices_in) {
    auto o1 = input.toOptional<at::Tensor>();

    if (!(o1.has_value() && !o1->defined())) {
      indices_vec.push_back(o1.value());
    } else {
      HABANA_ASSERT(
          0 &&
          "None is not yet supported for c10::List<c10::optional<Tensor>>");
    }
  }

  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].device().type() != c10::DeviceType::HPU) {
      indices_vec[i] = indices_vec[i].to(c10::kHPU);
    }
  }

  // handle views for tensorlist indices
  at::TensorList indices_in_list(indices_vec);
  indices_vec = habana_lazy::HandleViewsTensorList(indices_in_list);

  // for case where indices are Boolean tensor(s), convert these to integer
  // indices using nonzero operator before calling index
  if (indices_vec[0].scalar_type() == c10::ScalarType::Bool) {
    for (size_t i = 0; i < indices_vec.size(); i++) {
      auto list = torch::nonzero_numpy(indices_vec.at(i));
      indices_vec_out.insert(
          indices_vec_out.cend(), list.cbegin(), list.cend());
    }
  }

  at::TensorList indices =
      (indices_vec[0].scalar_type() == c10::ScalarType::Bool) ? indices_vec_out
                                                              : indices_vec;

  auto indices_out_vec = habana_lazy::HandleViewsTensorList(indices);
  at::TensorList indices_out_list(indices_out_vec);

  get_inputs().back() = indices_out_list;
}

template <>
at::Tensor LazyIndex<at::Tensor>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return {};
}

void IndexHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  c10::List<at::Tensor> indices = stack.at(1).toTensorList();

  // for this particular indices configuration gather_mxnet throws GC
  // compilation error, therefore use simple gather for now
  if (indices.size() == 1 && indices.get(0).dim() == 1) {
    auto outshape = GatherOperator::compute_output_shape(self, 0, indices[0]);

    int dim = 0;
    bool sparse_grad = false;
    at::Stack stack_ = {
        c10::IValue(self),
        c10::IValue(dim),
        c10::IValue(indices[0]),
        c10::IValue(sparse_grad)};

    // Fill params for gather
    size_t size = 0;
    const auto& gather_params = FillGatherParams(stack_, size);
    auto gatherOp = BuildOp(
        graph,
        "gather_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outshape, ScalarType(), 0}},
        gather_params.get(),
        size);
    syn_out(0) = std::move(gatherOp[0]);
    return;
  }

  auto tensorlist = stack[1].toTensorList().vec();

  auto max_size = broadcast_size(tensorlist);
  auto device_id = this->p_context_->device_id_;
  auto scalar_type = tensorlist[0].scalar_type();

  std::vector<at::Tensor> cat_input;
  auto cat_indices = make_operator<CatOperator>(device_id, scalar_type);

  for (size_t i = 0; i < tensorlist.size(); i++) {
    // broadcast index tensor to largest index tensor size
    auto bcastOp = make_operator<BroadcastOperator>(device_id, scalar_type);
    torch::jit::Stack stack_ = {
        c10::IValue(tensorlist[i]), c10::IValue(max_size), c10::IValue(false)};
    bcastOp->SetSynapseInput(p_context_->syn_inputs_[i + 1]);
    bcastOp->AllocateAndAddSynapseNode(graph, stack_, false);

    std::vector<int64_t> expanded_size{1};
    for (auto s : bcastOp->GetOutputs()[0].sizes()) {
      expanded_size.push_back(s);
    }
    stack_ = {
        c10::IValue(bcastOp->GetOutputs()[0]), c10::IValue(expanded_size)};
    auto ReshapeOp = make_operator<ReshapeOperator>(device_id, scalar_type);
    ReshapeOp->SetSynapseInput(bcastOp->GetSynOutputs()[0]);
    ReshapeOp->AllocateAndAddSynapseNode(graph, stack_, false);

    cat_input.emplace_back(ReshapeOp->GetOutputs()[0]);
    cat_indices->SetSynapseInput(ReshapeOp->GetSynOutputs()[0]);
  }

  torch::jit::Stack stack_ = {c10::IValue(cat_input), c10::IValue(0)};
  cat_indices->AllocateAndAddSynapseNode(graph, stack_, false);

  auto shape = IndexOperator::compute_output_shape(self, tensorlist);

  synapse_helpers::tensor& arg2_syn_tensor =
      std::move(cat_indices->GetSynOutputs()[0]);

  if (self.scalar_type() != c10::ScalarType::Double &&
      self.scalar_type() != c10::ScalarType::Float &&
      self.scalar_type() != c10::ScalarType::BFloat16) {
    // i8/i16/i32 -> f32
    const auto dtype = c10::ScalarType::Float;
    std::string cast_guid = "cast_" +
        habana_helpers::name_suffix_from_type(ScalarType()) + "_to_" +
        habana_helpers::name_suffix_from_type(dtype);
    auto castOp =
        BuildOp(graph, cast_guid, {syn_in(0)}, {{self.sizes().vec(), dtype}});

    auto indexOp = BuildOp(
        graph,
        "gather_nd_mxnet_fwd_" + habana_helpers::name_suffix_from_type(dtype),
        {castOp[0].get(), arg2_syn_tensor.get()},
        {{shape, dtype}});

    auto out_type = (self.scalar_type() == c10::ScalarType::Long)
        ? (c10::ScalarType::Int)
        : self.scalar_type();
    std::string cast_guid_2 = "cast_" +
        habana_helpers::name_suffix_from_type(dtype) + "_to_" +
        habana_helpers::name_suffix_from_type(out_type);
    castOp =
        BuildOp(graph, cast_guid_2, {indexOp[0].get()}, {{shape, out_type, 0}});
    syn_out(0) = std::move(castOp[0]);
    return;
  }
  auto indexOp = BuildOp(
      graph,
      "gather_nd_mxnet_fwd_" +
          habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), arg2_syn_tensor.get()},
      {{shape, ScalarType(), 0}});
  syn_out(0) = std::move(indexOp[0]);
}

} // namespace habana
