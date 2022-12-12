/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/gather.h"
#include "generated/index.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"

namespace habana {

FALLBACK_CHECK(
    IndexFallbackCheck,
    const c10::List<c10::optional<at::Tensor>>& indices) {
  at::Stack stack = {indices};
  c10::ArrayRef<c10::IValue> indices_in = stack.at(0).toListRef();
  for (auto input : indices_in) {
    auto o1 = input.toOptional<at::Tensor>();
    if (!(o1.has_value() && !o1->defined())) {
      continue;
    } else {
      return false; // advanced indexing is currently unsupported on HPU -
                    // fallback to CPU
    }
  }
  return true;
};

// brodcast index tensor shape and get the correct shape and size
static std::vector<int64_t> broadcast_size(at::TensorList indices) {
  auto size = indices[0].sizes().vec();
  for (size_t i = 1; i < indices.size(); i++) {
    size = at::infer_size(size, indices[i].sizes());
  }
  return size;
}
static std::vector<int64_t> CalcCatOutSize(
    const std::vector<std::vector<int64_t>>* tensors,
    int64_t* dim_inp) {
  auto tensor_count = tensors->size();

  if (tensor_count == 0) // if tensor is empty or its first element is empty,
                         // then concatenate out size is 0
    return {0};

  int64_t dim =
      at::maybe_wrap_dim(*dim_inp, tensors->at(0).size(), /*wrap_scalar=*/true);

  CatOutOperator::validate_cat_tensor_dim_sizes(tensors, *dim_inp);

  if (dim != *dim_inp) {
    *dim_inp = dim;
  }

  // out tensor size should match along all dimensions for input tensors except
  // along the dim in which to cat
  auto out_size = tensors->at(0);
  if (out_size.size() != 0) {
    out_size[dim] = 0;
    for (unsigned i = 0; i < tensor_count; i++)
      out_size[dim] += tensors->at(i)[dim];
  }

  return out_size;
}

sizes_vec IndexOutputShape(const at::Stack& stack) {
  if (!habana_lazy::isDeviceInLoweringMode()) {
    return {};
  }
  const at::Tensor input = stack_tensor(stack, 0);
  auto indices = stack.at(1).toTensorList().vec();
  sizes_vec shape = std::vector<std::vector<int64_t>>{
      {IndexOperator::compute_output_shape(input, indices)}};
  return shape;
}

template <>
LazyIndex<at::Tensor>::LazyIndex(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn, -1) {
  habana_lazy::NoAccThread no_acc_thread;

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
          "None is not yet supported on HPU for c10::List<c10::optional<Tensor>>");
    }
  }

  for (size_t i = 0; i < indices_vec.size(); i++) {
    if (indices_vec[i].device().type() != c10::DeviceType::HPU) {
      indices_vec[i] = indices_vec[i].to(c10::kHPU);
    }
  }

  // handle views for tensorlist indices
  at::TensorList indices_in_list(indices_vec);
  indices_vec =
      habana_lazy::HbLazyTensorViews::HandleViewsTensorList(indices_in_list);

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

  auto indices_out_vec =
      habana_lazy::HbLazyTensorViews::HandleViewsTensorList(indices);
  at::TensorList indices_out_list(indices_out_vec);

  get_inputs().back() = indices_out_list;
}

template <>
at::Tensor LazyIndex<at::Tensor>::get_result_overrideable() {
  auto inputs = get_inputs();
  const at::Tensor input = inputs[0].toTensor();
  auto indices = inputs[1].toTensorList().vec();

  auto shape = IndexOperator::compute_output_shape(input, indices);
  return habana_lazy::empty_hpu_lazy(
      shape, input.options(), input.suggest_memory_format(), false);
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
  auto scalar_type = tensorlist[0].scalar_type();

  std::vector<synTensor> cat_input_synTensor;
  std::vector<synapse_helpers::tensor> cat_input_tensor;
  std::vector<std::vector<int64_t>> cat_input_index;

  for (size_t i = 0; i < tensorlist.size(); i++) {
    // syn(i + 1) - Could give seg fault if undefined tensors are passed by user
    auto bcastOp = BroadcastHelper(graph, syn_in(i + 1), max_size, scalar_type);

    std::vector<int64_t> expanded_size{1}; // {1, max_size}
    for (auto s : bcastOp.pt_shape()) {
      expanded_size.push_back(s);
    }

    cat_input_tensor.emplace_back(
        ReshapeHelper(graph, bcastOp.get(), expanded_size, scalar_type));

    cat_input_synTensor.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].get());
    cat_input_index.emplace_back(
        cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
  }

  int64_t dim = 0;
  std::vector<int64_t> cat_out_size = CalcCatOutSize(&cat_input_index, &dim);
  dim = cat_out_size.size() > 0
      ? (cat_out_size.size() - dim) - 1
      : 0; // if tensor is empty then dim of the concatenated tensor will be 0

  synConcatenateParams concat_params{};
  concat_params.axis = dim;

  auto catop1 = BuildOp(
      graph,
      "concat",
      cat_input_synTensor,
      {{cat_out_size, scalar_type}},
      &concat_params,
      sizeof(concat_params));

  auto catop = std::move(catop1.at(0));

  auto shape = IndexOperator::compute_output_shape(self, tensorlist);
  auto indexOp = BuildOp(
      graph,
      "gather_nd_mxnet_fwd_" +
          habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), catop.get()},
      {{shape, ScalarType(), 0}});
  syn_out(0) = std::move(indexOp[0]);
}

} // namespace habana
