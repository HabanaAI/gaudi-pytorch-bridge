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

#include <cstddef>
#include <cstdint>
#include "generated/backend/gather.h"
#include "generated/backend/index.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "hpu_ops/backend/arange.h"
#include "hpu_ops/common/index.h"
#include "hpu_ops/indexing_ops_helper.h"

namespace habana {

static sizes_vec IndexOutShapeFromOrigStack(const at::Stack& stack) {
  at::Tensor self = stack_tensor(stack, 0);
  c10::ArrayRef<c10::IValue> indices_ival = stack.at(1).toListRef();
  std::array<int64_t, MAX_DIMS_FOR_ADVANCED_INDEXING> adv_index_dims = {
      -1, -1, -1, -1, -1};

  std::vector<at::Tensor> indices;
  std::vector<int64_t> self_permute_dims(self.dim());
  bool explicit_indices_together = false;
  int index_tensor_groups = 0;
  int explicit_indices_count = 0;
  bool adv_indexing_present = false;
  for (auto input : indices_ival) {
    auto o1 = input.toOptional<at::Tensor>();
    if (o1.has_value() && !o1->defined()) {
      if (explicit_indices_together) {
        explicit_indices_together = false;
      }
    } else if (o1.has_value() && o1->defined()) {
      if (!explicit_indices_together) {
        index_tensor_groups++;
      }
      explicit_indices_together = true;
      explicit_indices_count += (int)o1.value().sizes().size();
    }
  }
  if (index_tensor_groups > 1) {
    std::vector<at::Tensor> t_indices;
    std::tie(self_permute_dims, t_indices) = transposeToFront(stack);
    for (int i = 0; i < explicit_indices_count; i++) {
      adv_index_dims[i] = t_indices[i].sizes()[0];
    }
    for (int i = explicit_indices_count; i < (int)indices_ival.size(); i++) {
      adv_index_dims[i] = -1;
      adv_indexing_present = true;
    }
  } else {
    for (const auto i : c10::irange(self.dim())) {
      self_permute_dims[i] = i;
    }
    int i = 0;
    for (const auto& index_opt : indices_ival) {
      auto o1 = index_opt.toOptional<at::Tensor>();
      if (o1.has_value() && !o1.value().defined()) {
        // Don't add undefined tensors to list as Lazy infra can't handle such
        // tensors
        adv_index_dims[i] = -1;
        adv_indexing_present = true;
      } else if (o1.has_value() && o1.value().defined()) {
        const auto& index = o1.value();
        adv_index_dims[i] = index.sizes()[0];
      }
      i++;
    }
  }

  if (adv_indexing_present) {
    std::vector<int64_t> permuted_input_sizes, new_strides;
    std::tie(permuted_input_sizes, new_strides) =
        PermuteOperator::compute_output_shape(self, self_permute_dims);

    std::vector<int64_t> output_shape;
    std::vector<int64_t> input_shape;
    int64_t largest_specified_index_t_size = 0;
    for (int i = 0; i < (int)permuted_input_sizes.size(); i++) {
      if (adv_index_dims[i] > largest_specified_index_t_size)
        largest_specified_index_t_size = adv_index_dims[i];
    }
    bool explicit_index_found = false;
    for (int i = 0; i < (int)permuted_input_sizes.size(); i++) {
      if (adv_index_dims[i] != -1) { // dim has explicit index tensor
        if (!explicit_index_found) {
          output_shape.emplace_back(largest_specified_index_t_size);
          explicit_index_found = true;
        }
      } else { // advanced indexing done on this dim
        output_shape.emplace_back(permuted_input_sizes[i]);
      }
    }
    return std::vector<std::vector<int64_t>>{output_shape};
  } else {
    std::vector<at::Tensor> indices;
    for (const auto& i : indices_ival) {
      indices.emplace_back(i.toTensor());
    }
    sizes_vec shape = std::vector<std::vector<int64_t>>{
        ComputeIndexOperatorOutputShape(self, indices)};
    return shape;
  }
}

sizes_vec IndexOutputShape(const at::Stack& stack) {
  const at::Tensor input = stack_tensor(stack, 0);
  auto indices = stack.at(1).toTensorList().vec();
  if (stack.size() > 2) { // indicates that we are getting the custom schema
                          // with additional info
    std::vector<bool> adv_ind_dim = stack[2].toBoolList().vec();
    const int num_index_tensors = stack[4].toInt();
    const bool adv_indexing_present = std::any_of(
        adv_ind_dim.cbegin(),
        adv_ind_dim.cbegin() + num_index_tensors,
        [](const auto& i) { return i == true; });
    if (adv_indexing_present) {
      auto indexing_tensor_shapes = calc_indexing_tensors_shapes(stack);
      std::vector<int64_t> self_permute_dims = stack[3].toIntList().vec();
      std::vector<int64_t> permuted_input_sizes, new_strides;
      std::tie(permuted_input_sizes, new_strides) =
          PermuteOperator::compute_output_shape(input, self_permute_dims);
      sizes_vec shape = std::vector<std::vector<int64_t>>{
          {habana::ComputeOutputShapeWithAdvIndexing(
              permuted_input_sizes, adv_ind_dim, indexing_tensor_shapes)}};
      return shape;
    } else {
      sizes_vec shape = std::vector<std::vector<int64_t>>{
          ComputeIndexOperatorOutputShape(input, indices)};
      return shape;
    }
  } else {
    TORCH_CHECK(
        "!!!Not expected to hit IndexOutShapeFromOrigStack as index op uses custom schema!!!");
    return IndexOutShapeFromOrigStack(stack);
  }
}

OutputMetaDataVector IndexMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  OutputMetaData meta{};

  meta.dtype = self.scalar_type();
  meta.shape = IndexOutputShape(stack)[0];

  return {meta};
}

static std::shared_ptr<void> FillPermuteParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(synTransposeParamsNDims);
  auto self = stack.at(0).toTensor();
  auto permute_dim_arr = stack[3].toIntList().vec();
  params->tensorDim = self.dim();
  // params.permute has to be populated in a reverse order for HPU FCD-LCD order
  for (int i = 0; i < self.dim(); i++) {
    params->permutation[i] = static_cast<TransposePermutationDim>(
        self.dim() - permute_dim_arr[permute_dim_arr.size() - i - 1] - 1);
  }
  for (int i = self.dim(); i < HABANA_DIM_MAX; i++) {
    params->permutation[i] = static_cast<TransposePermutationDim>(i);
  }

  return params;
}

std::shared_ptr<void> FillIndexParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_IndexKernel::Params);

  auto const& adv_indexing_dims = stack.at(2).toBoolList();
  auto const aid_size = adv_indexing_dims.size();
  for (size_t i = 0; i < aid_size; ++i)
    params->advanced_indexing_dims[i] = adv_indexing_dims[i];

  auto const& self_permute_dims = stack.at(3).toIntList();
  auto const spd_size = self_permute_dims.size();
  for (size_t i = 0; i < spd_size; ++i)
    params->self_permute_dims[i] = self_permute_dims[i];

  params->num_index_tensors = stack.at(4).toScalar().toInt();

  return params;
}

SharedMetaDataVector IndexSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& input = stack_tensor(stack, 0);
  auto rank = input.dim();
  auto dtype = input.scalar_type();
  const auto indices = stack.at(1).toListRef();
  SharedMetaData indexSharedMeta{"index"};
  indexSharedMeta.inputs_data.emplace_back(rank, dtype);
  int64_t broadcastedIndicesRank = 1;
  auto notNoneIndicesNum = 0;
  for (const auto& index : indices) {
    if (!index.isNone()) {
      const auto rank = index.toTensor().dim();
      indexSharedMeta.inputs_data.emplace_back(rank, c10::ScalarType::Int);
      broadcastedIndicesRank = std::max(broadcastedIndicesRank, rank);
      notNoneIndicesNum++;
    } else {
      indexSharedMeta.inputs_data.emplace_back(1, c10::ScalarType::Int);
    }
  }

  auto outputRank = stack.back().isTensor()
      ? stack.back().toTensor().dim()
      : broadcastedIndicesRank + rank - notNoneIndicesNum;
  indexSharedMeta.outputs_data.emplace_back(outputRank, dtype);

  return {indexSharedMeta};
}

void IndexHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE)) {
    // new implementation only for eager.
    size_t size = 0;
    auto params = FillIndexParams(stack, size);
    auto meta = IndexMeta(stack)[0];

    StackGetter stackGetter(this, stack, "IndexHabanaOperator::AddNode");
    auto input = stackGetter.getNextInput<TensorsPair>();
    auto indices = stackGetter.getNextInput<std::vector<TensorsPair>>();

    std::vector<synTensor> index_input{input.syn_t};
    for (auto const& index : indices)
      index_input.push_back(index.syn_t);

    auto result = BuildOp(
        graph,
        get_guid_with_precision("index", meta.dtype),
        std::move(index_input),
        {{meta.shape, meta.dtype, 0}},
        params.get(),
        size);

    syn_out(0) = std::move(result[0]);

    return;
  }

  // leave old implementation for lazy.
  const at::Tensor self = stack_tensor(stack, 0);
  const c10::List<at::Tensor> indices = stack.at(1).toTensorList();

  bool adv_indexing_present = false;
  if (stack.size() > 2) {
    std::vector<bool> adv_ind_dim = stack[2].toBoolList().vec();
    const std::vector<int64_t> self_permute_dims = stack[3].toIntList().vec();
    const int num_index_tensors = stack[4].toInt();
    adv_indexing_present = std::any_of(
        adv_ind_dim.cbegin(),
        adv_ind_dim.cbegin() + num_index_tensors,
        [](const auto& i) { return i == true; });
  }

  // find out final indexing tensor shapes - includes dims with
  // advanced/implicit indexing
  auto indexing_tensor_shapes = calc_indexing_tensors_shapes(stack);
  c10::ScalarType index_dtype =
      common::IsInt64Supported() ? c10::ScalarType::Long : c10::ScalarType::Int;

  if (!adv_indexing_present) {
    // for this particular indices configuration gather_mxnet throws GC
    // compilation error, therefore use simple gather for now
    if (indices.size() == 1 && indices.get(0).dim() == 1) {
      auto outshape = ComputeGatherOperatorOutputShape(self, 0, indices[0]);
      const int dim = 0;
      const bool sparse_grad = false;
      const at::Stack stack_ = {
          c10::IValue(self),
          c10::IValue(dim),
          c10::IValue(indices[0]),
          c10::IValue(sparse_grad)};

      // Fill params for gather
      size_t size = 0;
      const auto& gather_params = FillGatherParams(stack_, size);
      auto gatherOp = BuildOp(
          graph,
          get_guid_with_precision("gather_fwd", ScalarType()),
          {syn_in(0), syn_in(1)},
          {{outshape, ScalarType(), 0}},
          gather_params.get(),
          size);
      syn_out(0) = std::move(gatherOp[0]);
      return;
    }

    auto tensorlist = stack[1].toTensorList().vec();

    auto max_size = broadcast_size(tensorlist);
    auto max_dims = (int)max_size.size();
    int64_t max_num_elems = (int64_t)std::accumulate(
        max_size.begin(), max_size.end(), 1, std::multiplies<int64_t>());
    auto scalar_type = tensorlist[0].scalar_type();

    std::vector<synTensor> cat_input_synTensor;
    std::vector<synapse_helpers::tensor> cat_input_tensor;
    std::vector<std::vector<int64_t>> cat_input_index;
    for (size_t i = 0; i < tensorlist.size(); i++) {
      auto t_sz = tensorlist[i].sizes().vec();
      int num_dims = (int)t_sz.size();
      int64_t num_elems = (int64_t)std::accumulate(
          t_sz.begin(), t_sz.end(), 1, std::multiplies<int64_t>());
      std::vector<synTensor> index_maybe_multidim_synTensor{syn_in(i + 1)};
      std::unique_ptr<synapse_helpers::tensor> index_maybe_multidim_shTensor;
      bool reshape_before_bcast = false;
      if (num_elems > 1 || (num_elems == 1 && num_dims > 1))
        reshape_before_bcast = true;
      if (((num_elems > 1) && (num_elems < max_num_elems)) ||
          (num_elems == 1 && num_dims > 1)) {
        std::vector<int64_t> reshape_outshape(max_dims, 1);
        for (int i = max_dims - num_dims; i < max_dims; i++) {
          reshape_outshape[i] = t_sz[i - (max_dims - num_dims)];
        }
        auto reshaped_index =
            ReshapeHelper(graph, syn_in(i + 1), reshape_outshape, index_dtype);
        index_maybe_multidim_shTensor =
            std::make_unique<synapse_helpers::tensor>(BroadcastHelper(
                graph, reshaped_index.get(), max_size, index_dtype));
        // re-calculate num_elems for expanded tensor
        auto t_sz = index_maybe_multidim_shTensor->pt_shape();
        num_elems = (int64_t)std::accumulate(
            t_sz.begin(), t_sz.end(), 1, std::multiplies<int64_t>());
        index_maybe_multidim_synTensor = {index_maybe_multidim_shTensor->get()};
      }
      auto bcastOp = BroadcastHelper(
          graph,
          (reshape_before_bcast) ? ReshapeHelper(
                                       graph,
                                       index_maybe_multidim_synTensor[0],
                                       std::vector<int64_t>{num_elems},
                                       index_dtype)
                                       .get()
                                 : index_maybe_multidim_synTensor[0],
          max_num_elems,
          index_dtype);
      std::vector<int64_t> expanded_size{1};
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
        std::move(cat_input_synTensor),
        {{cat_out_size, scalar_type}},
        &concat_params,
        sizeof(concat_params));
    auto catop = std::move(catop1.at(0));

    auto cat_out_shape = catop.pt_shape();
    auto self_sizes = self.sizes().vec();
    int cat_indices_count = cat_out_shape[0];
    std::vector<int64_t> shape = {cat_out_shape[1]};
    // If index tensors index only upper dimensions, then the result's lower
    // dims shapes should be determined based on self shape at those dims.
    for (int i = cat_indices_count; i < self.dim(); i++) {
      shape.emplace_back(self_sizes[i]);
    }
    auto indexOp = BuildOp(
        graph,
        get_guid_with_precision("gather_nd_mxnet_fwd", ScalarType()),
        {syn_in(0), catop.get()},
        {{shape, ScalarType()}});
    auto final_shape = ComputeIndexOperatorOutputShape(self, tensorlist);
    auto index_out =
        ReshapeHelper(graph, indexOp[0].get(), final_shape, ScalarType(), 0);
    syn_out(0) = std::move(index_out);
  } else { // start - advanced indexing present
    int64_t explicit_index_count = 0;
    std::vector<int64_t> broadcast_to_size = {1};
    std::vector<bool> index_all_elems(self.dim());
    int64_t repeats_needed[self.dim()];
    int64_t repeat_interleaves_needed[self.dim()];
    std::vector<int64_t> indices_size_with_adv_indexing;
    std::vector<synTensor> indices_list;
    std::vector<synTensor> cat_input_synTensor;
    std::vector<synapse_helpers::tensor> cat_input_tensor;
    std::vector<std::vector<int64_t>> cat_input_index;
    std::vector<bool> adv_ind_dim = stack[2].toBoolList().vec();
    const std::vector<int64_t> self_permute_dims = stack[3].toIntList().vec();

    synTensor permuted_self_t;
    size_t size = 0;
    const auto& params = FillPermuteParams(stack, size);
    std::vector<int64_t> new_sizes, new_strides;
    std::tie(new_sizes, new_strides) =
        PermuteOperator::compute_output_shape(self, self_permute_dims);
    auto permuted_self = BuildOp(
        graph,
        "transpose",
        {syn_in(0)},
        {{new_sizes, ScalarType()}},
        params.get(),
        size);

    auto permuted_self_shape = permuted_self[0].pt_shape();
    permuted_self_t = std::move(permuted_self[0].get());
    int64_t i = 0;
    for (; i < (int64_t)adv_ind_dim.size(); ++i) {
      if (adv_ind_dim[i] == true) {
        index_all_elems[i] = true;
      } else {
        auto cur_index_dim_size = (int64_t)std::accumulate(
            indexing_tensor_shapes[i].begin(),
            indexing_tensor_shapes[i].end(),
            1,
            std::multiplies<int64_t>());
        auto broadcast_to_size_numel = (int64_t)std::accumulate(
            broadcast_to_size.begin(),
            broadcast_to_size.end(),
            1,
            std::multiplies<int64_t>());
        if (cur_index_dim_size >= broadcast_to_size_numel) {
          broadcast_to_size = indexing_tensor_shapes[i];
        }
        index_all_elems[i] = false;
        explicit_index_count++;
      }
    }
    auto it = std::find(adv_ind_dim.begin(), adv_ind_dim.end(), false);
    int explicit_index_tensor_group_start = (int)(it - adv_ind_dim.begin());

    // account for any trailing dims that are not specified to be
    // indexed explicitly, but need to be taken care of.
    for (; i < self.dim(); ++i) {
      index_all_elems[i] = true;
    }

    auto broadcast_to_size_numel = (int64_t)std::accumulate(
        broadcast_to_size.begin(),
        broadcast_to_size.end(),
        1,
        std::multiplies<int64_t>());
    for (i = 0; i < self.dim(); i++) {
      repeats_needed[i] = 1;
      int64_t total_elements_above = 1;
      bool explicit_index_above = false;
      for (int64_t j = 0; j < i; j++) {
        if ((j >= explicit_index_tensor_group_start) &&
            (j < explicit_index_tensor_group_start + explicit_index_count)) {
          explicit_index_above = true;
        } else {
          total_elements_above *= permuted_self_shape[j];
        }
      }
      repeats_needed[i] = total_elements_above;
      int64_t index_numel;
      if (i < (int64_t)indexing_tensor_shapes.size()) {
        index_numel = (int64_t)std::accumulate(
            indexing_tensor_shapes[i].begin(),
            indexing_tensor_shapes[i].end(),
            1,
            std::multiplies<int64_t>());
      } else {
        index_numel = permuted_self_shape[i];
      }
      if (index_all_elems[i] && explicit_index_above) {
        repeats_needed[i] *= broadcast_to_size_numel;
      } else if (
          !index_all_elems[i] && (index_numel < broadcast_to_size_numel)) {
        repeats_needed[i] *= broadcast_to_size_numel;
      }
    }
    for (i = 0; i < self.dim(); i++) {
      repeat_interleaves_needed[i] = 1;
      int64_t total_elements_below = 1;
      bool explicit_index_below = false;
      for (int j = i + 1; j < self.dim(); j++) {
        if ((j >= explicit_index_tensor_group_start) &&
            (j < explicit_index_tensor_group_start + explicit_index_count)) {
          explicit_index_below = true;
        } else {
          total_elements_below *= permuted_self_shape[j];
        }
      }
      if (index_all_elems[i] && explicit_index_below) {
        total_elements_below *= broadcast_to_size_numel;
      }
      repeat_interleaves_needed[i] = total_elements_below;
    }

    int explicit_index_pos = 0;
    std::vector<synapse_helpers::tensor> index_tensor_to_use;
    for (int dim = 0; dim < self.dim(); dim++) {
      at::Tensor it;
      int64_t num_elems;
      if (index_all_elems[dim]) {
        std::vector<int64_t> outshape{permuted_self_shape[dim]};
        size_t size = 0;
        at::Stack arange_stack = {};
        num_elems = permuted_self_shape[dim];
        auto params = FillArangeParamsInternal(
            0, permuted_self_shape[dim], 1, index_dtype, size);

        index_tensor_to_use.emplace_back(ArangeCommon(
            this,
            graph,
            0,
            permuted_self_shape[dim],
            1,
            index_dtype,
            syn_in(0), // TBD: NOTE: This needs to be changed for DS
            syn_in(1), // TBD: NOTE: This needs to be changed for DS
            get_guid_with_precision("range", index_dtype),
            outshape,
            params,
            size,
            c10::nullopt));
        if ((broadcast_to_size_numel == 1) &&
            (repeat_interleaves_needed[dim] == 1) &&
            (repeats_needed[dim] == 1)) {
          std::vector<int64_t> expanded_size{1};
          for (auto s : index_tensor_to_use.back().pt_shape()) {
            expanded_size.push_back(s);
          }
          cat_input_tensor.emplace_back(ReshapeHelper(
              graph,
              index_tensor_to_use.back().get(),
              expanded_size,
              index_dtype));
          cat_input_synTensor.emplace_back(
              cat_input_tensor[cat_input_tensor.size() - 1].get());
          cat_input_index.emplace_back(
              cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
        }
      } else {
        // only explicit indices in stack and they start from pos=1.
        auto t_sz = indices.get(explicit_index_pos).sizes().vec();
        num_elems = (int64_t)std::accumulate(
            t_sz.begin(), t_sz.end(), 1, std::multiplies<int64_t>());
        explicit_index_pos++;
      }
      /*
        Implement the required repeat_interleave as broadcast followed by
        reshape: E.g., if self.sizes()[dim] = 4, and
        repeat_interleave_count[dim] = 3, then we need the resulting index
        tensor contents as [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]. To get this
        do a arange(self.sizes()[dim]) which gives a tensor with contents [0,
        1, 2, 3] of shape {4}. Now broadcast it to {3, 4} to get contents [[0,
        1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]. Transpose this to get
          [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]] of shape {4, 3}.
        Reshape this to {12} with contents [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,
        3].
        */
      if (repeat_interleaves_needed[dim] > 1) {
        std::vector<int64_t> rinlv_bcast_size{
            repeat_interleaves_needed[dim], num_elems};
        auto bcastOp = BroadcastHelper(
            graph,
            ((index_all_elems[dim]) ? index_tensor_to_use.back().get()
                                    : ReshapeHelper(
                                          graph,
                                          syn_in(explicit_index_pos),
                                          std::vector<int64_t>{num_elems},
                                          index_dtype)
                                          .get()),
            rinlv_bcast_size,
            index_dtype);

        auto rnilv_transpose_params =
            std::make_shared<synTransposeParamsNDims>();
        std::vector<int64_t> t_dim_arr{1, 0};
        std::vector<int64_t> t_new_sizes{
            num_elems, repeat_interleaves_needed[dim]};
        rnilv_transpose_params->tensorDim = (int64_t)rinlv_bcast_size.size();
        rnilv_transpose_params->permutation[0] =
            static_cast<TransposePermutationDim>(
                rnilv_transpose_params->tensorDim -
                t_dim_arr[t_dim_arr.size() - 1] - 1);
        rnilv_transpose_params->permutation[1] =
            static_cast<TransposePermutationDim>(
                rnilv_transpose_params->tensorDim -
                t_dim_arr[t_dim_arr.size() - 2] - 1);
        size = sizeof(synTransposeParamsNDims);
        auto t_op = BuildOp(
            graph,
            "transpose",
            {bcastOp.get()},
            {{t_new_sizes, index_dtype}},
            rnilv_transpose_params.get(),
            size);
        std::vector<int64_t> reshape_size = {
            num_elems * repeat_interleaves_needed[dim]};
        std::vector<int64_t> reshape_outshape = {reshape_size};
        auto reshaped_index =
            ReshapeHelper(graph, t_op[0].get(), reshape_outshape, index_dtype);

        std::vector<int64_t> rpt_outshape = {
            num_elems * repeat_interleaves_needed[dim] * repeats_needed[dim]};
        auto tile_params = std::make_shared<ns_TileKernel::ParamsV2>();
        size = sizeof(ns_TileKernel::ParamsV2);
        for (int i = 0; i < MAX_TPC_SUPPORTED_REPEAT_DIMS; i++) {
          tile_params->repeat[i] = 1;
        }
        tile_params->repeat[0] = repeats_needed[dim];
        auto rpt_op = BuildOp(
            graph,
            get_guid_with_precision("tile_fwd", index_dtype),
            {reshaped_index.get()},
            {{rpt_outshape, index_dtype}},
            tile_params.get(),
            size);

        indices_size_with_adv_indexing = rpt_outshape;
        auto tensorlist = stack[1].toTensorList().vec();
        std::vector<int64_t> expanded_size{1};
        for (auto s : rpt_op[0].pt_shape()) {
          expanded_size.push_back(s);
        }
        cat_input_tensor.emplace_back(
            ReshapeHelper(graph, rpt_op[0].get(), expanded_size, index_dtype));
        cat_input_synTensor.emplace_back(
            cat_input_tensor[cat_input_tensor.size() - 1].get());
        cat_input_index.emplace_back(
            cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
      } else if (repeats_needed[dim] > 1) {
        std::vector<int64_t> rpt_outshape = {num_elems * repeats_needed[dim]};
        auto tile_params = std::make_shared<ns_TileKernel::ParamsV2>();
        size = sizeof(ns_TileKernel::ParamsV2);
        for (int i = 0; i < MAX_TPC_SUPPORTED_REPEAT_DIMS; i++) {
          tile_params->repeat[i] = 1;
        }
        tile_params->repeat[0] = repeats_needed[dim];
        auto rpt_op = BuildOp(
            graph,
            get_guid_with_precision("tile_fwd", index_dtype),
            {((index_all_elems[dim]) ? index_tensor_to_use.back().get()
                                     : ReshapeHelper(
                                           graph,
                                           syn_in(explicit_index_pos),
                                           std::vector<int64_t>{num_elems},
                                           index_dtype)
                                           .get())},
            {{rpt_outshape, index_dtype}},
            tile_params.get(),
            size);
        indices_size_with_adv_indexing = rpt_outshape;
        auto tensorlist = stack[1].toTensorList().vec();
        std::vector<int64_t> expanded_size{1};
        for (auto s : rpt_op[0].pt_shape()) {
          expanded_size.push_back(s);
        }
        cat_input_tensor.emplace_back(
            ReshapeHelper(graph, rpt_op[0].get(), expanded_size, index_dtype));
        cat_input_synTensor.emplace_back(
            cat_input_tensor[cat_input_tensor.size() - 1].get());
        cat_input_index.emplace_back(
            cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
      } else if (!index_all_elems[dim]) {
        //"explicit_index_pos - 1" used below because it's already been
        // incremented
        std::vector<int64_t> expanded_size{1};
        expanded_size.push_back(indices.get(explicit_index_pos - 1).numel());
        cat_input_tensor.emplace_back(ReshapeHelper(
            graph,
            syn_in(explicit_index_pos), // no "-1" as indices tensors start from
                                        // pos-1 in syn_in
            expanded_size,
            index_dtype));

        cat_input_synTensor.emplace_back(
            cat_input_tensor[cat_input_tensor.size() - 1].get());
        cat_input_index.emplace_back(
            cat_input_tensor[cat_input_tensor.size() - 1].pt_shape());
      }
    }
    auto tensorlist = stack[1].toTensorList().vec();
    auto scalar_type = tensorlist[0].scalar_type();
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
        std::move(cat_input_synTensor),
        {{cat_out_size, scalar_type}},
        &concat_params,
        sizeof(concat_params));
    auto catop = std::move(catop1.at(0));

    auto cat_out_shape = catop.pt_shape();
    std::vector<int64_t> shape = {cat_out_shape[1]};
    auto indexOp = BuildOp(
        graph,
        get_guid_with_precision("gather_nd_mxnet_fwd", ScalarType()),
        {permuted_self_t, catop.get()},
        {{shape, ScalarType()}});
    auto index_out_shape = indexOp[0].pt_shape();
    auto final_shape = habana::ComputeOutputShapeWithAdvIndexing(
        permuted_self_shape, index_all_elems, indexing_tensor_shapes);
    auto index_out =
        ReshapeHelper(graph, indexOp[0].get(), final_shape, ScalarType(), 0);
    syn_out(0) = std::move(index_out);
  } // end - advanced indexing present
}

std::vector<int64_t> ComputeGatherOperatorOutputShape(
    const at::Tensor& self,
    int64_t dim_,
    const at::Tensor& index) {
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  auto shape = self.sizes().vec();
  if (shape.size()) {
    // for gather op, output size is same as index
    if (self.dim() == index.dim()) {
      shape = index.sizes().vec();
    } else {
      // for index_select and other index ops
      shape.erase(shape.begin() + dim);
      shape.insert(shape.begin() + dim, index.numel());
    }
  }
  return shape;
}

} // namespace habana
