/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/arange.h"
#include "generated/backend/gather.h"
#include "generated/backend/index.h"
#include "generated/backend/index_select.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "hpu_ops/backend/arange.h"
#include "hpu_ops/common/arange_gen.h"
#include "hpu_ops/common/index.h"

#define MAX_TPC_SUPPORTED_REPEAT_DIMS (5)

namespace habana {

// broadcast index tensor shape and get the correct shape and size
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

  CatOperator::validate_cat_tensor_dim_sizes(tensors, *dim_inp);

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
  bool adv_indexing_present = false;
  std::vector<int64_t> implicit_indices_pos_vec = stack[3].toIntList().vec();
  for (auto i : implicit_indices_pos_vec) {
    if (i == -1) {
      adv_indexing_present = true;
      break;
    }
  }
  if (adv_indexing_present) {
    auto adv_index_dims = stack.at(2).toIntList();
    std::vector<int64_t> self_permute_dims = stack[4].toIntList().vec();
    std::vector<int64_t> new_sizes, new_strides;
    std::tie(new_sizes, new_strides) =
        PermuteOperator::compute_output_shape(input, self_permute_dims);

    std::vector<int64_t> permuted_input_sizes;
    if (adv_indexing_present) {
      permuted_input_sizes = new_sizes;
    } else {
      permuted_input_sizes = input.sizes().vec();
    }

    sizes_vec shape = std::vector<std::vector<int64_t>>{
        {habana::ComputeOutputShapeWithAdvIndexing(
            permuted_input_sizes, indices, adv_index_dims, true)}};
    return shape;
  } else {
    const at::Tensor input = stack_tensor(stack, 0);
    auto indices = stack.at(1).toTensorList().vec();
    sizes_vec shape = std::vector<std::vector<int64_t>>{
        {ComputeIndexOperatorOutputShape(input, indices)}};
    return shape;
  }
}

static C10_UNUSED int hasContiguousSubspace(
    std::vector<int64_t> implicit_indices_pos_vec) {
  bool explicit_indices_together = false;
  int index_tensor_groups = 0;
  int index_tensor_group_start = 0;
  int dim = 0;
  for (auto pos : implicit_indices_pos_vec) {
    if (pos == -1) {
      if (explicit_indices_together) {
        explicit_indices_together = false;
      }
    } else {
      if (!explicit_indices_together) {
        index_tensor_group_start = dim;
        index_tensor_groups++;
      }
      explicit_indices_together = true;
    }
    dim++;
  }
  if (index_tensor_groups <= 1)
    return index_tensor_group_start;
  else
    return 0;
}

static std::shared_ptr<void> FillPermuteParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(synTransposeParamsNDims);
  auto self = stack.at(0).toTensor();
  auto permute_dim_arr = stack[4].toIntList().vec();

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

void IndexHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  std::vector<int64_t> implicit_indices_pos_vec = stack[3].toIntList().vec();
  bool adv_indexing_present = false;

  for (auto i : implicit_indices_pos_vec) {
    if (i == -1) { // advanced indexing is present
      adv_indexing_present = true;
      break;
    }
  }
  const at::Tensor self = stack_tensor(stack, 0);
  c10::List<at::Tensor> indices = stack.at(1).toTensorList();
  auto adv_index_dims = stack[2].toIntList();
  std::vector<int64_t> self_permute_dims = stack[4].toIntList().vec();

  if (!adv_indexing_present) {
    // for this particular indices configuration gather_mxnet throws GC
    // compilation error, therefore use simple gather for now
    if (indices.size() == 1 && indices.get(0).dim() == 1) {
      auto outshape = ComputeGatherOperatorOutputShape(self, 0, indices[0]);

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
      auto bcastOp =
          BroadcastHelper(graph, syn_in(i + 1), max_size, scalar_type);

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

    auto shape = ComputeIndexOperatorOutputShape(self, tensorlist);
    auto indexOp = BuildOp(
        graph,
        "gather_nd_mxnet_fwd_" +
            habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), catop.get()},
        {{shape, ScalarType(), 0}});
    syn_out(0) = std::move(indexOp[0]);
  } else { // start - advanced indexing present
    std::vector<int64_t> self_permuted_sizes(self.dim());
    int64_t explicit_index_count = 0;
    auto explicit_index_tensor_group_start =
        hasContiguousSubspace(implicit_indices_pos_vec);
    int64_t broadcast_to_size = 1;
    bool index_all_elems[self.dim()];
    int64_t index_t_sizes[self.dim()];
    int64_t repeats_needed[self.dim()];
    int64_t repeat_interleaves_needed[self.dim()];
    std::vector<int64_t> indices_size_with_adv_indexing;
    std::vector<synTensor> indices_list;
    std::vector<synTensor> cat_input_synTensor;
    std::vector<synapse_helpers::tensor> cat_input_tensor;
    std::vector<std::vector<int64_t>> cat_input_index;
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

    if (!explicit_index_tensor_group_start) {
      auto tv = self.sizes().vec();
      ;
      for (int i = 0; i < self.dim(); i++) {
        self_permuted_sizes[i] = tv[self_permute_dims[i]];
      }
    } else {
      self_permuted_sizes = self.sizes().vec();
    }
    int i = 0;
    for (; i < implicit_indices_pos_vec.size(); i++) {
      if (implicit_indices_pos_vec[i] == -1) {
        index_t_sizes[i] = self_permuted_sizes[i];
        index_all_elems[i] = true;
      } else {
        auto cur_index_dim_size = adv_index_dims[i];
        if (cur_index_dim_size >= broadcast_to_size) {
          broadcast_to_size = cur_index_dim_size;
        }
        index_all_elems[i] = false;
        index_t_sizes[i] = adv_index_dims[i];
        explicit_index_count++;
      }
    }

    // account for any trailing dims that are not specified to be
    // indexed explicitly, but need to be taken care of.
    for (; i < self.dim(); i++) {
      index_all_elems[i] = true;
    }

    for (i = 0; i < self.dim(); i++) {
      repeats_needed[i] = 1;
      int64_t total_elements_above = 1;
      bool explicit_index_above = false;
      for (int j = 0; j < i; j++) {
        if ((j >= explicit_index_tensor_group_start) &&
            (j < explicit_index_tensor_group_start + explicit_index_count)) {
          explicit_index_above = true;
        } else {
          total_elements_above *= self_permuted_sizes[j];
        }
      }
      repeats_needed[i] = total_elements_above;
      if (index_all_elems[i] && explicit_index_above) {
        repeats_needed[i] *= broadcast_to_size;
      } else if (!index_all_elems[i] && (1 == index_t_sizes[i])) {
        repeats_needed[i] *= broadcast_to_size;
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
          total_elements_below *= self_permuted_sizes[j];
        }
      }
      if (index_all_elems[i] && explicit_index_below) {
        total_elements_below *= broadcast_to_size;
      }
      repeat_interleaves_needed[i] = total_elements_below;
    }

    c10::ScalarType index_dtype =
        (GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) ? c10::ScalarType::Long
                                                   : c10::ScalarType::Int);

    int explicit_index_pos = 0;
    std::vector<synapse_helpers::tensor> index_tensor_to_use;
    for (int dim = 0; dim < self.dim(); dim++) {
      at::Tensor it;
      int64_t num_elems;
      if (index_all_elems[dim]) {
        std::vector<int64_t> outshape{self_permuted_sizes[dim]};
        size_t size = 0;
        at::Stack arange_stack = {};
        num_elems = self_permuted_sizes[dim];
        auto params = FillArangeParamsInternal(
            0,
            self_permuted_sizes[dim],
            1,
            (GET_ENV_FLAG_NEW(PT_ENABLE_INT64_SUPPORT) ? c10::ScalarType::Long
                                                       : c10::ScalarType::Int),
            size);

        index_tensor_to_use.emplace_back(ArangeCommon(
            this,
            graph,
            0,
            self_permuted_sizes[dim],
            1,
            index_dtype,
            syn_in(0), // TBD: NOTE: This needs to be changed for DS
            syn_in(1), // TBD: NOTE: This needs to be changed for DS
            "range_" + habana_helpers::name_suffix_from_type(index_dtype),
            outshape,
            params,
            size,
            c10::nullopt));
      } else {
        // only explicit indices in stack and they start from pos=1.
        num_elems = indices.get(explicit_index_pos).sizes()[0];
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
                                    : syn_in(explicit_index_pos)),
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
            ReshapeHelper(graph, t_op[0].get(), reshape_outshape, ScalarType());

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
            "tile_fwd_" + habana_helpers::name_suffix_from_type(index_dtype),
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
            "tile_fwd_" + habana_helpers::name_suffix_from_type(index_dtype),
            {((index_all_elems[dim]) ? index_tensor_to_use.back().get()
                                     : syn_in(explicit_index_pos))},
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
        cat_input_synTensor,
        {{cat_out_size, scalar_type}},
        &concat_params,
        sizeof(concat_params));
    auto catop = std::move(catop1.at(0));

    auto cat_out_shape = catop.pt_shape();
    std::vector<int64_t> shape = {cat_out_shape[1]};
    auto indexOp = BuildOp(
        graph,
        "gather_nd_mxnet_fwd_" +
            habana_helpers::name_suffix_from_type(ScalarType()),
        {permuted_self_t, catop.get()},
        {{shape, ScalarType()}});
    auto final_shape = habana::ComputeOutputShapeWithAdvIndexing(
        permuted_self_shape, tensorlist, adv_index_dims, true);
    auto index_out =
        ReshapeHelper(graph, indexOp[0].get(), final_shape, ScalarType(), 0);
    syn_out(0) = std::move(index_out);
  } // end - advanced indexing present
}

// index is implemented using mxnet_gatherNd, refer below for output shape
// computation
// ref:https://github.com/apache/incubator-mxnet/blob/master/src/operator/tensor/indexing_op.h#L1319
std::vector<int64_t> ComputeIndexOperatorOutputShape(
    const at::Tensor& input,
    at::TensorList indices) {
  auto input_shape = input.sizes();
  auto indices_shape = indices_size(indices);

  if (input.dim() == 0 && input.numel() == 1)
    return {input.sizes().vec()};

  auto output_rank = static_cast<int64_t>(
      indices_shape.size() + input.ndimension() - indices_shape[0] - 1);

  std::vector<int64_t> output_shape(output_rank, -1);

  for (size_t i = 0; i < indices_shape.size() - 1; i++) {
    output_shape[i] = indices_shape[i + 1];
  }

  for (int64_t i = 0;
       i < static_cast<int64_t>(input.ndimension() - indices_shape[0]);
       i++) {
    output_shape[indices_shape.size() - 1 + i] =
        input_shape[indices_shape[0] + i];
  }
  return output_shape;
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
