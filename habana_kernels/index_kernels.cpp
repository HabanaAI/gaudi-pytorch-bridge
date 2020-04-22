/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtils.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "kernel_utils.h"

using namespace torch;

/*************************************************************************
 * @brief This helper function makes the size of index tensor to be same as
 * value tensor, with broadcast of indices (within index tensor)
 ************************************************************************/
static Tensor make_index_same_size_as_value(
    const Tensor& index,
    const Tensor& value,
    int64_t dim) {
  // We use "View" + "Broadcast" so that index tensor becomes same shape as
  // value tensor with indices repeated in right pattern. This 2-step approach
  // is required because "Scatter" TPC kernel does not support Broadcast for
  // index tensor.

  // Expand 1D index tensor to same number of dimensions as value tensor
  auto expanded_sizes = std::vector<int64_t>(value.ndimension(), 1);
  expanded_sizes[dim] = index.sizes()[0];
  auto index_expanded = index.view(expanded_sizes);

  // Broadcast index tensor to same shape as value tensor
  auto index_broadcast = at::empty(DimVector(value.sizes()), index.options());
  std::vector<const at::Tensor*> pt_inputs{&index_expanded};
  std::vector<const at::Tensor*> pt_outputs{&index_broadcast};
  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "broadcast", nullptr, 0, SynapsePassType::NO_PASS);

  return index_broadcast;
}

/*************************************************************************
 * @brief Kernel implementation for scatter_.src(Tensor(a!) self, int dim,
 *Tensor index, Tensor src) -> Tensor(a!)
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param index - Tensor used to index into self
 * @param src -Tensor with values to be updated (of same type as self)
 ************************************************************************/
Tensor& scatter_inplace_src_hpu(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  ns_ScatterKernel::Params params;
  params.axis = self.dim() - dim - 1;

  std::vector<const at::Tensor*> pt_inputs{&self, &index, &src};

  synapse_simple_generic_inplace_kernel(
      pt_inputs,
      "scatter",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  return self;
}

/*************************************************************************
 * @brief Kernel implementation for index_add(dim, index, tensor) → Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - dimension along which to index
 * @param indices - Tensor used to index into self
 * @param source -Tensor with values to be updated (of same type as self)
 ************************************************************************/
Tensor& index_add_hpu_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(indices.dim() <= 1, "index tensor cannot be more than 1D")
  // Convert index tensor from 0D to 1D if required
  if (indices.dim() == 0) {
    indices.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  auto value_acc = source;
  auto slice = at::index_select(self, dim, indices);
  value_acc += slice;

  auto index_int = habana_helpers::cast_tensor_to_integer(indices);

  auto index_broadcast =
      make_index_same_size_as_value(index_int, value_acc, dim);
  self = scatter_inplace_src_hpu(self, dim, index_broadcast, value_acc);

  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for index_put(indices, value, accumulate=False)
 *→ Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param indices - Tensors used to index into self
 * @param value - Tensor with values to be updated (of same type as self)
 * @param accumulate - Flag to indicate whether to accumulate into self
 * @param unsafe -
 ************************************************************************/
Tensor& index_put_impl_hpu_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate,
    bool unsafe) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(unsafe == false, "Unsafe not supported in index_put");
  TORCH_CHECK(indices[0].dim() <= 1, "index tensor cannot be more than 1D")
  // Convert index tensor from 0D to 1D if required
  if (indices[0].dim() == 0) {
    indices[0].unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto value_acc = value;
  if (accumulate) {
    auto slice = at::index_select(self, 0, indices[0]);
    value_acc += slice;
  }

  auto index_int = habana_helpers::cast_tensor_to_integer(indices[0]);

  // Insertion of updates is always along dim=0 for this operator
  int64_t dim = 0;
  auto index_broadcast =
      make_index_same_size_as_value(index_int, value_acc, dim);
  self = scatter_inplace_src_hpu(self, dim, index_broadcast, value_acc);

  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for torch.index_select(input, dim, index) →
 *Tensor
 * @param self - Input tensor 1-4D bf16/fp32
 * @param dim - The dimension in which we index
 * @param index - 1D tensor containing the indices to index
 ************************************************************************/
Tensor index_select_hpu(const Tensor& self, int64_t dim, const Tensor& index) {
  LOG_FUNC_BEGIN;

  TORCH_CHECK(index.dim() <= 1, "index tensor cannot be more than 1D")
  // Convert index tensor from 0D to 1D if required
  if (index.dim() == 0) {
    index.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

  auto index_int = habana_helpers::cast_tensor_to_integer(index);

  auto shape = DimVector(self.sizes());
  shape.erase(shape.begin() + dim);
  shape.insert(shape.begin() + dim, index.numel());
  auto output = at::empty(shape, self.options());

  ns_GatherKernel::Params params;
  params.axis = self.dim() - dim - 1;

  std::vector<const at::Tensor*> pt_inputs{&self, &index_int};
  std::vector<const at::Tensor*> pt_outputs{&output};

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "gather",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  LOG_FUNC_END;
  return output;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::index_select(Tensor self, int dim, Tensor index) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(index_select_hpu),
                    &index_select_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(index_put_impl_hpu_),
                    &index_put_impl_hpu_>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(index_add_hpu_),
                    &index_add_hpu_>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(scatter_inplace_src_hpu),
                    &scatter_inplace_src_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
