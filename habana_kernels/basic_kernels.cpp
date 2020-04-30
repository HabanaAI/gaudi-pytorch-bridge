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
#include <synapse_api.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "kernel_utils.h"
#include "resize.h"

using namespace torch;

#ifdef LOG_FUNC_END
#undef LOG_FUNC_BEGIN
#define LOG_FUNC_BEGIN (void)(0)
#undef LOG_FUNC_END
#define LOG_FUNC_END (void)(0)
#endif

// cpu->hpu and hpu->cpu copy implementation
Tensor& copy_hpu_(Tensor& self, const Tensor& src, bool non_blocking) {
  LOG_FUNC_BEGIN;
  if (non_blocking)
    TORCH_WARN(
        "non_blocking flag is not supported, copy_hpu_ is always blocking");
  // TODO: (from torch code) this should be handled during dispatch, but that's
  // missing...
  Tensor& dst = self;
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");
  TORCH_CHECK(
      dst.nbytes() == src.nbytes(), "src and dst buffers size don't match");

  const auto src_device = src.device().type();
  const auto dst_device = dst.device().type();

  if (src_device == c10::DeviceType::CPU &&
      dst_device == c10::DeviceType::HABANA) {
    habana_helpers::copy_data_to_device(src.data_ptr(), dst, src.nbytes());
  } else if (
      src_device == c10::DeviceType::HABANA &&
      dst_device == c10::DeviceType::CPU) {
    habana_helpers::copy_data_to_host(src, dst.data_ptr(), src.nbytes());
  } else if (
      src_device == c10::DeviceType::HABANA &&
      dst_device == c10::DeviceType::HABANA) {
    habana_helpers::copy_data_within_device(src, dst);
  } else {
    TORCH_CHECK(
        false,
        "copy_hpu_ doesn't support ",
        src_device,
        " to ",
        dst_device,
        "copy");
  }

  if (src.strides() != dst.strides())
    TORCH_WARN(
        "src.strides(): ",
        src.strides(),
        " src.sizes(): ",
        src.sizes(),
        "\ndst.strides(): ",
        dst.strides(),
        " dst.sizes(): ",
        dst.sizes(),
        "\nData will be copied with with basic memcopy so you can expect wrong results");

  LOG_FUNC_END;
  return dst;
}

Tensor& set_hpu_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  LOG_FUNC_BEGIN;
  if (stride.data()) {
    TORCH_CHECK(size.size() == stride.size(), "inconsistent size/stride sizes");
  }

  auto scalar_type = self.scalar_type();
  auto self_ = checked_dense_tensor_unwrap(
      self, "self", 1, "_th_set_", false, DeviceType::HABANA, scalar_type);
  auto source_ = checked_storage(
      source,
      "source",
      2,
      DeviceType::HABANA,
      at::scalarTypeToTypeMeta(scalar_type));

  // Code below is based on THCTensor_setStorage
  TORCH_CHECK(
      self_->storage(),
      "Cannot use PyTorch operations on a half-constructed "
      "tensor.  If this tensor came from Caffe2, please call GetMutableData on "
      "it first; otherwise, this is a bug, please report it.");
  auto self_storage = self_->storage().unsafeGetStorageImpl();
  auto source_storage = source_.unsafeGetStorageImpl();
  if (self_storage != source_storage) {
    TORCH_CHECK(self_storage, "Invalid null storage");
    auto data_type = self_storage->dtype();
    if (self_storage) {
      c10::raw::intrusive_ptr::incref(source_storage);
      THTensor_stealAndSetStoragePtr(self_, source_storage);
    } else {
      auto THHStorage_new = [](caffe2::TypeMeta data_type) -> THStorage* {
        THStorage* storage =
            c10::make_intrusive<at::StorageImpl>(
                data_type, 0, habana::getHABANADeviceAllocator(), true)
                .release();
        return storage;
      };
      THTensor_stealAndSetStoragePtr(self_, THHStorage_new(data_type));
    }
  }

  TORCH_CHECK(storage_offset >= 0, "Invalid storage offset: ", storage_offset);
  self_->set_storage_offset(storage_offset);

  /* size and stride */
  THHTensor_resizeNd(self_, stride.size(), size.data(), stride.data());

  LOG_FUNC_END;
  return self;
}

void validate_tensor_dim_sizes(const TensorList tensors, int64_t dim) {
  unsigned i = 0;
  auto tensor_count = tensors.size();
  auto out_size = tensors[0].sizes().vec();

  Tensor tempT = tensors[0];
  for (i = 0; i < tensor_count; i++) {
    // check whether sizes along dimensions match except for cat dimension.
    unsigned j = 0;
    auto sz1 = tensors[i].sizes().vec();
    auto sz2 = tempT.sizes().vec();
    for (j = 0; j < tensors[i].dim(); j++) {
      if (j != dim) {
        if ((sz1[j] - sz2[j]) != 0)
          std::cout
              << "Sizes of tensors along one of the non-cat dimensions don't match"
              << std::endl;
        TORCH_CHECK(
            ((sz1[j] - sz2[j]) == 0),
            "Sizes of tensors along one of the non-cat dimensions don't match");
      }
    }
    tempT = tensors[i];
  }
}
/*************************************************************************
 * @brief Kernel implementation for torch.cat(tensors, dim)
 * @param tensors - tensor list/tuple of inputs
 * @param dim - dimension along which to concatenate the tensors
 ************************************************************************/
Tensor cat_hpu(const TensorList tensors, int64_t dim_ = 0) {
  LOG_FUNC_BEGIN;
  std::vector<const at::Tensor*> pt_inputs;
  std::vector<const at::Tensor*> pt_outputs;
  int64_t dim =
      at::maybe_wrap_dim(dim_, tensors[0].dim(), /*wrap_scalar=*/true);
  validate_tensor_dim_sizes(tensors, dim);

  auto out_size = tensors[0].sizes().vec();
  out_size[dim] = 0;
  unsigned i = 0;
  auto tensor_count = tensors.size();
  for (i = 0; i < tensor_count; i++) {
    pt_inputs.push_back(&tensors[i]);
    out_size[dim] += tensors[i].sizes()[dim];
  }
  auto out = at::empty(out_size, tensors[0].options());
  pt_outputs.push_back(&out);
  // python level cat matches dim num with the order of sizes in tensor creation
  auto kernel_dim = (out_size.size() - dim) - 1;
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "concat",
      &kernel_dim,
      sizeof(kernel_dim),
      SynapsePassType::NO_PASS);

  LOG_FUNC_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for torch.cat(tensors, dim, out=result)
 * @param result - result of concatenate
 * @param tensors - tensor list/tuple of inputs
 * @param dim - dimension along which to concatenate the tensors
 ************************************************************************/
Tensor& cat_hpu_out(
    Tensor& result,
    const TensorList tensors,
    int64_t dim_ = 0) {
  LOG_FUNC_BEGIN;
  std::vector<const at::Tensor*> pt_inputs;
  std::vector<const at::Tensor*> pt_outputs;
  int64_t dim =
      at::maybe_wrap_dim(dim_, tensors[0].dim(), /*wrap_scalar=*/true);
  validate_tensor_dim_sizes(tensors, dim);

  auto out_size = tensors[0].sizes().vec();
  out_size[dim] = 0;
  unsigned i = 0;
  auto tensor_count = tensors.size();
  for (i = 0; i < tensor_count; i++) {
    pt_inputs.push_back(&tensors[i]);
    out_size[dim] += tensors[i].sizes()[dim];
  }
  if (result.defined()) {
    TORCH_CHECK(
        tensors[0].type() == result.type(),
        "output values must be of same type as input");
    auto tht_result = result.unsafeGetTensorImpl();
    THHTensor_resizeNd(tht_result, tensors[0].dim(), out_size.data(), nullptr);
  } else {
    result = at::empty(out_size, tensors[0].options());
  }
  pt_outputs.push_back(&result);
  // python level cat matches dim num with the order of sizes in tensor creation
  auto kernel_dim = (out_size.size() - dim) - 1;
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "concat",
      &kernel_dim,
      sizeof(kernel_dim),
      SynapsePassType::NO_PASS);
  return result;
}

inline void recalc_strides(
    std::vector<int64_t>& self_strides,
    const std::vector<int64_t>& self_sizes) {
  int k;
  self_strides[self_strides.size() - 1] = 1;
  for (k = self_strides.size() - 2; k >= 0; k--) {
    self_strides[k] = self_strides[k + 1] * self_sizes[k + 1];
  }
  return;
}

/****************************************************************************
 * @brief Kernel implementation for N-D out = torch.transpose(self,dim0,dim1)
 * @param self - input
 * @param dim0 - first dimension to swap
 * @param dim0 - second dimension to swap
 ***************************************************************************/
Tensor transpose_hpu(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  LOG_FUNC_BEGIN;
  // handle negative dimensions (backward indexing) in pytorch
  int64_t dim0 = at::maybe_wrap_dim(dim0_, self.dim(), /*wrap_scalar=*/true);
  int64_t dim1 = at::maybe_wrap_dim(dim1_, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      (dim0 < self.dim()) && (dim1 < self.dim()),
      "Specified dims are beyond tensor dims");

  auto self_sizes = self.sizes().vec();
  auto self_strides = self.strides().vec();
  std::swap(self_sizes[dim0], self_sizes[dim1]);
  // Recalculate the strides to account for transpose size changes
  // In effect, keep the tensor contiguous.
  recalc_strides(self_strides, self_sizes);
  std::vector<const at::Tensor*> pt_inputs;
  std::vector<const at::Tensor*> pt_outputs;
  pt_inputs.push_back(&self);
  auto out = at::empty_strided(self_sizes, self_strides, self.options());
  pt_outputs.push_back(&out);

  synTransposeParams params;
  params.tensorDim = self.dim();
  int i;
  for (i = 0; i < MAX_DIMENSIONS_NUM; i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(
      params.permutation[self.dim() - 1 - dim0],
      params.permutation[self.dim() - 1 - dim1]);
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "transpose",
      &params,
      sizeof(synTransposeParams),
      SynapsePassType::FORWARD_PASS);
  LOG_FUNC_END;
  return out;
}

/*******************************************************************************
 * @brief Kernel implementation for N-D inplace torch.transpose_(self,dim0,dim1)
 * @param self - input as well as output
 * @param dim0 - first dimension to swap
 * @param dim0 - second dimension to swap
 *******************************************************************************/
Tensor& transpose_hpu_(Tensor& self, int64_t dim0_, int64_t dim1_) {
  LOG_FUNC_BEGIN;
  /*NOTE: The normal inplace op implementation approach to through a duplicate
   * synapse tensor for input won't work as synapse backend does block
   * transposes - so if your matrix is AB CD then C will overwrite B before B is
   * written or vice-versa. So, we do the following (inefficient way, but helps
   * to support the functionality). tempTensor = transpose_outofplace(inTensor)
   * Reshape inTensor to transposed sizes for required dims.
   * Use synapse memcpy guid to do a transfer data from tempTensor to inTensor
   * Return inTensor back to PyTorch frontend
   */
  auto tempT = transpose_hpu(self, dim0_, dim1_);

  std::vector<const at::Tensor*> pt_inputs;
  std::vector<const at::Tensor*> pt_outputs;
  pt_inputs.push_back(&tempT);

  // handle negative dimensions (backward indexing) in pytorch
  int64_t dim0 = at::maybe_wrap_dim(dim0_, self.dim(), /*wrap_scalar=*/true);
  int64_t dim1 = at::maybe_wrap_dim(dim1_, self.dim(), /*wrap_scalar=*/true);

  auto self_sizes = self.sizes().vec();
  auto self_strides = self.strides().vec();
  std::swap(self_sizes[dim0], self_sizes[dim1]);
  // Recalculate the strides to account for transpose size changes
  // In effect, keep the tensor contiguous.
  recalc_strides(self_strides, self_sizes);
  auto tht_result = self.unsafeGetTensorImpl();
  THHTensor_resizeNd(
      tht_result, self.dim(), self_sizes.data(), self_strides.data());
  pt_outputs.push_back(&self);

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "memcpy", nullptr, 0, SynapsePassType::NO_PASS);

  LOG_FUNC_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for 2D torch.t(self,dim0,dim1)
 * @param self - input
 * @param dim0 - first dimension to swap
 * @param dim0 - second dimension to swap
 ************************************************************************/
Tensor t_hpu(const Tensor& self) { // t() is defined only for dims <= 2
  LOG_FUNC_BEGIN;
  if ((1 == self.dim())) {
    Tensor out = self;
    LOG_FUNC_END;
    return out;
  }
  auto ret = transpose_hpu(self, 0, 1);
  LOG_FUNC_END;
  return ret;
}

/*************************************************************************
 * @brief Kernel implementation for 2D inplace torch.t_(self,dim0,dim1)
 * @param self - input as well as output
 * @param dim0 - first dimension to swap
 * @param dim0 - second dimension to swap
 ************************************************************************/
Tensor& t_hpu_(Tensor& self) { // t_() is defined only for dims <= 2
  LOG_FUNC_BEGIN;
  if (1 == self.dim()) {
    LOG_FUNC_END;
    return self;
  }
  self = transpose_hpu_(self, 0, 1);
  LOG_FUNC_END;

  return self;
}

Tensor permute_4d(const Tensor& self, int* dims) {
  LOG_FUNC_BEGIN;
  auto self_sizes = self.sizes().vec();
  // calculate new sizes and strides after permute for out tensor
  auto new_sizes = self.sizes().vec();
  auto new_strides = self.strides().vec();
  int i;
  new_sizes[new_sizes.size() - 1] = self_sizes[dims[new_sizes.size() - 1]];
  new_strides[new_sizes.size() - 1] = 1;
  for (i = new_strides.size() - 2; i >= 0; i--) {
    new_sizes[i] = self_sizes[dims[i]];
    new_strides[i] = new_strides[i + 1] * new_sizes[i + 1];
  }
  std::vector<const at::Tensor*> pt_inputs;
  std::vector<const at::Tensor*> pt_outputs;
  pt_inputs.push_back(&self);
  auto out = at::empty_strided(new_sizes, new_strides, self.options());
  pt_outputs.push_back(&out);

  synTransposeParams params;
  params.tensorDim = self.dim();
  // params.permute has to be populated in a reverse order for HPU FCD-LCD order
  for (i = 0; i < self.dim(); i++) {
    params.permutation[self.dim() - 1 - dims[i]] =
        static_cast<TransposePermutationDim>(self.dim() - 1 - i);
  }
  for (i = self.dim(); i < MAX_DIMENSIONS_NUM; i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "transpose",
      &params,
      sizeof(synTransposeParams),
      SynapsePassType::FORWARD_PASS);
  LOG_FUNC_END;
  return out;
}

inline int is_hpu_supported_transpose_type(const c10::ScalarType pt_type) {
  int ret = -1;
  switch (pt_type) {
    case c10::ScalarType::Float:
    case c10::ScalarType::BFloat16:
    case c10::ScalarType::Int:
      ret = 0;
      break;
    default:
      break;
  }
  return ret;
}

/*************************************************************************
 * @brief Kernel implementation for torch.Tensor.permute(dims)
 * @param self - input on which permute needs to be applied
 * @param dims_ - permute dims array
 ************************************************************************/
Tensor permute_hpu(const Tensor& self, IntArrayRef dims_) {
  LOG_FUNC_BEGIN;
  TORCH_CHECK(
      static_cast<unsigned>(dims_.size()) == self.dim(),
      "Number of dims in tensor don't match in permute");
  int dims[self.dim()];
  for (unsigned i = 0; i < self.dim(); i++) {
    dims[i] = dims_[i];
  }
  if ((self.dim() <= 4) &&
      !is_hpu_supported_transpose_type(self.scalar_type())) {
    // single transpose "permute" from synapse
    return permute_4d(self, dims);
  }
  // HPU won't support permute for larger num of dims - do it on CPU
  auto ret =
      self.to(DeviceType::CPU).permute(dims_).contiguous().to(self.device());
  return ret;
}

/*************************************************************************
 * @brief Kernel implementation for torch.Tensor.expand(*sizes)
 * @param self - input that needs to be expanded to a larger size.
 * @param dims_ - expanded dim sizes
 * NOTE: Tensor can be also expanded to a larger number of dimensions, and the
 * new ones will be appended at the front. For the new dimensions, the size
 * cannot be set to -1. We are using expand for braodcast op implementation
 * and we differ from the PyTorch expand that says "does not allocate new
 * memory, but only creates a new view on the existing tensor where a dimension
 * of size one is expanded to a larger size by setting the stride to 0. "
 ************************************************************************/
Tensor expand_hpu(const Tensor& self, IntArrayRef size, bool implicit) {
  // [expand implicit]
  // The implicit flag is set to true for any expand calls inserted by broadcast
  // operators in ExpandUtils.h This flag is recorded by the tracer to
  // distinguish between expands inserted by broadcasts and those explicitly
  // requested by the user, because it is legal to remove implicit expands
  // from the graph, but not legal to remove the explicit ones.
  // implicit is not used in this kernel.
  LOG_FUNC_BEGIN;
  TORCH_CHECK(
      size.size() >= (size_t)self.dim(),
      "expand(",
      self.toString(),
      "{",
      self.sizes(),
      "}, size=",
      size,
      "): the number of sizes provided (",
      size.size(),
      ") ",
      "must be greater or equal to the number of dimensions in the tensor (",
      self.dim(),
      ")",
      "implicit = ",
      implicit);
  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) =
      at::inferExpandGeometry(self.sizes(), self.strides(), size);

  // expandedStrides will be set to 0 by inferExpandGeometry.
  // Since we give back a contiguous tensor, we will set strides
  // to proper values.
  recalc_strides(expandedStrides, expandedSizes);
  Tensor result; //(tensors.size());
  if (self.sizes().equals(expandedSizes)) {
    result = self;
  } else {
    result = at::empty_strided(expandedSizes, expandedStrides, self.options());
    auto expanded_self_view_sizes =
        std::vector<int64_t>(expandedSizes.size(), 1);
    for (unsigned i = 0; i < self.dim(); i++) {
      expanded_self_view_sizes[expandedSizes.size() - self.dim() + i] =
          self.sizes()[i];
    }
    auto self_view =
        self.view(expanded_self_view_sizes); // prepend dims of size 1
    synapse_simple_generic_kernel(
        {&result},
        {&self_view},
        "broadcast",
        nullptr,
        0,
        SynapsePassType::NO_PASS);
  }
  LOG_FUNC_END;
  return result;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(copy_hpu_), &copy_hpu_>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)")
                .impl_unboxedOnlyKernel<
                    decltype(at::native::as_strided_tensorimpl),
                    &at::native::as_strided_tensorimpl>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)")
                .impl_unboxedOnlyKernel<decltype(permute_hpu), &permute_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::set_.source_Storage_storage_offset( Tensor(a !) self, Storage source, int storage_offset, int[] size, int[] stride = []) ->Tensor(a !)")
                .impl_unboxedOnlyKernel<decltype(set_hpu_), &set_hpu_>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::view(Tensor(a) self, int[] size) -> Tensor(a)")
                .impl_unboxedOnlyKernel<
                    decltype(at::native::view),
                    &at::native::view>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)")
                .impl_unboxedOnlyKernel<decltype(expand_hpu), &expand_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::cat(Tensor[] tensors, int dim=0) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(cat_hpu), &cat_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(cat_hpu_out), &cat_hpu_out>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::_cat(Tensor[] tensors, int dim=0) -> Tensor")
                .impl_unboxedOnlyKernel<decltype(cat_hpu), &cat_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::_cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(cat_hpu_out), &cat_hpu_out>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)")
                .impl_unboxedOnlyKernel<
                    decltype(transpose_hpu),
                    &transpose_hpu>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(transpose_hpu_),
                    &transpose_hpu_>(TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::t(Tensor(a) self) -> Tensor(a)")
                .impl_unboxedOnlyKernel<decltype(t_hpu), &t_hpu>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::t_(Tensor(a!) self) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(t_hpu_), &t_hpu_>(
                    TensorTypeId::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
