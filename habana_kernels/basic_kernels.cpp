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
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;

#ifdef PT_KERNEL_END
#undef PT_KERNEL_BEGIN
#define PT_KERNEL_BEGIN (void)(0)
#undef PT_KERNEL_END
#define PT_KERNEL_END (void)(0)
#endif

// cpu->hpu and hpu->cpu copy implementation
Tensor& copy_hpu_(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_KERNEL_BEGIN;
  if (non_blocking)
    PT_KERNEL_WARN(
        "non_blocking flag is not supported, copy_hpu_ is always blocking");
  // TODO: (from torch code) this should be handled during dispatch, but that's
  // missing...
  Tensor& dst = self;
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  const auto src_device = src.device().type();
  const auto dst_device = dst.device().type();

  if (src_device == c10::DeviceType::CPU &&
      dst_device == c10::DeviceType::HABANA) {
    HABANA_ASSERT(dst.nbytes() == src.nbytes());
    habana_helpers::copy_data_to_device(src.data_ptr(), dst, src.nbytes());
  } else if (
      src_device == c10::DeviceType::HABANA &&
      dst_device == c10::DeviceType::CPU) {
    HABANA_ASSERT(dst.nbytes() == src.nbytes());
    habana_helpers::copy_data_to_host(src, dst.data_ptr(), src.nbytes());
  } else if (
      src_device == c10::DeviceType::HABANA &&
      dst_device == c10::DeviceType::HABANA) {
    if ((src.scalar_type() == c10::ScalarType::Float) &&
        (dst.scalar_type() == c10::ScalarType::BFloat16)) {
      dst = habana_helpers::hpu_cast_tensor(
          src, at::scalarTypeToTypeMeta(c10::ScalarType::BFloat16));
    } else if (
        (src.scalar_type() == c10::ScalarType::BFloat16) &&
        (dst.scalar_type() == c10::ScalarType::Float)) {
      dst = habana_helpers::hpu_cast_tensor(
          src, at::scalarTypeToTypeMeta(c10::ScalarType::Float));
    } else {
      HABANA_ASSERT(dst.nbytes() == src.nbytes());
      habana_helpers::copy_data_within_device(src, dst);
    }
  } else {
    PT_KERNEL_FATAL(
        "copy_hpu_ doesn't support ", src_device, " to ", dst_device, "copy");
  }

  if (src.strides() != dst.strides())
    PT_KERNEL_WARN(
        "src.strides(): ",
        src.strides(),
        " src.sizes(): ",
        src.sizes(),
        "\ndst.strides(): ",
        dst.strides(),
        " dst.sizes(): ",
        dst.sizes(),
        "\nData will be copied with with basic memcopy so you can expect wrong results");

  PT_KERNEL_END;
  return dst;
}

Tensor& set_hpu_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  PT_KERNEL_BEGIN;
  if (stride.data()) {
    TORCH_CHECK(size.size() == stride.size(), "inconsistent size/stride sizes");
  }

  auto scalar_type = self.scalar_type();
  auto self_ = checked_dense_tensor_unwrap(
      self, "self", 1, "_th_set_", false, DeviceType::HABANA, scalar_type);
// TODO: remove this commented section
// part of 1.5 migration related change - revert once not needed
#if 0
  auto source_ = checked_storage(
      source,
      "source",
      2,
      DeviceType::HABANA,
      at::scalarTypeToTypeMeta(scalar_type));
#else
  auto source_ = source;
#endif
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
                data_type, 0, at::habana::getHABANADeviceAllocator(), true)
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

  PT_KERNEL_END;
  return self;
}

void ToDtypeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  // This function can handle following 2 schemas only:
  // (1) to.device(Tensor self, Device device, ScalarType dtype, bool
  // non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) ->
  // (2) Tensor to.dtype(Tensor self, ScalarType dtype, bool non_blocking=False,
  // bool copy=False, MemoryFormat? memory_format=None) -> Tensor
  TORCH_CHECK(
      inputs.size() >= 5,
      "Incorrect size of inputs expected for cast operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for toDtype operator");

  if (inputs.size() == 6) {
    // Erase device information to unify subsequent code for both schemas.
    // Should be ok since we come here only for Habana device
    inputs.erase(inputs.cbegin() + 1);
  }

  auto self = inputs[0].toTensor();
  auto type = inputs[1].toScalarType();

  std::string node_type;
  if (self.dtype() == c10::ScalarType::BFloat16 &&
      type == c10::ScalarType::Float) {
    node_type = "cast_bf16_to_f32";
  } else if (
      type == c10::ScalarType::BFloat16 &&
      self.dtype() == c10::ScalarType::Float) {
    node_type = "cast_f32_to_bf16";
  } else {
    // Casts between other types are not supported for now
    HABANA_ASSERT(0);
  }

  // we do not care about last 3 entries dtype conversion, so throw them away
  inputs.pop_back();
  inputs.pop_back();
  inputs.pop_back();

  CastOperator Op(self.device().index(), node_type);
  auto& syn_arg1 = Op.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  Op.AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  p_context_->syn_inputs_[0] = std::move(syn_arg1);
  p_context_->syn_outputs_.emplace_back(std::move(Op.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(Op.GetOutputs()[0]));
}

/*************************************************************************
 * @brief Kernel implementation for memcpy, used for D2D mem transfers
 * @param self - input which needs to be transferred
 * @param dest - Destination tensor
 ************************************************************************/
void MemCopyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input arguments for Reshape Operator");
  auto self = inputs[0].toTensor();
  auto output = inputs[1].toTensor();
  p_context_->params_size_ = 0;
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, NULL, 0);
}

static auto& KernelRegistry = ::habana::KernelRegistry()
    .add("hababna_d2d_memcpy",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<MemCopyOperator>(device_id, node_type);})
    .add("aten::to",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<ToDtypeOperator>(device_id, node_type);});

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(copy_hpu_), &copy_hpu_>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)")
                .impl_unboxedOnlyKernel<
                    decltype(at::native::as_strided_tensorimpl),
                    &at::native::as_strided_tensorimpl>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::set_.source_Storage_storage_offset( Tensor(a !) self, Storage source, int storage_offset, int[] size, int[] stride = []) ->Tensor(a !)")
                .impl_unboxedOnlyKernel<decltype(set_hpu_), &set_hpu_>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema("aten::view(Tensor(a) self, int[] size) -> Tensor(a)")
                .impl_unboxedOnlyKernel<
                    decltype(at::native::view),
                    &at::native::view>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
