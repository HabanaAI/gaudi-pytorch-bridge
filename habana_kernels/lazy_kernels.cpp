/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <ATen/InferSize.h>
#include "habana_helpers/logging.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/conv_kernels.h"
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/embedding_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/pool_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/ops/cast_ops.h"
#include "habana_lazy/ops/cat.h"
#include "habana_lazy/ops/constant.h"
#include "habana_lazy/ops/convolution.h"
#include "habana_lazy/ops/embedding_bag.h"
#include "habana_lazy/ops/index.h"
#include "habana_lazy/ops/loss.h"
#include "habana_lazy/ops/mse_loss.h"
#include "habana_lazy/ops/norm.h"
#include "habana_lazy/ops/pool.h"
#include "habana_lazy/ops/reduce_ops.h"
#include "habana_lazy/ops/shape_ops.h"
#include "habana_lazy/ops/softmax.h"
#include "habana_lazy/ops/tensor_shape.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/synapse_helpers/util.h"

#include "habana_lazy/ops/optimizer_sparse_sgd_with_valid_count.h"
at::Tensor preProcessIfLongorDouble(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool& processed) {
  at::Tensor processed_tensor_cpu;
  c10::ScalarType old_type = src.scalar_type();
  c10::ScalarType new_type = src.scalar_type();
  // We need to cast data on CPU before copying if there is some unsupported
  // type
  if (src.scalar_type() == c10::ScalarType::Long) {
    processed_tensor_cpu = src.to(c10::ScalarType::Int);
    processed = true;
    old_type = c10::ScalarType::Long;
    new_type = c10::ScalarType::Int;
  } else if (src.scalar_type() == c10::ScalarType::Double) {
    processed_tensor_cpu = src.to(c10::ScalarType::Float);
    processed = true;
    old_type = c10::ScalarType::Double;
    new_type = c10::ScalarType::Float;
  }
  if (processed) {
    auto hl_tensor = habana_lazy::GetOrCreateHbLazyTensor(dst, dst.device());
    hl_tensor.setTensorOriginalType(old_type);
    hl_tensor.SetScalarType(c10::make_optional(new_type));
  }
  return processed_tensor_cpu;
}

Tensor& copy_hpu_lazy_D2D(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node;
  std::vector<at::Tensor> input_pt_vec;
  habana_lazy::HbLazyTensor hb_tensor =
      habana_lazy::GetOrCreateHbLazyTensor(src, src.device());
  auto hlresult = habana_lazy::GetHbLazyTensor(self);
  if (src.dtype() == self.dtype()) {
    // If both src and dst are already processed ,  go and do the DMA dont wait
    // Else , If we already have storage in dst, add memcopy node to lazy
    // graph and we want to copy to existing tensor and not a new one
    // Kernel expects us to pass dst as second input in that case
    auto result_data = hlresult.CurrentTensorData();
    auto src_data = hb_tensor.CurrentTensorData();
    if (hlresult.isStorageAttached()) {
      node = habana_lazy::ir::Node::Create(
          Symbol::fromQualString("hpu::habana_d2d_memcpy_other"),
          {hb_tensor.GetIrValue(), hlresult.GetIrValue()});
      input_pt_vec.push_back(src);
      input_pt_vec.push_back(self);
      auto context =
          habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
              self.device().index());
      context->MarkTensorRegistered(hlresult.getTensorUniqueId());
      habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
      out.m_index = 0;
      out.SetNode(node);
      node->AddInputPtTensors(input_pt_vec);
    } else {
      node = habana_lazy::ir::Node::Create(
          Symbol::fromQualString("hpu::habana_d2d_memcpy"),
          {hb_tensor.GetIrValue()});
      input_pt_vec.push_back(src);
      habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
      out.m_index = 0;
      out.SetNode(node);
      node->AddInputPtTensors(input_pt_vec);
    }
  } else {
    node = habana_lazy::ir::Node::Create(
        Symbol::fromQualString("hpu::habana_d2d_memcpy"),
        {hb_tensor.GetIrValue()});
    auto hlresult = habana_lazy::GetHbLazyTensor(self);
    habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
    out.m_index = 0;
    out.SetNode(node);
    std::vector<at::Tensor> input_pt_vec{src};
    node->AddInputPtTensors(input_pt_vec);
    return self;
  }
  return self;
}

Tensor& copy_hpu_lazy_D2H(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;
  // If src is a lazy tensor make sure the execution till the point of src
  // getting flled has finished before we start copying
  if (habana_lazy::IsHbLazyTensor(src)) {
    habana_lazy::HbLazyTensor hb_tensor =
        habana_lazy::GetOrCreateHbLazyTensor(src, src.device());
    auto tensor_data = hb_tensor.GetHbLazyTensorData();

    TORCH_CHECK(
        tensor_data, "Trying to copy from lazy tensor with no backend memory");
    auto type = hb_tensor.getTensorOriginalType();
    // This path is disabled for now, when we return back from Habana to
    // CPU we can check if the original tensor was long/double , if soe we
    // can upscale it and send it back. For now we just send the 32bit
    // tensor that Habana holds

    if (type != typeMetaToScalarType(src.dtype())) {
      // If we need to upscale the CPU tensor using the .to for now
      // It rebinds the self reference to the new tensor
      // We need to check the memory deletion of the original tensor created
      // by PT
      self = copy_hpu_(self, tensor_data.value(), non_blocking);
      self = self.to(type);
    } else {
      self = copy_hpu_(self, tensor_data.value(), non_blocking);
    }
    self = habana_lazy::CreateHbLazyTensor(
        self, habana_lazy::GetHblazyDevice(self));
  } else {
    // This situation should not occur
    // Throwing an exception here for now to catch any cases that arise
    TORCH_CHECK(
        false,
        "Habana Lazy : trying to copy back a tensor which does not have a lazy tensor");
  }
  return self;
}

Tensor& copy_hpu_lazy_H2D(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;
  bool processed = false;
  // This tensor would have been created without storage(as all H2D .to calls
  // come via lazy), so create actual memory and set as input and mark
  // executed
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      self.device().index());
  auto exec_mode = context->getExecutionMode();
  if (exec_mode != kLOWERING) {
    auto self_hb_tensor = habana_lazy::GetHbLazyTensor(self);
    // WE need to add storage if it wasnt created
    // right now as soon as we do a H2D transfer, we create memory and mark
    // executed
    auto isStorageAttached = self_hb_tensor.isStorageAttached();
    if (!isStorageAttached) {
      c10 ::Allocator* allocator;
      allocator = at::habana::getHABANADeviceAllocator();
      int64_t nelements = prod_intlist(self.sizes());
      int elem_size = self.dtype().itemsize();
      auto storage_impl = c10::make_intrusive<StorageImpl>(
          self.dtype(),
          nelements,
          allocator->allocate(nelements * elem_size),
          allocator,
          /*resizeable=*/true);
      Tensor at_internal_tensor =
          habana_lazy::AtenInternalHbTensor(std::move(storage_impl));
      // Setup the tensor sizes/strides, for now assuming contiguous
      at_internal_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(
          self.sizes());
      self_hb_tensor.SetTensorData(at_internal_tensor);
    }
  }
  auto new_tensor = preProcessIfLongorDouble(src, self, processed);

  // Get the internal tensor for copy kernel
  // First get the lazy tensor
  auto self_hb_tensor = habana_lazy::GetHbLazyTensor(self);
  // We need to mark this tensor as executed
  // As this will be an input coming from host side, its doesnt need further
  // execution and is ready for consumption as input
  auto self_hb_tensor_data = self_hb_tensor.GetHbLazyTensorData();
  // This is the internal tensor, it isn't a lazy tensor
  auto self_internal_tesor = self_hb_tensor_data.value();
  HABANA_ASSERT(!habana_lazy::TryGetHbLazyTensor(self_internal_tesor));
  if (processed) {
    auto internal_tensor_from_copy =
        copy_hpu_(self_internal_tesor, new_tensor, non_blocking);
    // We should get back the same internal tensor passed to copy
    HABANA_ASSERT(
        self_internal_tesor.storage().data_ptr() ==
        internal_tensor_from_copy.storage().data_ptr());
  } else {
    auto internal_tensor_from_copy =
        copy_hpu_(self_internal_tesor, src, non_blocking);
    // We should get back the same internal tensor passed to copy
    HABANA_ASSERT(
        self_internal_tesor.storage().data_ptr() ==
        internal_tensor_from_copy.storage().data_ptr());
  }
  setTensorAsInputNode(self_hb_tensor);
  context->MarkTensorStatus(
      self_hb_tensor.getTensorUniqueId(), LazyTensorExecutionStatus::kINPUT);

  // Return the self tensor, as copy_hpu_ doesn't create a new tensor and
  // returns the dst
  return self;
}

// calling eager mode kernels as a temporary placeholder to avoid warnings
Tensor& copy_hpu_lazy_(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_TRACE;
  TORCH_CHECK(self.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  const auto src_device = src.device().type();
  const auto dst_device = self.device().type();

  bool is_d2d_copy = false;
  if (src_device == c10::DeviceType::HABANA &&
      dst_device == c10::DeviceType::HABANA) {
    is_d2d_copy = true;
  }
  if (habana_lazy::IsHbLazyTensor(src) && !is_d2d_copy) {
    auto src_hb_tensor = habana_lazy::GetHbLazyTensor(src);
    auto src_hb_tensor_data = src_hb_tensor.GetHbLazyTensorData();
    if (!src_hb_tensor_data) {
      TORCH_CHECK(
          false,
          "Habana copy_hpu_lazy_: no storage tensor attached for copy lazy source");
    }
    if (!src_hb_tensor_data.value().has_storage()) {
      TORCH_CHECK(
          false,
          "Habana copy_hpu_lazy_: trying to copy from a storage less lazy tensor");
    }
  } else if (habana_lazy::IsHbLazyTensor(src) && !is_d2d_copy) {
    auto src_hb_tensor = habana_lazy::GetHbLazyTensor(src);
    TORCH_CHECK(
        src_hb_tensor.isStorageAttached(),
        "Habana copy_hpu_lazy_: trying to copy from a storage less tensor");
  }

  // If it isnt a device to device copy, we are transferring data to and
  // from CPU. This becomes an execution step point and we need to flush
  // graph execution NOW to generate tensor data where required as we are in
  // lazy mode. Otherwise we have to add the copy induced nodes(like cast)
  // to lazy graph for execution later
  if (!is_d2d_copy) {
    if (src_device == c10::DeviceType::CPU) {
      self = copy_hpu_lazy_H2D(self, src, non_blocking);
    } else if (src_device == c10::DeviceType::HABANA) {
      self = copy_hpu_lazy_D2H(self, src, non_blocking);
    }
  } else {
    self = copy_hpu_lazy_D2D(self, src, non_blocking);
  }

  return self;
}

Tensor emtpy_from_storage_lazy(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    c10::optional<int64_t> storage_offset) {
  auto hb_tensor_self = habana_lazy::GetHbLazyTensor(self);
  TORCH_CHECK(
      hb_tensor_self.isStorageAttached(),
      "Habana Lazy : we dont support as_strided for non storage");
  auto storage_impl = hb_tensor_self.getAttachedTensorImpl();
  Tensor at_internal_tensor = habana_lazy::AtenInternalHbTensor(
      std::move(c10::Storage(storage_impl->storage())));

  at_internal_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  c10::IntArrayRef stride_new;
  if (!stride.has_value()) {
    at_internal_tensor.unsafeGetTensorImpl()->empty_tensor_restride(
        at_internal_tensor.suggest_memory_format());
    stride_new = at_internal_tensor.strides();
  } else {
    stride_new = stride.value();
  }
  // Setup the tensor sizes/strides, for now assuming contiguous
  at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
      size, stride_new);
  if (storage_offset)
    at_internal_tensor.unsafeGetTensorImpl()->set_storage_offset(
        storage_offset.value());

  Tensor at_tensor;
  bool is_in_lowering_mode = false;
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      self.device().index());
  if (context != nullptr) {
    auto exec_mode = context->getExecutionMode();
    is_in_lowering_mode = exec_mode == kLOWERING ? true : is_in_lowering_mode;
  }

  // This call could have come from a .to call and not from a lowering
  // context. In such case, create the lazt tensor.
  if (!is_in_lowering_mode) {
    habana_lazy::HbLazyTensor hb_tensor =
        habana_lazy::HbLazyTensor::CreateHbLazyTensor(
            size, 0, self.device(), c10::typeMetaToScalarType(self.dtype()));

    at_tensor = habana_lazy::AtenFromHbLazyTensor(hb_tensor);

    // The lazy tensor will have a reference to the internal tensor
    hb_tensor.SetTensorData(at_internal_tensor);

    // Setup the tensor sizes/strides, for now assuming contiguous
    at_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride_new);
    if (storage_offset)
      at_tensor.unsafeGetTensorImpl()->set_storage_offset(
          storage_offset.value());

    // Keep a pointer to the storageless tensor from the internal tensor
    auto at_internal_impl =
        habana_lazy::GetHbInternalTensorImpl(at_internal_tensor);
    HABANA_ASSERT(at_internal_impl != nullptr);
    // We need to mark this tensor as executed
    // As this will be an input coming from host side, its doesnt need further
    // execution and is ready for consumption as input
    // context->MarkTensorExecuted(hb_tensor.getTensorUniqueId());
    // This lazy tensor is newly created and should have the ir_value
    // pointing to a hpu::input
    // setTensorAsInputNode(hb_tensor);
  }
  // If we are not from lowering context, return the storageless one.
  if (!is_in_lowering_mode) {
    return at_tensor;
  } else {
    // else return the internal tensor with storage
    return at_internal_tensor;
  }
}

Tensor as_strided_hpu_lazy(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  auto hb_tensor = habana_lazy::GetHbLazyTensor(self);
  auto src_data = hb_tensor.CurrentTensorData();
  if (size.vec().size() == 1 && stride.vec()[0] == 1) {
    auto result = emtpy_from_storage_lazy(
        self, size, c10::make_optional(stride), storage_offset);
    auto hb_result = habana_lazy::GetHbLazyTensor(result);
    auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
        self.device().index());
    // auto status =
    //    context->getTensorExecutionStatus(hb_tensor.getTensorUniqueId());
    // status = (status == kINPUT || status == kEXECUTION_COMPLETE) ? kINPUT
    //                                                             :
    //                                                             kREGISTERED;
    context->MarkTensorStatus(
        hb_result.getTensorUniqueId(), kEXECUTION_COMPLETE);
    setTensorAsInputNode(hb_result);
    return result;
  } else {
    TORCH_CHECK(
        false,
        "Habana Lazy : we dont support strided tensors for non 1D/stride != 1");
    return self;
  }
  return self;
};
Tensor& set_hpu_lazy_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  HABANA_ASSERT(0);
  return set_hpu_(self, source, storage_offset, size, stride);
};
Tensor view_hpu_lazy(const Tensor& self, IntArrayRef size) {
  PT_LAZY_TRACE;
  // Make the size non zero if -1 is used
  // Make sure it points to
  // /aten/src/ATen/InferSize.h
  // Header file mismatch can point it to other variant which is not correct
  // Did not duplicate code from aten for maintenance.
  auto inferred_size = at::infer_size(size, static_cast<int64_t>(self.numel()));
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::View>(self, inferred_size);
  // View is internally handled as reshape and we get a new tensor as output
  auto result = at::native::empty_hpu_lazy(
      inferred_size, self.options(), self.suggest_memory_format(), false);

  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};
Tensor addcmul_hpu_lazy(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  HABANA_ASSERT(0);
  return addcmul_hpu(self, tensor1, tensor2, alpha);
};
Tensor& addcmul_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_LAZY_TRACE;
  if (!tensor1.is_same(tensor2)) {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto hl_tensor1 =
        habana_lazy::GetOrCreateHbLazyTensor(tensor1, c10::kHABANA);
    auto hl_tensor2 =
        habana_lazy::GetOrCreateHbLazyTensor(tensor2, c10::kHABANA);
    auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

    auto node = habana_lazy::ir::Node::Create(
        Symbol::fromQualString("aten::addcmul_"),
        {hl_self.GetIrValue(),
         hl_tensor1.GetIrValue(),
         hl_tensor2.GetIrValue(),
         hl_alpha});

    habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
    out.m_index = 0;
    out.SetNode(node);

    std::vector<at::Tensor> input_pt_vec{self, tensor1, tensor2};
    node->AddInputPtTensors(input_pt_vec);
    // As its an inplace op and we want this op to execute
    // we want to wind back status of this tensor to registered
    // so that when post order is created, we actually execute it
    auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
        self.device().index());
    context->MarkTensorStatus(
        hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);
  } else {
    // implement addcmul_ as add_(pow(tensor1,2), alpha)
    auto temp = pow_tensor_scalar_hpu_lazy(tensor1, 2.0);
    add_tensor_hpu_lazy_(self, temp, alpha);
  }

  return self;
};
Tensor addcdiv_hpu_lazy(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  HABANA_ASSERT(0);
  return addcdiv_hpu(self, tensor1, tensor2, alpha);
};
Tensor& addcdiv_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_tensor1 = habana_lazy::GetOrCreateHbLazyTensor(tensor1, c10::kHABANA);
  auto hl_tensor2 = habana_lazy::GetOrCreateHbLazyTensor(tensor2, c10::kHABANA);
  auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::addcdiv_"),
      {hl_self.GetIrValue(),
       hl_tensor1.GetIrValue(),
       hl_tensor2.GetIrValue(),
       hl_alpha});

  habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self, tensor1, tensor2};
  node->AddInputPtTensors(input_pt_vec);
  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      self.device().index());
  context->MarkTensorStatus(
      hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);
  // context->MarkTensorRegistered(hl_self.getTensorUniqueId());

  return self;
};

Tensor add_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(other, c10::kHABANA);
  auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::add"),
      {hl_self.GetIrValue(), hl_other.GetIrValue(), hl_alpha});
  auto shape_out = BinaryOperator::compute_output_shape(self, other);
  auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self, other};
  node->AddInputPtTensors(input_pt_vec);

  return result;
}

Tensor add_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  HABANA_ASSERT(0);
  return add_scalar_hpu(self, other, alpha);
};
Tensor& add_scalar_hpu_lazy_(Tensor& self, Scalar other, Scalar alpha) {
  HABANA_ASSERT(0);
  return add_scalar_hpu_(self, other, alpha);
};

Tensor& add_tensor_hpu_lazy_(Tensor& self, const Tensor& other, Scalar alpha) {
  PT_LAZY_TRACE;
  auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

  if (other.device().type() == c10::DeviceType::CPU) {
    if (other.scalar_type() == c10::ScalarType::Double) {
      // Convert 0-dim CPU tensor to a scalar and then add to JIT graph
      auto val = other.item();
      auto hl_other = habana_lazy::GetIrValueForScalar(val);
      auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);

      auto node = habana_lazy::ir::Node::Create(
          Symbol::fromQualString("aten::add_"),
          {hl_self.GetIrValue(), hl_other, hl_alpha});

      habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
      out.m_index = 0;
      out.SetNode(node);

      std::vector<at::Tensor> input_pt_vec{self};
      node->AddInputPtTensors(input_pt_vec);
      // As its an inplace op and we want this op to execute
      // we want to wind back status of this tensor to registered
      // so that when post order is created, we actually execute it
      auto context =
          habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
              self.device().index());
      // context->MarkTensorRegistered(hl_self.getTensorUniqueId());
      context->MarkTensorStatus(
          hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);
    }
  } else {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(other, c10::kHABANA);

    auto node = habana_lazy::ir::Node::Create(
        Symbol::fromQualString("aten::add_"),
        {hl_self.GetIrValue(), hl_other.GetIrValue(), hl_alpha});

    habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
    out.m_index = 0;
    out.SetNode(node);

    std::vector<at::Tensor> input_pt_vec{self, other};
    node->AddInputPtTensors(input_pt_vec);
    // As its an inplace op and we want this op to execute
    // we want to wind back status of this tensor to registered
    // so that when post order is created, we actually execute it
    auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
        self.device().index());
    // context->MarkTensorRegistered(hl_self.getTensorUniqueId());
    context->MarkTensorStatus(
        hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);
  }

  return self;
};

Tensor sub_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  HABANA_ASSERT(0);
  return sub_tensor_hpu(self, other, alpha);
};
Tensor& sub_tensor_hpu_lazy_(Tensor& self, const Tensor& other, Scalar alpha) {
  HABANA_ASSERT(0);
  return sub_tensor_hpu_(self, other, alpha);
};
Tensor sub_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  HABANA_ASSERT(0);
  return sub_scalar_hpu(self, other, alpha);
};
Tensor& sub_scalar_hpu_lazy_(Tensor& self, Scalar other, Scalar alpha) {
  HABANA_ASSERT(0);
  return sub_scalar_hpu_(self, other, alpha);
};
Tensor rsub_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  HABANA_ASSERT(0);
  return rsub_scalar_hpu(self, other, alpha);
};

Tensor& mul_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  if (other.device().type() == c10::DeviceType::CPU) {
    if (other.scalar_type() == c10::ScalarType::Double) {
      // Convert 0-dim CPU tensor to a scalar and then add to JIT graph
      auto val = other.item();
      auto hl_other = habana_lazy::GetIrValueForScalar(val);
      auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);

      auto node = habana_lazy::ir::Node::Create(
          Symbol::fromQualString("aten::mul_"),
          {hl_self.GetIrValue(), hl_other});

      habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
      out.m_index = 0;
      out.SetNode(node);

      std::vector<at::Tensor> input_pt_vec{self};
      node->AddInputPtTensors(input_pt_vec);
      // As its an inplace op and we want this op to execute
      // we want to wind back status of this tensor to registered
      // so that when post order is created, we actually execute it
      auto context =
          habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
              self.device().index());
      // context->MarkTensorRegistered(hl_self.getTensorUniqueId());
      context->MarkTensorStatus(
          hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);
    }
  } else {
    auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
    auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(other, c10::kHABANA);

    auto node = habana_lazy::ir::Node::Create(
        Symbol::fromQualString("aten::mul_"),
        {hl_self.GetIrValue(), hl_other.GetIrValue()});

    habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
    out.m_index = 0;
    out.SetNode(node);

    std::vector<at::Tensor> input_pt_vec{self, other};
    node->AddInputPtTensors(input_pt_vec);
    // As its an inplace op and we want this op to execute
    // we want to wind back status of this tensor to registered
    // so that when post order is created, we actually execute it
    auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
        self.device().index());
    // context->MarkTensorRegistered(hl_self.getTensorUniqueId());
    context->MarkTensorStatus(
        hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);
  }

  return self;
};

Tensor mul_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(other, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::mul"),
      {hl_self.GetIrValue(), hl_other.GetIrValue()});
  auto shape_out = BinaryOperator::compute_output_shape(self, other);
  auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self, other};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};
Tensor mul_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  HABANA_ASSERT(0);
  return mul_scalar_hpu(self, other);
};
Tensor& mul_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  HABANA_ASSERT(0);
  return mul_scalar_hpu_(self, other);
};
Tensor div_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  HABANA_ASSERT(0);
  return div_tensor_hpu(self, other);
};
Tensor& div_tensor_hpu_lazy_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  HABANA_ASSERT(0);
  return div_tensor_hpu_out(result, self, other);
};
Tensor& div_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  HABANA_ASSERT(0);
  return div_tensor_hpu_(self, other);
};
Tensor div_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  HABANA_ASSERT(0);
  return div_scalar_hpu(self, other);
};
Tensor& div_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetIrValueForScalar(other);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::div_"), {hl_self.GetIrValue(), hl_other});

  habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      self.device().index());
  // context->MarkTensorRegistered(hl_self.getTensorUniqueId());
  context->MarkTensorStatus(
      hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);

  return self;
};

Tensor pow_tensor_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  HABANA_ASSERT(0);
  return pow_tensor_tensor_hpu(self, other);
};

Tensor& pow_tensor_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  HABANA_ASSERT(0);
  return pow_tensor_tensor_hpu_(self, other);
};

Tensor pow_tensor_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetIrValueForScalar(other);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::pow"), {hl_self.GetIrValue(), hl_other});
  auto result = at::native::empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};

Tensor& pow_tensor_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  HABANA_ASSERT(0);
  return pow_tensor_scalar_hpu_(self, other);
};
Tensor pow_scalar_tensor_hpu_lazy(Scalar other, const Tensor& self) {
  HABANA_ASSERT(0);
  return pow_scalar_tensor_hpu(other, self);
};
Tensor gt_hpu_lazy(Tensor& self, Tensor& other) {
  HABANA_ASSERT(0);
  return gt_hpu(self, other);
};

void eq_tensor_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    const Tensor& other) {
  HABANA_ASSERT(0);
  return eq_tensor_out_hpu(output, self, other);
};
Tensor eq_tensor_hpu_lazy(Tensor& self, Tensor& other) {
  HABANA_ASSERT(0);
  return eq_tensor_hpu(self, other);
};
Tensor eq_tensor_scalar_hpu_lazy(Tensor& self, Scalar other) {
  HABANA_ASSERT(0);
  return eq_tensor_scalar_hpu(self, other);
};
Tensor lt_scalar_hpu_lazy(Tensor& self, Scalar other) {
  HABANA_ASSERT(0);
  return lt_scalar_hpu(self, other);
};
Tensor lt_tensor_hpu_lazy(Tensor& self, Tensor& other) {
  HABANA_ASSERT(0);
  return lt_tensor_hpu(self, other);
};
Tensor convolution_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr conv_node =
      std::make_shared<habana_lazy::ir::Convolution>(
          input,
          weight,
          bias,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups);

  // shape inference expects weights in HWCK irrespective of memory format
  // HWCK weights layout need to set in the user script
  // shape in NCHW/NHWC depending on memory format layout
  auto memory_format = input.suggest_memory_format();
  auto shape_out = ConvOperator::compute_output_shape(
      input.sizes().vec(),
      weight.sizes().vec(),
      padding.vec(),
      stride.vec(),
      false,
      memory_format);

  auto result = at::native::empty_hpu_lazy(
      shape_out, input.options(), memory_format, false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(conv_node);

  return result;
};

std::tuple<Tensor, Tensor, Tensor> convolution_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    std::array<bool, 3> output_mask) {
  PT_LAZY_TRACE;
  std::vector<bool> output_mask_vec(output_mask.begin(), output_mask.end());
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Convolution>(
          grad_output,
          input,
          weight,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups,
          output_mask_vec);

  // output shape inference
  c10::MemoryFormat memory_format =
      habana_helpers::get_memory_format({&input, &grad_output, &weight});
  auto grad_weight = at::native::empty_hpu_lazy(
      weight.sizes(), grad_output.options(), memory_format, false);
  auto grad_input = at::native::empty_hpu_lazy(
      input.sizes(), grad_output.options(), memory_format, false);
  auto grad_bias = at::native::empty_hpu_lazy(
      {grad_output.size(1)}, grad_output.options(), memory_format, false);

  auto hl_grad_input = habana_lazy::GetHbLazyTensor(grad_input);
  auto hl_grad_weight = habana_lazy::GetHbLazyTensor(grad_weight);
  auto hl_grad_bias = habana_lazy::GetHbLazyTensor(grad_bias);

  habana_lazy::ir::Value& value_grad_input = hl_grad_input.CurrentIrValue();
  value_grad_input.m_index = 0;
  value_grad_input.SetNode(node);

  habana_lazy::ir::Value& value_grad_weight = hl_grad_weight.CurrentIrValue();
  value_grad_weight.m_index = 1;
  value_grad_weight.SetNode(node);

  habana_lazy::ir::Value& value_grad_bias = hl_grad_bias.CurrentIrValue();
  value_grad_bias.m_index = 2;
  value_grad_bias.SetNode(node);

  auto conv_out = std::make_tuple(grad_input, grad_weight, grad_bias);

  return conv_out;
};
std::tuple<Tensor, Tensor, Tensor, Tensor> embedding_bag_hpu_lazy(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    bool scale_grad_by_freq,
    int64_t mode,
    UNUSED bool sparse,
    Tensor& per_sample_weights,
    UNUSED bool include_last_offset) {
  HABANA_ASSERT(0);
  return embedding_bag_hpu(
      weight,
      indices,
      offsets,
      scale_grad_by_freq,
      mode,
      sparse,
      per_sample_weights,
      include_last_offset);
};
Tensor embedding_bag_bwd_hpu_lazy(
    Tensor& grad,
    Tensor& indices,
    Tensor& offsets,
    UNUSED Tensor& offset2bag,
    UNUSED Tensor& bag_size,
    UNUSED Tensor& maximum_indices,
    int num_weights,
    bool scale_grad_by_freq,
    int mode,
    Tensor per_sample_weights) {
  HABANA_ASSERT(0);
  return embedding_bag_bwd_hpu(
      grad,
      indices,
      offsets,
      offset2bag,
      bag_size,
      maximum_indices,
      num_weights,
      scale_grad_by_freq,
      mode,
      per_sample_weights);
};
Tensor constant_pad_hpu_lazy(
    const Tensor& self,
    IntArrayRef pad,
    Scalar value) {
  HABANA_ASSERT(0);
  return constant_pad_hpu(self, pad, value);
};
Tensor embedding_hpu_lazy(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  HABANA_ASSERT(0);
  return embedding_hpu(
      weight, indices, padding_idx, scale_grad_by_freq, sparse);
};
Tensor embedding_dense_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  HABANA_ASSERT(0);
  return embedding_dense_backward_hpu(
      grad, indices, num_weights, padding_idx, scale_grad_by_freq);
};
Tensor embedding_bag_sum_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::EmbeddingBagSum>(
          input, indices, offsets, valid_count, kernel_mode);

  auto result = at::native::empty_hpu_lazy(
      {offsets.sizes()[0] - 1, input.size(1)},
      input.options(),
      input.suggest_memory_format(),
      false);

  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};
Tensor embedding_bag_sum_fwd_hpu_lazy(
    const Tensor& input,
    const Tensor& indices_fwd,
    const Tensor& offsets_fwd,
    const Tensor& valid_count,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd,
    const Tensor& grad_weight) {
  PT_LAZY_TRACE;
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::embedding_bag_sum_fwd"), {});

  std::vector<habana_lazy::HbLazyTensor> hl_tensors;
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(indices_fwd, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(offsets_fwd, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(valid_count, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(indices_bwd, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(offsets_bwd, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(grad_weight, c10::kHABANA));

  for (auto& i : hl_tensors) {
    node->AddInput(i.GetIrValue());
  }

  auto result = at::native::empty_hpu_lazy(
      {offsets_fwd.numel() - 1, input.size(1)},
      input.options(),
      input.suggest_memory_format(),
      false);

  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{
      input,
      indices_fwd,
      offsets_fwd,
      valid_count,
      indices_bwd,
      offsets_bwd,
      grad_weight};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};
Tensor& embedding_bag_sum_bwd_out_hpu_lazy(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd) {
  PT_LAZY_TRACE;
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::embedding_bag_sum_bwd.out"), {});

  std::vector<habana_lazy::HbLazyTensor> hl_tensors;
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(indices_bwd, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(offsets_bwd, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(valid_count_bwd, c10::kHABANA));

  for (auto& i : hl_tensors) {
    node->AddInput(i.GetIrValue());
  }

  auto hlresult = habana_lazy::GetHbLazyTensor(out);
  habana_lazy::ir::Value& out_value = hlresult.CurrentIrValue();
  out_value.m_index = 0;
  out_value.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{
      input, indices_bwd, offsets_bwd, valid_count_bwd};
  node->AddInputPtTensors(input_pt_vec);

  return out;
};
Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_lazy(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::EmbeddingBagSumBwd>(
          out, input, indices, offsets, valid_count, kernel_mode);

  auto hlresult = habana_lazy::GetHbLazyTensor(out);
  habana_lazy::ir::Value& out_value = hlresult.CurrentIrValue();
  out_value.m_index = 0;
  out_value.SetNode(node);
  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      out.device().index());
  context->MarkTensorRegistered(hlresult.getTensorUniqueId());
  return out;
};
Tensor& fill_hpu_lazy_(Tensor& self, Scalar value) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_alpha = habana_lazy::GetIrValueForScalar(value);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::fill_"), {hl_self.GetIrValue(), hl_alpha});

  habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);
  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      self.device().index());
  // context->MarkTensorRegistered(hl_self.getTensorUniqueId());
  context->MarkTensorStatus(
      hl_self.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);

  return self;
};
Tensor& masked_fill_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  HABANA_ASSERT(0);
  return masked_fill_hpu_(self, mask, value);
};
Tensor& masked_fill_scalar_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    Scalar value) {
  HABANA_ASSERT(0);
  return masked_fill_scalar_hpu_(self, mask, value);
};
Tensor gather_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad) {
  HABANA_ASSERT(0);
  return gather_src_hpu(self, dim_, index, sparse_grad);
};
Tensor& scatter_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  HABANA_ASSERT(0);
  return scatter_inplace_src_hpu(self, dim_, index, src);
};
Tensor scatter_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  HABANA_ASSERT(0);
  return scatter_src_hpu(self, dim_, index, src);
};
Tensor scatter_add_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  HABANA_ASSERT(0);
  return scatter_add_src_hpu(self, dim_, index, src);
};
Tensor& scatter_add_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  HABANA_ASSERT(0);
  return scatter_add_inplace_src_hpu(self, dim_, index, src);
};
Tensor& index_add_hpu_lazy_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source) {
  PT_LAZY_TRACE;
  auto node =
      std::make_shared<habana_lazy::ir::IndexAdd_>(self, dim_, indices, source);

  auto hl_result = habana_lazy::GetHbLazyTensor(self);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self, indices, source};
  node->AddInputPtTensors(input_pt_vec);

  return self;
};
Tensor index_put_hpu_lazy(
    const Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  HABANA_ASSERT(0);
  return index_put_hpu(self, indices, value, accumulate);
};
Tensor& index_put_hpu_lazy_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  HABANA_ASSERT(0);
  return index_put_hpu_(self, indices, value, accumulate);
};
Tensor index_select_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<habana_lazy::ir::IndexSelect>(self, dim, index);

  auto result = index_select_hpu(self, dim, index);

  auto hl_result = habana_lazy::GetHbLazyTensor(result);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self, index};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};
Tensor gather2d_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  HABANA_ASSERT(0);
  return gather2d_hpu(input, indices, validCount);
};
Tensor slice_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  PT_LAZY_TRACE;
  if (self.dim() <= 1 && step == 1) {
    return at::native::slice(self, dim, start, end, step);
  }
  auto node =
      std::make_shared<habana_lazy::ir::Slice>(self, dim, start, end, step);

  auto result = slice_hpu(self, dim, start, end, step);

  auto hl_result = habana_lazy::GetHbLazyTensor(result);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};

Tensor select_hpu_lazy(const Tensor& self, int64_t dim, int64_t index) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<habana_lazy::ir::Slice>(self, dim, index);

  // infer shape
  auto result = select_hpu(self, dim, index);

  auto hl_result = habana_lazy::GetHbLazyTensor(result);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};

Tensor& arange_hpu_lazy(Tensor& output, Scalar start, Scalar end, Scalar step) {
  HABANA_ASSERT(0);
  return arange_hpu(output, start, end, step);
};
Tensor mm_hpu_lazy(const at::Tensor& mat1, const at::Tensor& mat2) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(mat1, c10::kHABANA);
  auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(mat2, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::mm"),
      {hl_self.GetIrValue(), hl_other.GetIrValue()});

  auto shape_out = MMOperator::compute_output_shape(mat1, mat2);
  auto result = at::native::empty_hpu_lazy(
      shape_out, mat1.options(), mat1.suggest_memory_format(), false);

  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{mat1, mat2};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};

Tensor addmm_hpu_lazy(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
  PT_LAZY_TRACE;
  const auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  const auto hl_mat1 = habana_lazy::GetOrCreateHbLazyTensor(mat1, c10::kHABANA);
  const auto hl_mat2 = habana_lazy::GetOrCreateHbLazyTensor(mat2, c10::kHABANA);
  const auto hl_beta = habana_lazy::GetIrValueForScalar(beta);
  const auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

  const auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::addmm"),
      {hl_self.GetIrValue(),
       hl_mat1.GetIrValue(),
       hl_mat2.GetIrValue(),
       hl_beta,
       hl_alpha});
  const std::vector<int64_t> shape_out = {mat1.size(0), mat2.size(1)};
  const auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  const auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self, mat1, mat2};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};

Tensor& batch_gemm_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& mat2) {
  PT_LAZY_TRACE;
  const auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  const auto hl_mat2 = habana_lazy::GetOrCreateHbLazyTensor(mat2, c10::kHABANA);

  const auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::bmm"),
      {hl_self.GetIrValue(), hl_mat2.GetIrValue()});

  const auto hlresult = habana_lazy::GetHbLazyTensor(out);
  habana_lazy::ir::Value& out_val = hlresult.CurrentIrValue();
  out_val.m_index = 0;
  out_val.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self, mat2};
  node->AddInputPtTensors(input_pt_vec);

  return out;
};

Tensor batch_gemm_hpu_lazy(const Tensor& self, const Tensor& mat2) {
  PT_LAZY_TRACE;
  const auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  const auto hl_mat2 = habana_lazy::GetOrCreateHbLazyTensor(mat2, c10::kHABANA);

  const auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::bmm"),
      {hl_self.GetIrValue(), hl_mat2.GetIrValue()});

  auto shape_out = BmmOperator::compute_output_shape(self, mat2);
  const auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  const auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self, mat2};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};

Tensor dot_hpu_lazy(const Tensor& self, const Tensor& other) {
  HABANA_ASSERT(0);
  return dot_hpu(self, other);
};
Tensor mv_hpu_lazy(const Tensor& self, const Tensor& other) {
  HABANA_ASSERT(0);
  return mv_hpu(self, other);
};
std::tuple<Tensor, Tensor> nll_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr nll_loss_node =
      std::make_shared<habana_lazy::ir::NllLoss_forward>(
          self, target, weight, reduction, ignore_index);

  // allocate Output_0
  auto result_0 = at::native::empty_hpu_lazy(
      {1}, self.options(), self.suggest_memory_format(), false);
  result_0.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});

  auto hlresult_0 = habana_lazy::GetHbLazyTensor(result_0);
  habana_lazy::ir::Value& out_0 = hlresult_0.CurrentIrValue();
  out_0.m_index = 0;
  out_0.SetNode(nll_loss_node);

  // allocate Output_1
  auto result_1 = at::native::empty_hpu_lazy(
      {}, self.options(), self.suggest_memory_format(), false);
  result_1.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});

  auto hlresult_1 = habana_lazy::GetHbLazyTensor(result_1);
  habana_lazy::ir::Value& out_1 = hlresult_1.CurrentIrValue();
  out_1.m_index = 1;
  out_1.SetNode(nll_loss_node);

  return {result_0, result_1};
};
Tensor nll_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    UNUSED const Tensor& total_weight) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr nll_loss_bwd_node =
      std::make_shared<habana_lazy::ir::NllLoss_backward>(
          grad_output,
          self,
          target,
          weight,
          reduction,
          ignore_index,
          total_weight);

  // allocate
  auto result = at::native::empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);

  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(nll_loss_bwd_node);

  return result;
};

Tensor mse_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  PT_LAZY_TRACE;
  auto node =
      std::make_shared<habana_lazy::ir::MseLoss>(self, target, reduction);
  Tensor result = mse_loss_forward_hpu(self, target, reduction);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out =
      habana_lazy::GetHbLazyTensor(result).CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};

Tensor mse_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<habana_lazy::ir::MseLoss>(
      grad_output, self, target, reduction);
  Tensor result = mse_loss_backward_hpu(grad_output, self, target, reduction);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out =
      habana_lazy::GetHbLazyTensor(result).CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};

Tensor binary_cross_entropy_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr bce_loss_node =
      std::make_shared<habana_lazy::ir::BceLoss_forward>(
          self, target, weight, reduction);

  // allocate Output
  auto result = at::native::empty_hpu_lazy(
      {1}, self.options(), self.suggest_memory_format(), false);

  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(bce_loss_node);

  return result;
};

Tensor binary_cross_entropy_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr bce_bwd_loss_node =
      std::make_shared<habana_lazy::ir::BceLoss_backward>(
          grad_output, self, target, weight, reduction);

  // allocate Output
  auto result = at::native::empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);

  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(bce_bwd_loss_node);

  return result;
};

std::tuple<Tensor, Tensor, Tensor> batch_norm_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  HABANA_ASSERT(0);
  return batch_norm_hpu(
      input, weight, bias, running_mean, running_var, training, momentum, eps);
};
std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd_hpu_lazy(
    Tensor& grad_out,
    Tensor& input,
    Tensor& weight,
    UNUSED Tensor& running_mean,
    UNUSED Tensor& running_var,
    Tensor& save_mean,
    Tensor& save_invstd,
    bool train,
    double eps,
    UNUSED std::array<bool, 3> output_mask) {
  HABANA_ASSERT(0);
  return batch_norm_bwd_hpu(
      grad_out,
      input,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_invstd,
      train,
      eps,
      output_mask);
};
std::tuple<Tensor, Tensor, Tensor> layer_norm_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    int64_t m,
    int64_t n,
    double eps) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::LayerNormForward>(
          input, weight, bias, m, n, eps);

  auto sizes = LayerNormOperator::getOutputSizes(input, m);
  // Get Output Image
  auto result_img = at::native::empty_hpu_lazy(
      std::get<0>(sizes),
      input.options(),
      input.suggest_memory_format(),
      false);
  const auto hlresult = habana_lazy::GetHbLazyTensor(result_img);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  // Get output mean and var
  auto result_mean = at::native::empty_hpu_lazy(
      std::get<1>(sizes),
      input.options(),
      input.suggest_memory_format(),
      false);
  const auto hlresult2 = habana_lazy::GetHbLazyTensor(result_mean);
  habana_lazy::ir::Value& out2 = hlresult2.CurrentIrValue();
  out2.m_index = 1;
  out2.SetNode(node);

  auto result_var = at::native::empty_hpu_lazy(
      std::get<2>(sizes),
      input.options(),
      input.suggest_memory_format(),
      false);
  const auto hlresult3 = habana_lazy::GetHbLazyTensor(result_var);
  habana_lazy::ir::Value& out3 = hlresult3.CurrentIrValue();
  out3.m_index = 2;
  out3.SetNode(node);
  return std::make_tuple(result_img, result_mean, result_var);
};
std::tuple<Tensor, Tensor, Tensor> layer_norm_backward_hpu_lazy(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    std::array<bool, 3> grad_input_mask) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::LayerNormBackward>(
          dY, X, mean, rstd, gamma, M, N, grad_input_mask);
  auto sizes = LayerNormBackwardOperator::getOutputSizes(dY, gamma);
  // Get Output Image
  auto result_dY = at::native::empty_hpu_lazy(
      std::get<0>(sizes), dY.options(), dY.suggest_memory_format(), false);
  const auto hlresult = habana_lazy::GetHbLazyTensor(result_dY);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  at::Tensor result2, result3;
  // Check if to return optional results
  if (grad_input_mask[1]) {
    result2 = at::native::empty_hpu_lazy(
        std::get<1>(sizes),
        mean.options(),
        mean.suggest_memory_format(),
        false);
    const auto hlresult2 = habana_lazy::GetHbLazyTensor(result2);
    habana_lazy::ir::Value& out2 = hlresult2.CurrentIrValue();
    out2.m_index = 1;
    out2.SetNode(node);
  }
  if (grad_input_mask[2]) {
    result3 = at::native::empty_hpu_lazy(
        std::get<2>(sizes),
        rstd.options(),
        rstd.suggest_memory_format(),
        false);
    const auto hlresult3 = habana_lazy::GetHbLazyTensor(result3);
    habana_lazy::ir::Value& out2 = hlresult3.CurrentIrValue();
    out2.m_index = 2;
    out2.SetNode(node);
  }
  return std::make_tuple(result_dY, result2, result3);
};

Tensor norm_scalar_hpu_lazy(const Tensor& self, Scalar p) {
  HABANA_ASSERT(0);
  return norm_scalar_hpu(self, p);
};

std::tuple<Tensor, Tensor> max_pool2d_with_indices_hpu_lazy(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr maxpool_node =
      std::make_shared<habana_lazy::ir::MaxPool>(
          input, kernel_size, stride, padding, dilation, ceil_mode);

  // shaper inferrence
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  bool is_nhwc = (memory_format == c10::MemoryFormat::ChannelsLast);
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, is_nhwc);

  // retunr always nhwc. convert to nchw
  std::vector<long int> shape_out = {
      opsize_nhwc.at(0),
      opsize_nhwc.at(3),
      opsize_nhwc.at(1),
      opsize_nhwc.at(2)};

  // allocate Output_0 storage
  auto result_0 = at::native::empty_hpu_lazy(
      shape_out, input.options(), input.suggest_memory_format(), false);
  auto hlresult_0 = habana_lazy::GetHbLazyTensor(result_0);
  habana_lazy::ir::Value& out_0 = hlresult_0.CurrentIrValue();
  out_0.m_index = 0;
  out_0.SetNode(maxpool_node);

  // allocate Output_1 storage
  auto type = kByte;
  if (input.scalar_type() == c10::ScalarType::BFloat16) {
    type = kShort;
  }
  auto result_1 = at::native::empty_hpu_lazy(
      shape_out,
      input.options().dtype(type),
      input.suggest_memory_format(),
      false);
  auto hlresult_1 = habana_lazy::GetHbLazyTensor(result_1);
  habana_lazy::ir::Value& out_1 = hlresult_1.CurrentIrValue();
  out_1.m_index = 1;
  out_1.SetNode(maxpool_node);

  return {result_0, result_1};
};
Tensor& max_pool2d_with_indices_backward_out_hpu_lazy(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  HABANA_ASSERT(0);
  return max_pool2d_with_indices_backward_out_hpu(
      grad_input,
      grad_output,
      input,
      indices,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
};
Tensor max_pool2d_with_indices_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& indices) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr maxpool_bwd_node =
      std::make_shared<habana_lazy::ir::MaxPoolBackWard>(
          grad_output,
          input,
          kernel_size,
          stride,
          padding,
          dilation,
          ceil_mode,
          indices);

  // shape inferrence
  // since grad_input should match memory format only checking for input
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  bool is_nhwc = (memory_format == c10::MemoryFormat::ChannelsLast);
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, is_nhwc);

  // retunr always nhwc. convert to nchw
  std::vector<long int> out_shape = {
      opsize_nhwc.at(0),
      opsize_nhwc.at(3),
      opsize_nhwc.at(1),
      opsize_nhwc.at(2)};

  TORCH_CHECK(grad_output.sizes().vec() == out_shape);
  TORCH_CHECK(
      (indices.scalar_type() == c10::ScalarType::Byte) ||
      (indices.scalar_type() == c10::ScalarType::Short));

  // allocate storage
  auto result = at::native::empty_hpu_lazy(
      input.sizes(), input.options(), input.suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(maxpool_bwd_node);

  return result;
};

Tensor avg_pool2d_hpu_lazy(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr avgpool_node =
      std::make_shared<habana_lazy::ir::AvgPool>(
          input,
          kernel_size,
          stride,
          padding,
          ceil_mode,
          count_include_pad,
          divisor_override);

  // shaper inference
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  bool is_nhwc = (memory_format == c10::MemoryFormat::ChannelsLast);
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, is_nhwc);

  // return always nhwc. convert to nchw
  std::vector<long int> shape_out = {
      opsize_nhwc.at(0),
      opsize_nhwc.at(3),
      opsize_nhwc.at(1),
      opsize_nhwc.at(2)};

  // allocate Output storage
  auto result = at::native::empty_hpu_lazy(
      shape_out, input.options(), input.suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(avgpool_node);

  return result;
};

Tensor& avg_pool2d_backward_out_hpu_lazy(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  HABANA_ASSERT(0);
  return avg_pool2d_backward_out_hpu(
      grad_input,
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
};
Tensor avg_pool2d_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr avgpool_bwd_node =
      std::make_shared<habana_lazy::ir::AvgPoolBackWard>(
          grad_output,
          input,
          kernel_size,
          stride,
          padding,
          ceil_mode,
          count_include_pad,
          divisor_override);

  // shape inference
  c10::MemoryFormat memory_format = habana_helpers::get_memory_format({&input});
  bool is_nhwc = (memory_format == c10::MemoryFormat::ChannelsLast);
  std::vector<int64_t> d{1, 1}; // setting dilation to 1 for avg pool
  IntArrayRef dilation(d.data(), d.size());
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, is_nhwc);

  // retunr always nhwc. convert to nchw
  std::vector<long int> out_shape = {
      opsize_nhwc.at(0),
      opsize_nhwc.at(3),
      opsize_nhwc.at(1),
      opsize_nhwc.at(2)};

  TORCH_CHECK(grad_output.sizes().vec() == out_shape);

  // allocate storage
  auto result = at::native::empty_hpu_lazy(
      input.sizes(), input.options(), input.suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(avgpool_bwd_node);

  return result;
};

void uniform_hpu_lazy(
    const Tensor& self,
    double from,
    double to,
    CPUGenerator* gen) {
  HABANA_ASSERT(0);
  uniform_hpu(self, from, to, gen);
};
void normal_hpu_lazy(
    const Tensor& self,
    double mean,
    double std,
    CPUGenerator* gen) {
  HABANA_ASSERT(0);
  normal_hpu(self, mean, std, gen);
};
Tensor bernoulli_hpu_lazy(const Tensor& self, CPUGenerator* gen) {
  HABANA_ASSERT(0);
  return bernoulli_hpu(self, gen);
};
Tensor& bernoulli_scalar_hpu_lazy(Tensor& self, double p, CPUGenerator* gen) {
  HABANA_ASSERT(0);
  return bernoulli_scalar_hpu(self, p, gen);
};
std::tuple<Tensor, Tensor> fused_dropout_hpu_lazy(
    const Tensor& self,
    double p,
    CPUGenerator* gen) {
  HABANA_ASSERT(0);
  return fused_dropout_hpu(self, p, gen);
};
Tensor sum_dim_IntList_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::SumDimIntList>(
          self, dim, keepdim, dtype);
  auto result = sum_dim_IntList_hpu(self, dim, keepdim, dtype);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};

Tensor& sum_IntList_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  HABANA_ASSERT(0);
  return sum_IntList_out_hpu(output, self, dim, keepdim, dtype);
};
Tensor mean_dim_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  HABANA_ASSERT(0);
  return mean_dim_hpu(self, dim, keepdim, dtype);
};
Tensor& mean_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  HABANA_ASSERT(0);
  return mean_dim_out_hpu(output, self, dim, keepdim, dtype);
};

Tensor sum_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Sum>(self, dtype);
  PT_LAZY_TRACE;
  auto result = sum_hpu(self, dtype);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};

Tensor mean_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  HABANA_ASSERT(0);
  return mean_hpu(self, dtype);
};
Tensor& any_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  HABANA_ASSERT(0);
  return any_dim_out_hpu(output, self, dim, keepdim);
};
Tensor any_dim_hpu_lazy(const Tensor& self, int64_t dim, bool keepdim) {
  HABANA_ASSERT(0);
  return any_dim_hpu(self, dim, keepdim);
};
Tensor any_hpu_lazy(const Tensor& self) {
  HABANA_ASSERT(0);
  return any_hpu(self);
};
namespace habana {
Tensor log_softmax_hpu_lazy(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<habana_lazy::ir::LogSoftMax>(
      self, dim, half_to_float, "aten::_log_softmax");
  // infer shape
  auto result = log_softmax_hpu(self, dim, half_to_float);

  auto hl_result = habana_lazy::GetHbLazyTensor(result);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};
Tensor log_softmax_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<habana_lazy::ir::LogSoftMaxBackward>(
      grad, output, dim, input, "aten::_log_softmax_backward_data");
  // infer output shape
  auto result = log_softmax_backward_hpu(grad, output, dim, input);

  auto hl_result = habana_lazy::GetHbLazyTensor(result);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};

Tensor softmax_hpu_lazy(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<habana_lazy::ir::LogSoftMax>(
      self, dim, half_to_float, "aten::_softmax");
  // infer shape
  auto result = softmax_hpu(self, dim, half_to_float);

  auto hl_result = habana_lazy::GetHbLazyTensor(result);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};

Tensor softmax_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<habana_lazy::ir::LogSoftMaxBackward>(
      grad, output, dim, input, "aten::_softmax_backward_data");
  // infer output shape
  auto result = softmax_backward_hpu(grad, output, dim, input);

  auto hl_result = habana_lazy::GetHbLazyTensor(result);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};
} // namespace habana
namespace at {
namespace native {
Tensor empty_hpu_lazy(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format,
    bool create_storage) {
  PT_LAZY_TRACE;
  auto dtype = options.dtype();
  auto type = typeMetaToScalarType(dtype);
  // Dont allocate 8 bytes for double/long as we are anyway going to cast at
  // CPU and then copy to device @ 4byts per element
  type = type == c10::ScalarType::Long ? c10::ScalarType::Int : type;
  type = type == c10::ScalarType::Double ? c10::ScalarType::Float : type;
  dtype = scalarTypeToTypeMeta(type);

  if (create_storage) {
    c10 ::Allocator* allocator;
    if (options.pinned_memory()) {
      TORCH_CHECK(false, "habana allocator doesn't supported pinned memory");
    } else {
      allocator = habana::getHABANADeviceAllocator();
    }
    int64_t nelements = prod_intlist(size);
    int elem_size = dtype.itemsize();
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        dtype,
        nelements,
        allocator->allocate(nelements * elem_size),
        allocator,
        /*resizeable=*/true);
    Tensor at_internal_tensor =
        habana_lazy::AtenInternalHbTensor(std::move(storage_impl));
    // Setup the tensor sizes/strides, for now assuming contiguous
    at_internal_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);

    Tensor at_tensor;
    bool is_in_lowering_mode = false;
    auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
        options.device().index());
    if (context != nullptr) {
      auto exec_mode = context->getExecutionMode();
      is_in_lowering_mode = exec_mode == kLOWERING ? true : is_in_lowering_mode;
    }

    // This call could have come from a .to call and not from a lowering
    // context. In such case, create the lazt tensor.
    if (!is_in_lowering_mode) {
      habana_lazy::HbLazyTensor hb_tensor =
          habana_lazy::HbLazyTensor::CreateHbLazyTensor(
              size, 0, options.device(), c10::typeMetaToScalarType(dtype));

      at_tensor = habana_lazy::AtenFromHbLazyTensor(hb_tensor);

      // The lazy tensor will have a reference to the internal tensor
      hb_tensor.SetTensorData(at_internal_tensor);

      // Setup the tensor sizes/strides, for now assuming contiguous
      at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);

      // Keep a pointer to the storageless tensor from the internal tensor
      auto at_internal_impl =
          habana_lazy::GetHbInternalTensorImpl(at_internal_tensor);
      HABANA_ASSERT(at_internal_impl != nullptr);
      // setTensorAsInputNode(hb_tensor);
    }

    // If we are not from lowering context, return the storageless one.
    if (!is_in_lowering_mode) {
      return at_tensor;
    } else {
      // else return the internal tensor with storage
      return at_internal_tensor;
    }
  } else {
    habana_lazy::HbLazyTensor hb_tensor =
        habana_lazy::HbLazyTensor::CreateHbLazyTensor(
            size, 0, options.device(), c10::typeMetaToScalarType(dtype));
    Tensor at_tensor = habana_lazy::AtenFromHbLazyTensor(hb_tensor);
    // Setup the tensor sizes/strides, for now assuming contiguous
    at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    return at_tensor;
  }
};
Tensor empty_strided_hpu_lazy(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options,
    bool create_storage) {
  PT_LAZY_TRACE;
  at::Tensor empty_tensor =
      empty_hpu_lazy(size, options, c10::nullopt, create_storage);
  empty_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  // If we have created a tensor with storage, set the strides and sizes to
  // backend tensor as well
  if (create_storage) {
    auto hl_empty = habana_lazy::TryGetHbLazyTensor(empty_tensor);
    if (hl_empty) {
      setTensorAsInputNode(hl_empty.value());
      hl_empty.value().getAttachedTensorImpl()->set_sizes_and_strides(
          size, stride);
    }
  }
  return empty_tensor;
};
} // namespace native
} // namespace at
Tensor clone_hpu_lazy(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format) {
  PT_LAZY_TRACE;
  TORCH_CHECK(self.defined(), "src is undefined");
  TORCH_CHECK(
      self.device().type() == c10::DeviceType::HABANA,
      "Lazy kernel only supports clone on Habana Device");
  TORCH_CHECK(
      habana_lazy::IsHbLazyTensor(self),
      "src is not a Habana Lazy Tensor, currently NOT supported in cloning");

  // We need to add device to device copy kernel here
  // As d2D copies may not mean trigger execution, we just need to add the
  // nodes like memcopy to our lazy graph that we are creating
  habana_lazy::HbLazyTensor hb_tensor =
      habana_lazy::GetOrCreateHbLazyTensor(self, self.device());
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::habana_d2d_memcpy"),
      {hb_tensor.GetIrValue()});
  auto result = at::native::empty_hpu_lazy(
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      /*storage=*/false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);
  return result;
};
Tensor& zero_hpu_lazy(Tensor& self) {
  return fill_hpu_lazy_(self, 0);
};
Tensor cat_hpu_lazy(const TensorList tensors, int64_t dim_) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Cat>(tensors, dim_);

  auto first_tensor = tensors[0];

  auto shape_out = first_tensor.sizes().vec();
  shape_out[dim_] = 0;
  auto tensor_count = tensors.size();
  for (unsigned i = 0; i < tensor_count; i++) {
    shape_out[dim_] += tensors[i].sizes()[dim_];
  }

  auto result = at::native::empty_hpu_lazy(
      shape_out,
      first_tensor.options(),
      first_tensor.suggest_memory_format(),
      false);

  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};
Tensor& cat_hpu_lazy_out(
    Tensor& result,
    const TensorList tensors,
    int64_t dim_) {
  HABANA_ASSERT(0);
  return cat_hpu_out(result, tensors, dim_);
};

Tensor transpose_hpu_lazy(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Transpose>(self, dim0_, dim1_);
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      TransposeOperator::compute_output_shape(self, dim0_, dim1_);
  auto result = at::native::empty_strided_hpu_lazy(
      new_sizes, new_strides, self.options(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};

Tensor& transpose_hpu_lazy_(Tensor& self, int64_t dim0_, int64_t dim1_) {
  HABANA_ASSERT(0);
  return transpose_hpu_(self, dim0_, dim1_);
};

Tensor t_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::t"), {hl_self.GetIrValue()});
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) = TOperator::compute_output_shape(self);
  auto result = at::native::empty_strided_hpu_lazy(
      new_sizes, new_strides, self.options(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};

Tensor& t_hpu_lazy_(Tensor& self) {
  HABANA_ASSERT(0);
  return t_hpu_(self);
};

Tensor permute_hpu_lazy(const Tensor& self, IntArrayRef dims_) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Permute>(self, dims_);
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      PermuteOperator::compute_output_shape(self, dims_.vec());
  auto result = at::native::empty_strided_hpu_lazy(
      new_sizes, new_strides, self.options(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};

Tensor expand_hpu_lazy(const Tensor& self, IntArrayRef size, bool implicit) {
  HABANA_ASSERT(0);
  return expand_hpu(self, size, implicit);
};
std::vector<Tensor> split_with_sizes_hpu_lazy(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  HABANA_ASSERT(0);
  return split_with_sizes_hpu(self, split_sizes, dim);
};
Tensor threshold_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold) {
  PT_LAZY_TRACE;
  auto hl_grad =
      habana_lazy::GetOrCreateHbLazyTensor(grad_output, c10::kHABANA);
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_threshold = habana_lazy::GetIrValueForScalar(threshold);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::threshold_backward"),
      {hl_grad.GetIrValue(), hl_self.GetIrValue(), hl_threshold});
  auto result = at::native::empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{grad_output, self};
  node->AddInputPtTensors(input_pt_vec);

  return result;
}

std::tuple<Tensor&, Tensor&> topk_out_hpu_lazy(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool largest,
    bool sorted) {
  HABANA_ASSERT(0);
  return topk_out_hpu(values, indices, self, k, dim_, largest, sorted);
};
std::tuple<Tensor, Tensor> topk_hpu_lazy(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  HABANA_ASSERT(0);
  return topk_hpu(self, k, dim, largest, sorted);
};
std::tuple<Tensor, Tensor> sort_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  HABANA_ASSERT(0);
  return sort_hpu(self, dim, descending);
};
Tensor unary_op_hpu_lazy(
    const Tensor& input,
    std::string& node_type,
    UnaryOperator* Op) {
  HABANA_ASSERT(0);
  return unary_op_hpu(input, node_type, Op);
};
Tensor unary_backward_op_hpu_lazy(
    const Tensor& grad_in,
    const Tensor& input,
    std::string& node_type,
    UnaryBackwardOperator* Op) {
  HABANA_ASSERT(0);
  return unary_backward_op_hpu(grad_in, input, node_type, Op);
};
Tensor relu_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::relu"), {hl_input.GetIrValue()});
  auto result = at::native::empty_hpu_lazy(
      input.sizes(), input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};
Tensor& relu_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::relu"), {hl_input.GetIrValue()});
  auto hl_result = habana_lazy::GetHbLazyTensor(input);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);
  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      input.device().index());
  context->MarkTensorStatus(
      hl_input.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);

  return input;
};
Tensor sigmoid_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::sigmoid"), {hl_input.GetIrValue()});
  auto shape_out = input.sizes();
  auto result = at::native::empty_hpu_lazy(
      shape_out, input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};

Tensor sigmoid_backward_hpu_lazy(const Tensor& grad_in, const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_grad_in = habana_lazy::GetOrCreateHbLazyTensor(grad_in, c10::kHABANA);
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::sigmoid_backward"),
      {hl_grad_in.GetIrValue(), hl_input.GetIrValue()});

  auto result = at::native::empty_hpu_lazy(
      input.sizes(), input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{grad_in, input};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};

// make sqrt as inplace op for workaround in SW-26172
Tensor sqrt_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::sqrt_"), {hl_input.GetIrValue()});

  habana_lazy::ir::Value& out = hl_input.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);
  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      input.device().index());
  // context->MarkTensorRegistered(hl_input.getTensorUniqueId());
  context->MarkTensorStatus(
      hl_input.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);

  return input;
};
Tensor sqrt_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::sqrt"), {hl_input.GetIrValue()});
  auto shape_out = input.sizes();
  auto result = at::native::empty_hpu_lazy(
      shape_out, input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};

Tensor tanh_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::tanh"), {hl_input.GetIrValue()});
  auto shape_out = input.sizes();
  auto result = at::native::empty_hpu_lazy(
      shape_out, input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};

Tensor& tanh_hpu_lazy_(Tensor& self) {
  HABANA_ASSERT(0);
  return tanh_hpu_(self);
};
Tensor& tanh_out_hpu_lazy(Tensor& out, Tensor& self) {
  HABANA_ASSERT(0);
  return tanh_out_hpu(out, self);
};

Tensor tanh_backward_hpu_lazy(const Tensor& grad_in, const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_grad_in = habana_lazy::GetOrCreateHbLazyTensor(grad_in, c10::kHABANA);
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::tanh_backward"),
      {hl_grad_in.GetIrValue(), hl_input.GetIrValue()});

  auto result = at::native::empty_hpu_lazy(
      input.sizes(), input.options(), input.suggest_memory_format());
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{grad_in, input};
  node->AddInputPtTensors(input_pt_vec);

  return result;
};

Tensor gelu_hpu_lazy(const Tensor& self) {
  HABANA_ASSERT(0);
  return gelu_hpu(self);
};
Tensor gelu_backward_hpu_lazy(const Tensor& grad, const Tensor& self) {
  HABANA_ASSERT(0);
  return gelu_backward_hpu(grad, self);
};
Tensor& erf_hpu_lazy_(Tensor& self) {
  HABANA_ASSERT(0);
  return erf_hpu_(self);
};
Tensor erf_hpu_lazy(const Tensor& self) {
  HABANA_ASSERT(0);
  return erf_hpu(self);
};
Tensor& exp_hpu_lazy_(Tensor& self) {
  HABANA_ASSERT(0);
  return exp_hpu_(self);
};
Tensor exp_hpu_lazy(const Tensor& self) {
  HABANA_ASSERT(0);
  return exp_hpu(self);
};
Tensor& neg_out_hpu_lazy(Tensor& result, const Tensor& input) {
  HABANA_ASSERT(0);
  return neg_out_hpu(result, input);
};
Tensor& reciprocal_hpu_lazy_(Tensor& self) {
  HABANA_ASSERT(0);
  return reciprocal_hpu_(self);
};
Tensor reciprocal_hpu_lazy(const Tensor& self) {
  HABANA_ASSERT(0);
  return reciprocal_hpu(self);
};
Tensor& reciprocal_out_hpu_lazy(Tensor& result, const Tensor& self) {
  HABANA_ASSERT(0);
  return reciprocal_out_hpu(result, self);
};
Tensor clamp_min_hpu_lazy(const Tensor& self, Scalar min) {
  HABANA_ASSERT(0);
  return clamp_min_hpu(self, min);
};
Tensor& clamp_hpu_lazy_(
    Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  HABANA_ASSERT(0);
  return clamp_hpu_(self, min, max);
};
Tensor clamp_hpu_lazy(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  HABANA_ASSERT(0);
  return clamp_hpu(self, min, max);
};
Tensor abs_hpu_lazy(const Tensor& input) {
  HABANA_ASSERT(0);
  PT_LAZY_TRACE;
};
Tensor neg_hpu_lazy(const Tensor& self) {
  HABANA_ASSERT(0);
  return neg_hpu(self);
};
namespace at {
namespace native {
Scalar _local_scalar_dense_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  Scalar out;
  // If self is a lazy tensor make sure the execution till the point of self
  // getting flled has finished before we start copying
  if (habana_lazy::IsHbLazyTensor(self)) {
    habana_lazy::HbLazyTensor hb_tensor =
        habana_lazy::GetOrCreateHbLazyTensor(self, self.device());
    // Trigger point execution
    auto tensor_data = hb_tensor.GetHbLazyTensorData();
    out = _local_scalar_dense_hpu(tensor_data.value());
  } else {
    out = _local_scalar_dense_hpu(self);
  }
  return out;
}
} // namespace native
} // namespace at
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_lazy(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor,
    float mom,
    bool nesterov) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::OptimizerSparseSgdValidCount>(
          gradients,
          weights_in,
          moments_in,
          indices,
          learning_rate,
          valid_count_tensor,
          mom,
          nesterov);

  auto hlweights = habana_lazy::GetHbLazyTensor(weights_in);
  habana_lazy::ir::Value& out1 = hlweights.CurrentIrValue();
  out1.m_index = 0;
  out1.SetNode(node);
  auto hlmoments = habana_lazy::GetHbLazyTensor(moments_in);
  habana_lazy::ir::Value& out2 = hlmoments.CurrentIrValue();
  out2.m_index = 1;
  out2.SetNode(node);
  return std::tie(weights_in, moments_in);
}
std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_lazy(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor) {
  PT_LAZY_TRACE;
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::habanaOptimizerSparseAdagrad"), {});

  std::vector<habana_lazy::HbLazyTensor> hl_tensors;
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(gradients, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(weights_in, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(moments_in, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(indices, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(learning_rate, c10::kHABANA));
  hl_tensors.push_back(
      habana_lazy::GetOrCreateHbLazyTensor(valid_count_tensor, c10::kHABANA));

  for (auto& i : hl_tensors) {
    node->AddInput(i.GetIrValue());
  }

  auto hlweights = habana_lazy::GetHbLazyTensor(weights_in);
  habana_lazy::ir::Value& out1 = hlweights.CurrentIrValue();
  out1.m_index = 0;
  out1.SetNode(node);
  auto hlmoments = habana_lazy::GetHbLazyTensor(moments_in);
  habana_lazy::ir::Value& out2 = hlmoments.CurrentIrValue();
  out2.m_index = 1;
  out2.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{
      gradients,
      weights_in,
      moments_in,
      indices,
      learning_rate,
      valid_count_tensor};
  node->AddInputPtTensors(input_pt_vec);

  return std::tie(weights_in, moments_in);
}

void optimizer_adamw_hpu_lazy(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay) {
  PT_LAZY_TRACE;
  TORCH_CHECK(false, "Not implemented yet");
}

Tensor ones_like_hpu_lazy(
    const Tensor& self,
    const TensorOptions& options,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  // Note that currently we are not lowering parameters as per the ones_like
  // schema. This works for the ones_like usage in MNIST (where it is used
  // only for filling grad_out tensor with 1's), but we may need to revisit
  // this in future.
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  habana_lazy::ir::NodePtr node = std::make_shared<habana_lazy::ir::OnesLike>(
      self, options, optional_memory_format);

  auto result = at::native::empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};