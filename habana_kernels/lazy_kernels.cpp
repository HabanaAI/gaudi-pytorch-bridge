/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_kernels/lazy_kernels.h"
#include <ATen/InferSize.h>
#include <bitset>
#include <cstdlib>
#include <ctime>
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/aten_hpu_type_default.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/bitwise_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/conv_kernels.h"
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/embedding_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_kernels/loss_kernels.h"
#include "habana_kernels/nonzero_kernel.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/pool_kernels.h"
#include "habana_kernels/reduction2_kernels.h"
#include "habana_kernels/repeat.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/upsample_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/ops/cast_ops.h"
#include "habana_lazy/ops/cat.h"
#include "habana_lazy/ops/clamp.h"
#include "habana_lazy/ops/constant.h"
#include "habana_lazy/ops/convolution.h"
#include "habana_lazy/ops/embedding.h"
#include "habana_lazy/ops/embedding_bag.h"
#include "habana_lazy/ops/hpu_input.h"
#include "habana_lazy/ops/index.h"
#include "habana_lazy/ops/loss.h"
#include "habana_lazy/ops/matmul.h"
#include "habana_lazy/ops/mse_loss.h"
#include "habana_lazy/ops/norm.h"
#include "habana_lazy/ops/optimizer.h"
#include "habana_lazy/ops/optimizer_sparse_sgd_with_valid_count.h"
#include "habana_lazy/ops/pool.h"
#include "habana_lazy/ops/reduce_ops.h"
#include "habana_lazy/ops/shape_ops.h"
#include "habana_lazy/ops/softmax.h"
#include "habana_lazy/ops/tensor_shape.h"
#include "habana_lazy/ops/topk.h"
#include "habana_lazy/ops/unpack.h"
#include "habana_lazy/ops/upsample.h"
#include "habana_lazy/view.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/synapse_helpers/env_flags.h"
#include "pytorch_helpers/synapse_helpers/util.h"

using namespace habana_lazy;
using namespace habana;
using namespace at;

#define STRINGIFY(op_code) #op_code

#define HPU_LAZY_FUNC_NAME(op_code) op_code##_hpu_lazy
#define HPU_LAZY_FUNC_NAME_INPLACE(op_code) op_code##hpu_lazy_
#define HPU_LAZY_WRAP_KERNEL(op_code)                       \
  Tensor HPU_LAZY_FUNC_NAME(op_code)(const Tensor& self) {  \
    PT_LAZY_TRACE;                                          \
    LazyOp<at::Tensor> k{STRINGIFY(aten::op_code), {self}}; \
    return k.call();                                        \
  }
#define HPU_LAZY_WRAP_KERNEL_INPLACE(op_code)                  \
  Tensor& HPU_LAZY_FUNC_NAME_INPLACE(op_code)(Tensor & self) { \
    PT_LAZY_TRACE;                                             \
    LazyOp<at::Tensor&> k{STRINGIFY(aten::op_code), {self}};   \
    return k.call(self);                                       \
  }

bool to_lower_as_strided() {
  return GET_ENV_FLAG(PT_HPU_LOWER_AS_STRIDED);
}

void flushWithMarkStep() {
  // Generate a random number and invoke the mark_step
  static std::once_flag flag;
  std::call_once(flag, [&]() { srand((unsigned)time(0)); });

  // Generate a random number between 1 - 100
  auto rand_num = rand() % 100 + 1;

  // By default, we want to trigger 50% of the time
  auto aggressiveness = 50;
  if (const auto envp =
          std::getenv("INTERNAL_PT_HPU_LAZY_MARK_STEP_TEST_TRIGGER")) {
    aggressiveness = std::stoul(envp, nullptr, 10);
    // Cap the trigger to at least 1% to at most 100%
    if (aggressiveness < 1) {
      aggressiveness = 0;
    } else if (aggressiveness > 100) {
      aggressiveness = 100;
    }
  }
  if (rand_num < aggressiveness) {
    PT_LAZY_DEBUG("Triggering a mark_step");
    HbLazyTensor::StepMarker({});
  }
}

// For the ops that don't use LazyOp to construct nodes.
// Remove when all ops move to LazyOp style.
static void flush_op(at::TensorList tensors) {
  static const bool m_flush_op = std::getenv("PT_HPU_LAZY_MODE") &&
      *std::getenv("PT_HPU_LAZY_MODE") == '2';
  static const bool m_random_flush = std::getenv("PT_HPU_LAZY_MODE") &&
      *std::getenv("PT_HPU_LAZY_MODE") == '3';

  if (m_flush_op) {
    std::vector<HbLazyTensor> hl_tensors;
    hl_tensors.reserve(tensors.size());
    for (const auto& t : tensors) {
      hl_tensors.push_back(GetOrCreateHbLazyTensor(t));
    }
    HbLazyTensor::SyncTensorsGraph(&hl_tensors);
  } else if (m_random_flush) {
    flushWithMarkStep();
  }
}

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

std::vector<int64_t> CalculateStrides(
    const IntArrayRef sizes,
    c10::MemoryFormat format) {
  HABANA_ASSERT(sizes.size() == 4);
  if (c10::MemoryFormat::ChannelsLast == format) {
    return {sizes[1] * sizes[2] * sizes[3], 1, sizes[1] * sizes[3], sizes[1]};
  }
  return {sizes[1] * sizes[2] * sizes[3], sizes[1] * sizes[2], sizes[1], 1};
}

habana_lazy::ir::Value AddControlEdge(
    const at::Tensor& src,
    const at::Tensor& dst) {
  auto hb_result = habana_lazy::GetOrCreateHbLazyTensor(dst, dst.device());
  auto hb_tensor = habana_lazy::GetOrCreateHbLazyTensor(src, src.device());
  // We are using this lazy tensor as output on some op
  // Version counter tracks the number of times we do that
  // if its zero, that means this tensor hasnt been output in any op
  hb_result.updateVersion();
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::control_edge_other_"),
      {hb_tensor.GetIrValue(), hb_result.GetIrValue()});
  node->set_as_control_edge();
  std::vector<at::Tensor> input_pt_vec;
  input_pt_vec.push_back(src);
  input_pt_vec.push_back(dst);
  habana_lazy::ir::Value& out = hb_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  node->AddInputPtTensors(input_pt_vec);
  return out;
}

void updateDstDependencies(
    habana_lazy::HbLazyTensor& hl_dst,
    const Tensor& dst,
    bool in_place) {
  auto view = hl_dst.getView();
  // FIXME: deactivating code to add control edge for updating views of the
  // tensors
  // This case isnt hit right now and ww got cache crashes with this control
  // edge needs a design review and fix to activate
  if (view) {
    if (!to_lower_as_strided()) {
      PT_LAZY_DEBUG(
          "WARNING: We are hitting a case where the dst tensor has a view. Not all cases are covered so functionality might be impacted ");
      return;
    }
    // Now we have to update the IR of the input as it got modified
    // We need to link it as output of this node as we need to generate the
    // correct order
    // Create a dummy node from the output back to input thats creating a
    // view on  input
    auto view_val = view.value();
    at::Tensor view_tensor = view_val.getAtTensor();
    habana_lazy::ir::Value val{view_val.getIR().m_data_ptr.lock()};

    auto view_lazy_tensor = habana_lazy::GetHbLazyTensor(view_tensor);
    view_lazy_tensor.AssignIrValue(val);
    AddControlEdge(dst, view_tensor);
  }
  if (in_place) {
    // We are using this lazy tensor as output on some op
    // Version counter tracks the number of times we do that
    // if its zero, that means this tensor hasnt been output in any op
    hl_dst.updateVersion();
    auto hb_result = habana_lazy::GetOrCreateHbLazyTensor(dst, dst.device());
    habana_lazy::ir::Value val{hb_result.GetIrValue().m_data_ptr.lock()};
    auto node = habana_lazy::ir::Node::Create(
        Symbol::fromQualString("hpu::control_edge_"), {hb_result.GetIrValue()});
    node->set_as_control_edge();
    std::vector<at::Tensor> input_pt_vec;
    input_pt_vec.push_back(dst);
    habana_lazy::ir::Value& out = val;
    out.m_index = 0;
    out.SetNode(node);
    hb_result.AssignIrValue(val);
    node->AddInputPtTensors(input_pt_vec);
  }
}

at::Tensor get_tensor_for_scalar(float alpha) {
  at::Tensor alpha_tensor;

  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);

  auto map_it = context->scalar_to_tensor_map.find(alpha);
  if (map_it == context->scalar_to_tensor_map.end()) {
    alpha_tensor = at::tensor(alpha).to(c10::kHABANA, true);
    context->scalar_to_tensor_map[alpha] = alpha_tensor;
  } else {
    alpha_tensor = map_it->second;
  }

  return alpha_tensor;
}

Tensor& copy_hpu_lazy_D2D(Tensor& self, const Tensor& src, bool non_blocking) {
  if (to_lower_as_strided()) {
    PT_LAZY_TRACE;
    habana_lazy::ir::NodePtr node;
    std::vector<at::Tensor> input_pt_vec;
    habana_lazy::HbLazyTensor hb_tensor =
        habana_lazy::GetOrCreateHbLazyTensor(src, src.device());
    auto hlresult = habana_lazy::GetOrCreateHbLazyTensor(self, src.device());
    bool permuted = false;
    /* We can't create a long/double target in the device. Even a cast will not
      work as these data types are not available within the device. The only way
      to make progress is to just do a normal D2D so that the target will also
      be the same as source, and when we want to pull this out to CPU, the D2H
      will handle the type conversion*/
    if ((self.scalar_type() == c10::ScalarType::Long) ||
        (self.scalar_type() == c10::ScalarType::Double) ||
        (src.dtype() == self.dtype())) {
      // If both src and dst are already processed ,  go and do the DMA dont
      // wait Else , If we already have storage in dst, add memcopy node to lazy
      // graph and we want to copy to existing tensor and not a new one
      // Kernel expects us to pass dst as second input in that case
      auto result_data = hlresult.CurrentTensorData();
      auto src_data = hb_tensor.CurrentTensorData();
      if (copy_transpose_valid(self, src)) {
        permuted = true;
        int64_t dim_chl_pos[] = {0, 2, 3, 1};
        at::IntArrayRef chl_pos = dim_chl_pos;
        self = permute_cl_hpu_lazy(src, chl_pos);
      } else if (!permuted) {
        if (hb_tensor.getTensorUniqueId() == hlresult.getTensorUniqueId()) {
          return self;
        }
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
        // updatet the view if any
        updateDstDependencies(hlresult, self);
      }
    } else {
      node = std::make_shared<habana_lazy::ir::Cast>(
          src, self.scalar_type(), non_blocking);
      auto hlresult = habana_lazy::GetHbLazyTensor(self);
      habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
      out.m_index = 0;
      out.SetNode(node);
      // updatet the view if any
      updateDstDependencies(hlresult, self);
    }

    flush_op(self);
    return self;
  } else {
    PT_LAZY_TRACE;
    habana_lazy::ir::NodePtr node;
    std::vector<at::Tensor> input_pt_vec;
    habana_lazy::HbLazyTensor hb_tensor =
        habana_lazy::GetOrCreateHbLazyTensor(src, src.device());
    auto hlresult = habana_lazy::GetOrCreateHbLazyTensor(self, src.device());
    bool permuted = false;
    bool storage_attached = hlresult.isStorageAttached();
    /* We can't create a long/double target in the device. Even a cast will not
      work as these data types are not available within the device. The only way
      to make progress is to just do a normal D2D so that the target will also
      be the same as source, and when we want to pull this out to CPU, the D2H
      will handle the type conversion*/
    if ((self.scalar_type() == c10::ScalarType::Long) ||
        (self.scalar_type() == c10::ScalarType::Double) ||
        (src.dtype() == self.dtype())) {
      // If both src and dst are already processed ,  go and do the DMA dont
      // wait Else , If we already have storage in dst, add memcopy node to lazy
      // graph and we want to copy to existing tensor and not a new one
      // Kernel expects us to pass dst as second input in that case
      auto result_data = hlresult.CurrentTensorData();
      auto src_data = hb_tensor.CurrentTensorData();
      if (copy_transpose_valid(self, src)) {
        permuted = true;
        int64_t dim_chl_pos[] = {0, 2, 3, 1};
        at::IntArrayRef chl_pos = dim_chl_pos;
        self = permute_cl_hpu_lazy(src, chl_pos);
      } else if (!permuted || storage_attached) {
        if (storage_attached) {
          if (hb_tensor.getTensorUniqueId() == hlresult.getTensorUniqueId()) {
            return self;
          }
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
          // updatet the view if any
          updateDstDependencies(hlresult, self);
        } else {
          node = habana_lazy::ir::Node::Create(
              Symbol::fromQualString("hpu::habana_d2d_memcpy"),
              {hb_tensor.GetIrValue()});
          input_pt_vec.push_back(src);
          habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
          out.m_index = 0;
          out.SetNode(node);
          node->AddInputPtTensors(input_pt_vec);
          // updatet the view if any
          updateDstDependencies(hlresult, self);
        }
      }
    } else {
      node = std::make_shared<habana_lazy::ir::Cast>(
          src, self.scalar_type(), non_blocking);
      auto hlresult = habana_lazy::GetHbLazyTensor(self);
      habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
      out.m_index = 0;
      out.SetNode(node);
      // updatet the view if any
      updateDstDependencies(hlresult, self);
    }

    flush_op(self);
    return self;
  }
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
      PT_LAZY_DEBUG(
          "WARNING: We are hitting a case in H2D where the PyTorch tensor original data types mismatch.");
      self = self.to(src.dtype());
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

  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      self.device().index());
  auto exec_mode = context->getExecutionMode();
  if (exec_mode != kLOWERING) {
    auto self_hb_tensor =
        habana_lazy::GetOrCreateHbLazyTensor(self, self.device());
    // WE need to add storage if it wasnt created
    // right now as soon as we do a H2D transfer, we create memory and mark
    // executed
    auto isStorageAttached = self_hb_tensor.isStorageAttached();
    if (!isStorageAttached) {
      c10 ::Allocator* allocator;
      allocator = habana::getHABANADeviceAllocator();
      int64_t nelements = prod_intlist(self.sizes());
      int elem_size = self.dtype().itemsize();
      auto storage_impl = c10::make_intrusive<StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          nelements,
          allocator->allocate(nelements * elem_size),
          allocator,
          /*resizeable=*/true);
      Tensor at_internal_tensor = habana_lazy::AtenInternalHbTensor(
          std::move(storage_impl), self.dtype());
      // Setup the tensor sizes & strides for tensor with dim = 4, else for
      // now assuming contiguous
      if (4 == self.dim()) {
        at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
            src.sizes(),
            CalculateStrides(src.sizes(), src.suggest_memory_format()));
      } else {
        at_internal_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(
            src.sizes());
      }
      self_hb_tensor.SetTensorData(at_internal_tensor);
    }
  }
  auto new_tensor = preProcessIfLongorDouble(src, self, processed);

  // Get the internal tensor for copy kernel
  // First get the lazy tensor
  auto self_hb_tensor =
      habana_lazy::GetOrCreateHbLazyTensor(self, self.device());
  // We need to mark this tensor as executed
  // As this will be an input coming from host side, its doesnt need further
  // execution and is ready for consumption as input
  auto self_hb_tensor_data = self_hb_tensor.GetHbLazyTensorData();
  // This is the internal tensor, it isn't a lazy tensor
  auto self_internal_tesor = self_hb_tensor_data.value();
  HABANA_ASSERT(!habana_lazy::TryGetHbLazyTensor(self_internal_tesor));

  // self may have been resized, so re-set its size and strides
  self_internal_tesor.unsafeGetTensorImpl()->set_sizes_and_strides(
      self.sizes(), self.strides());
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
  flush_op(self);
  return self;
}

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
    auto src_hb_tensor =
        habana_lazy::GetOrCreateHbLazyTensor(src, src.device());
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
    auto src_hb_tensor =
        habana_lazy::GetOrCreateHbLazyTensor(src, src.device());
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

// This API should be called from lowering mode only
// It is used to create th backend tensor for as_strided
Tensor empty_as_strided_lazy(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  PT_LAZY_TRACE;
  auto storage_impl = self.unsafeGetTensorImpl();
  Tensor at_internal_tensor = habana_lazy::AtenInternalHbTensor(
      c10::Storage(storage_impl->storage()), self.dtype());
  at_internal_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  if (storage_offset) {
    at_internal_tensor.unsafeGetTensorImpl()->set_storage_offset(
        storage_offset.value());
  }

  return at_internal_tensor;
}
Tensor empty_from_storage_lazy(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<IntArrayRef> stride,
    c10::optional<int64_t> storage_offset) {
  PT_LAZY_TRACE;

  auto hb_tensor_self =
      habana_lazy::GetOrCreateHbLazyTensor(self, self.device());
  if (!hb_tensor_self.isStorageAttached()) {
    hb_tensor_self.GetHbLazyTensorData();
  }
  TORCH_CHECK(
      hb_tensor_self.isStorageAttached(),
      "Habana Lazy : we dont support as_strided for non storage");
  auto storage_impl = hb_tensor_self.getAttachedTensorImpl();
  Tensor at_internal_tensor = habana_lazy::AtenInternalHbTensor(
      c10::Storage(storage_impl->storage()), self.dtype());

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

// THis kernel has two paths, lowering and lazy
// During lazy we set up the as strided tensor meta data
// when we get a call back from lowering, we attache the tensor from same memory
// as source
Tensor as_strided_hpu_lazy(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  if (!to_lower_as_strided()) {
    auto hb_tensor = habana_lazy::GetOrCreateHbLazyTensor(self, self.device());
    auto src_data = hb_tensor.CurrentTensorData();
    if (size.vec().size() <= 4 && stride.vec()[stride.size() - 1] == 1) {
      auto result = empty_from_storage_lazy(
          self, size, c10::make_optional(stride), storage_offset);
      auto hb_result =
          habana_lazy::GetOrCreateHbLazyTensor(result, result.device());
      AddControlEdge(self, result);
      // Add a view of the parent to the result so that its remembered
      // If this tensor is used as a dst in any op, we need to update the parent
      auto hb_tensor =
          habana_lazy::GetOrCreateHbLazyTensor(self, self.device());
      habana_lazy::ir::LazyView view(self, hb_tensor.GetIrValue());
      hb_result.addView(view);
      flush_op(result);
      return result;
    } else {
      return AtenHpuTypeDefault::as_strided(self, size, stride, storage_offset);
    }
    flush_op(self);
    return self;
  } else {
    auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
        self.device().index());

    // when we get a call from lowering, we create a storage based backend
    // tensor
    if (context != nullptr) {
      auto exec_mode = context->getExecutionMode();
      if (exec_mode == kLOWERING) {
        auto result = empty_as_strided_lazy(self, size, stride, storage_offset);
        return result;
      }
    }

    auto hb_tensor = habana_lazy::GetOrCreateHbLazyTensor(self, self.device());
    auto src_data = hb_tensor.CurrentTensorData();

    // We only support contiguous chunks of data to be taken as strided
    // As Device doesnt support strided tensors we dont support that case
    // We can add a better check here to check contigous on all sub dims
    if (size.vec().size() <= 4 && stride.vec()[stride.size() - 1] == 1) {
      int64_t offset = storage_offset ? storage_offset.value() : 0;
      habana_lazy::ir::NodePtr node =
          std::make_shared<habana_lazy::ir::AsStrided>(
              self, size, stride, offset);
      auto result = at::native::empty_strided_hpu_lazy(
          size, stride, self.options(), false);

      auto hb_result = habana_lazy::GetHbLazyTensor(result);
      auto hb_tensor =
          habana_lazy::GetOrCreateHbLazyTensor(self, self.device());
      habana_lazy::ir::Value& out = hb_result.CurrentIrValue();
      out.m_index = 0;
      out.SetNode(node);
      // update the view if any
      updateDstDependencies(hb_result, result);
      // std::vector<at::Tensor> input_pt_vec{self};
      // node->AddInputPtTensors(input_pt_vec);
      // Add a view of the parent to the result so that its remembered
      // If this tensor is used as a dst in any op, we need to update the parent
      habana_lazy::ir::LazyView view(self, hb_tensor.GetIrValue());
      hb_result.addView(view);
      flush_op(result);
      return result;
    } else {
      return AtenHpuTypeDefault::as_strided(self, size, stride, storage_offset);
    }
    flush_op(self);
    return self;
  }
};

Tensor asin_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::asin", {self}};
  return k.call();
}

HPU_LAZY_WRAP_KERNEL(acos)
HPU_LAZY_WRAP_KERNEL_INPLACE(acos_)

Tensor acosh_hpu_lazy(const at::Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::acosh", {self}};
  return k.call();
}

Tensor& acosh_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::acosh_", {self}};
  return k.call(self);
}

Tensor asinh_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::asinh", {self}};
  return k.call();
}
Tensor& asinh_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::asinh_", {self}};
  return k.call(self);
}
Tensor atan_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::atan", {self}};
  return k.call();
}
Tensor& atan_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::atan_", {self}};
  return k.call(self);
}
Tensor atanh_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::atanh", {self}};
  return k.call();
}
Tensor& atanh_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::atanh_", {self}};
  return k.call(self);
}
Tensor cosh_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::cosh", {self}};
  return k.call();
}
Tensor& cosh_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::cosh_", {self}};
  return k.call(self);
}
Tensor& cos_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::cos_", {self}};
  return k.call(self);
}
Tensor& tanh_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::tanh_", {self}};
  return k.call(self);
}

Tensor& set_hpu_lazy_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  HABANA_ASSERT(0);
  return set_hpu_(self, source, storage_offset, size, stride);
}
Tensor view_hpu_lazy(const Tensor& self, IntArrayRef size) {
  PT_LAZY_TRACE;

  int64_t sum_elm = 1;
  for (auto& i : self.sizes()) {
    sum_elm *= i;
  }
  auto inferred_size = at::infer_size(size, static_cast<int64_t>(sum_elm));
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::View>(self, inferred_size);
  // View is internally handled as reshape and we get a new tensor as output
  auto result = at::native::empty_hpu_lazy(
      inferred_size, self.options(), self.suggest_memory_format(), false);

  auto hl_result =
      habana_lazy::GetOrCreateHbLazyTensor(result, result.device());
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  // updatet the view if any
  updateDstDependencies(hl_result, result);
  out.SetNode(node);
  flush_op(result);
  return result;
}
Tensor addcmul_hpu_lazy(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::addcmul", {self, tensor1, tensor2, alpha}};
  return k.call();
}
Tensor& addcmul_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_LAZY_TRACE;
  if (!tensor1.is_same(tensor2)) {
    auto mul_out = mul_tensor_hpu_lazy(tensor1, tensor2);
    add_tensor_hpu_lazy_(self, mul_out, alpha);
  } else {
    // implement addcmul_ as add_(pow(tensor1,2), alpha)
    auto temp = pow_tensor_scalar_hpu_lazy(tensor1, 2.0);
    add_tensor_hpu_lazy_(self, temp, alpha);
  }

  flush_op(self);
  return self;
}
Tensor addcdiv_hpu_lazy(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::addcdiv", {self, tensor1, tensor2, alpha}};
  return k.call();
}

Tensor& addcdiv_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_tensor1 = habana_lazy::GetOrCreateHbLazyTensor(tensor1, c10::kHABANA);
  auto hl_tensor2 = habana_lazy::GetOrCreateHbLazyTensor(tensor2, c10::kHABANA);

  auto alpha_float = alpha.toFloat();
  if (alpha_float == 1.0) {
    auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

    updateDstDependencies(hl_self, self, true);

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
  } else {
    auto div_out = div_tensor_hpu_lazy(tensor1, tensor2);
    auto alpha_tensor = get_tensor_for_scalar(alpha_float);
    auto mul_out = mul_tensor_hpu_lazy(alpha_tensor, div_out);
    auto out = add_tensor_hpu_lazy_(self, mul_out, 1.0);
  }

  flush_op(self);
  return self;
}

Tensor add_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  PT_LAZY_TRACE;
  auto alpha_float = alpha.toFloat();

  if (alpha_float != 1.0) {
    at::Tensor alpha_tensor = get_tensor_for_scalar(alpha_float);

    auto hl_alpha =
        habana_lazy::GetOrCreateHbLazyTensor(alpha_tensor, c10::kHABANA);
    auto mul_out = mul_tensor_hpu_lazy(other, alpha_tensor);
    return add_tensor_hpu_lazy(self, mul_out, 1.0);
  } else {
    LazyBinaryOp<at::Tensor> k{
        "aten::add",
        {self, other, alpha},
        {},
        {BinaryOperator::compute_output_shape(self, other)}};
    return k.call();
  }
}

Tensor add_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetIrValueForScalar(other);
  auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

  if (self.dim() == 0) {
    auto tensor_impl = hl_self.getAttachedTensorImpl();
    HABANA_ASSERT(tensor_impl);
    tensor_impl->set_sizes_and_strides({1}, {1});
  }

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::add"),
      {hl_self.GetIrValue(), hl_other, hl_alpha});
  auto shape_out = self.sizes().vec();
  auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, self);

  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor& add_scalar_hpu_lazy_(Tensor& self, Scalar other, Scalar alpha) {
  PT_LAZY_TRACE;
  LazyOp<Tensor&> op("aten::add_", {self, other, alpha});
  return op.call(self);
}

Tensor& add_tensor_hpu_lazy_(Tensor& self, const Tensor& other, Scalar alpha) {
  PT_LAZY_TRACE;

  LazyBinaryOp<Tensor&> op("aten::add_", {self, other, alpha});
  return op.call(self);
}

Tensor sub_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  PT_LAZY_TRACE;
  LazyBinaryOp<at::Tensor> k{
      "aten::sub",
      {self, other, alpha},
      {},
      {BinaryOperator::compute_output_shape(self, other)}};
  return k.call();
}

Tensor& sub_tensor_hpu_lazy_(Tensor& self, const Tensor& other, Scalar alpha) {
  PT_LAZY_TRACE;

  LazyBinaryOp<Tensor&> op("aten::sub_", {self, other, alpha});
  return op.call(self);
}

Tensor sub_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetIrValueForScalar(other);
  auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

  if (self.dim() == 0) {
    auto tensor_impl = hl_self.getAttachedTensorImpl();
    HABANA_ASSERT(tensor_impl);
    tensor_impl->set_sizes_and_strides({1}, {1});
  }

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::sub"),
      {hl_self.GetIrValue(), hl_other, hl_alpha});
  auto shape_out = self.sizes().vec();
  auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, self);

  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}
Tensor& sub_scalar_hpu_lazy_(Tensor& self, Scalar other, Scalar alpha) {
  PT_LAZY_TRACE;
  LazyOp<Tensor&> op("aten::sub_", {self, other, alpha});
  return op.call(self);
}
Tensor rsub_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetIrValueForScalar(other);
  auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::rsub"),
      {hl_self.GetIrValue(), hl_other, hl_alpha});

  auto result = at::native::empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hlresult, self);

  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}
Tensor& mul_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;

  LazyBinaryOp<Tensor&> op("aten::mul_", {self, other});
  return op.call(self);
}

Tensor where_tensor_hpu_lazy(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k(
      "aten::_s_where",
      {condition, self, other},
      {},
      {},
      1 /*output metadata is picked from self*/);
  return k.call();
}

Tensor mul_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyBinaryOp<at::Tensor> k{
      "aten::mul",
      {self, other},
      {},
      {BinaryOperator::compute_output_shape(self, other)}};
  return k.call();
}
Tensor& mul_out_hpu_lazy(Tensor& out, const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor&> k(
      "hpu::mul_out",
      {out, self, other},
      {},
      {BinaryOperator::compute_output_shape(self, other)});
  k.do_dma_non_first_cpu_tensor(); // find a better way to do this
  return k.call(out);
}

Tensor mul_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k("aten::mul", {self, other});
  return k.call();
}

Tensor& mul_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k("aten::mul_", {self, other});
  return k.call(self);
}

Tensor div_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;

  LazyBinaryOp<at::Tensor> k{
      "aten::div",
      {self, other},
      {},
      {BinaryOperator::compute_output_shape(self, other)}};
  return k.call();
}
Tensor& div_tensor_hpu_lazy_out(
    Tensor& out,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{
      "hpu::div_out",
      {out, self, other},
      {},
      {BinaryOperator::compute_output_shape(self, other)}};
  return k.call(out);
}

Tensor& div_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;

  LazyBinaryOp<at::Tensor&> k{"aten::div_", {self, other}};
  return k.call(self);
}

Tensor div_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k{"aten::div", {self, other}};
  return k.call();
}

Tensor& div_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor&> k{"aten::div_", {self, other}};
  return k.call(self);
}

Tensor pow_tensor_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyBinaryOp<at::Tensor> k{
      "aten::pow",
      {self, other},
      {},
      {BinaryOperator::compute_output_shape(self, other)}};
  return k.call();
}

Tensor& pow_tensor_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyBinaryOp<Tensor&> k("aten::pow_", {self, other});
  return k.call(self);
}

Tensor pow_tensor_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::pow", {self, other}};
  return k.call();
}

Tensor& pow_tensor_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::pow_", {self, other}};
  return k.call(self);
}

Tensor pow_scalar_tensor_hpu_lazy(Scalar other, const Tensor& self) {
  HABANA_ASSERT(0);
  return pow_scalar_tensor_hpu(other, self);
}

Tensor maximum_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(other, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::maximum"),
      {hl_self.GetIrValue(), hl_other.GetIrValue()});
  auto shape_out = BinaryOperator::compute_output_shape(self, other);
  auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  // updatet the view if any
  updateDstDependencies(hlresult, result);

  std::vector<at::Tensor> input_pt_vec{self, other};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor minimum_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(other, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::minimum"),
      {hl_self.GetIrValue(), hl_other.GetIrValue()});
  auto shape_out = BinaryOperator::compute_output_shape(self, other);
  auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  // updatet the view if any
  updateDstDependencies(hlresult, result);

  std::vector<at::Tensor> input_pt_vec{self, other};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}
Tensor gt_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyCompareOp<at::Tensor> k{
      "aten::gt",
      {self, other},
      {},
      {CompareWrapperOperator::compute_output_shape(self, other)}};
  return k.call();
}

Tensor gt_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  PT_LAZY_TRACE;
  LazyCompareOp<at::Tensor> k{
      "aten::gt", {self, other}, {}, {self.sizes().vec()}};
  return k.call();
}

Tensor& eq_tensor_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    const Tensor& other) {
  HABANA_ASSERT(0);
  return eq_tensor_out_hpu(output, self, other);
}
Tensor eq_tensor_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  PT_LAZY_TRACE;
  LazyCompareOp<at::Tensor> k{
      "aten::eq", {self, other}, {}, {self.sizes().vec()}};
  return k.call();
}

Tensor eq_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyCompareOp<at::Tensor> k{
      "aten::eq",
      {self, other},
      {},
      {CompareWrapperOperator::compute_output_shape(self, other)}};
  return k.call();
}

Tensor ne_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  PT_LAZY_TRACE;
  LazyCompareOp<at::Tensor> k{
      "aten::ne", {self, other}, {}, {self.sizes().vec()}};
  return k.call();
}

Tensor ne_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyCompareOp<at::Tensor> k{
      "aten::ne",
      {self, other},
      {},
      {CompareWrapperOperator::compute_output_shape(self, other)}};
  return k.call();
}

Tensor all_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::all"), {hl_self.GetIrValue()});

  // Output of torch.all is single dimension
  std::vector<int64_t> shape_out{1};
  auto result = at::native::empty_hpu_lazy(
      shape_out,
      self.options().dtype(c10::ScalarType::Bool),
      self.suggest_memory_format(),
      false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);
  flush_op(result);
  return result;
}

Tensor all_dim_hpu_lazy(const Tensor& self, int64_t dim, bool keepdim) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::AllDim>(self, dim, keepdim);

  // Infer Output shape
  auto shape_out = self.sizes().vec();
  if (keepdim == true) {
    shape_out[dim] = 1;
  } else {
    shape_out.erase(shape_out.begin() + dim);
  }
  auto result = at::native::empty_hpu_lazy(
      shape_out,
      self.options().dtype(c10::ScalarType::Bool),
      self.suggest_memory_format(),
      false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);
  flush_op(result);
  return result;
}

Tensor lt_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  PT_LAZY_TRACE;
  LazyCompareOp<at::Tensor> k{
      "aten::lt", {self, other}, {}, {self.sizes().vec()}};
  return k.call();
}

Tensor lt_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyCompareOp<at::Tensor> k{
      "aten::lt",
      {self, other},
      {},
      {CompareWrapperOperator::compute_output_shape(self, other)}};
  return k.call();
}

Tensor ge_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  PT_LAZY_TRACE;
  LazyCompareOp<at::Tensor> k{
      "aten::ge", {self, other}, {}, {self.sizes().vec()}};
  return k.call();
}

Tensor ge_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyCompareOp<at::Tensor> k{
      "aten::ge",
      {self, other},
      {},
      {CompareWrapperOperator::compute_output_shape(self, other)}};
  return k.call();
}

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

  LazyOp<at::Tensor> k(
      "aten::convolution_overrideable",
      {input,
       weight,
       bias,
       stride,
       padding,
       dilation,
       transposed,
       output_padding,
       groups},
      {3, 4, 5, 6, 7, 8},
      {ConvOperator::compute_output_shape(
          input.sizes().vec(),
          weight.sizes().vec(),
          padding.vec(),
          stride.vec(),
          false,
          transposed,
          c10::MemoryFormat::Contiguous)});
  return k.call();
}

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

  // Construct using LazyOp templated with class habana_lazy::ir::Convolution
  std::vector<bool> output_mask_vec(output_mask.begin(), output_mask.end());
  ir::NodePtr node = std::make_shared<habana_lazy::ir::Convolution>(
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

  using T = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
  using U = habana_lazy::ir::Convolution;
  class Kernel : public LazyOp<T, U> {
   public:
    Kernel(
        ir::NodePtr node,
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor weight)
        : LazyOp<T, U>(std::move(node), {}, {}, -1),
          grad_output{std::move(grad_output)},
          input{std::move(input)},
          weight{std::move(weight)} {}

   private:
    std::tuple<at::Tensor, at::Tensor, at::Tensor> get_result_overrideable()
        override {
      c10::MemoryFormat memory_format =
          habana_helpers::get_memory_format({&input, &grad_output, &weight});
      auto grad_input = at::native::empty_hpu_lazy(
          input.sizes(), grad_output.options(), memory_format, false);
      auto grad_weight = at::native::empty_hpu_lazy(
          weight.sizes(), grad_output.options(), memory_format, false);
      auto grad_bias = at::native::empty_hpu_lazy(
          {grad_output.size(1)}, grad_output.options(), memory_format, false);
      return {grad_input, grad_weight, grad_bias};
    }
    Tensor grad_output;
    Tensor input;
    Tensor weight;
  };

  Kernel k(node, grad_output, input, weight);
  return k.call();
}

Tensor constant_pad_hpu_lazy(
    const Tensor& self,
    IntArrayRef pad,
    Scalar value) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{
      "aten::constant_pad_nd",
      {self, pad, value},
      {1, 2},
      {PadOperator::compute_output_shape(self, pad)}};
  return k.call();
}
Tensor embedding_hpu_lazy(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  habana_lazy::ir::NodePtr embedding_node =
      std::make_shared<habana_lazy::ir::Embedding_forward>(
          weight, indices, padding_idx, scale_grad_by_freq, sparse);

  // allocate Output storage
  auto size = indices.sizes().vec();

  if (indices.dim() == 1) {
    // TORCH_CHECK(false, "Lazy Emedding: indices of dim 1 not expected");
    size = weight.sizes().vec();
  }

  else {
    // append size of last N-1 dimensions of weight (assuming its a Nd tensor)
    for (auto d : weight.sizes().slice(1)) {
      size.push_back(d);
    }
  }
  auto result = at::native::empty_hpu_lazy(
      size, weight.options(), weight.suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(embedding_node);

  flush_op(result);
  return result;
}
Tensor embedding_dense_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  habana_lazy::ir::NodePtr embedding_bwd_node =
      std::make_shared<habana_lazy::ir::Embedding_backward>(
          grad, indices, num_weights, padding_idx, scale_grad_by_freq);

  // allocate Output storage
  auto result = at::native::empty_hpu_lazy(
      {num_weights, grad.size(-1)},
      grad.options(),
      grad.suggest_memory_format(),
      false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(embedding_bwd_node);

  flush_op(result);
  return result;
}
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
  updateDstDependencies(hlresult, result);
  flush_op(result);
  return result;
}
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
  static_cast<void>(valid_count_bwd);
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
  updateDstDependencies(hlresult, result);
  std::vector<at::Tensor> input_pt_vec{
      input,
      indices_fwd,
      offsets_fwd,
      valid_count,
      indices_bwd,
      offsets_bwd,
      grad_weight};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}
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

  flush_op(out);
  return out;
}
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
  flush_op(out);
  return out;
}

Tensor& fill_hpu_lazy_(Tensor& self, Scalar value) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::fill_", {self, value}};
  return k.call(self);
}

Tensor& masked_fill_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  TORCH_CHECK(
      value.dim() == 0, "value supports only 0D tensor to match CPU behavior");
  auto mask_expand = mask;
  if (self.sizes() != mask.sizes()) {
    // this explicit broadcast can be removed when
    // binary kernels start supporting broadcase
    mask_expand = mask.expand(self.sizes());
  }

  TORCH_CHECK(
      self.sizes() == mask_expand.sizes(),
      "input & mask tensor shapes not matching");
  auto new_mask = mask_expand.to(self.dtype());
  // create a inverted mask
  auto zero_tensor = at::zeros_like(
      new_mask, new_mask.options(), new_mask.suggest_memory_format());
  auto inv_mask = at::eq(new_mask, zero_tensor).to(self.dtype());
  auto value_expand = value.expand(self.sizes());

  LazyBinaryOp<Tensor&> op("aten::mul_", {self, inv_mask});
  op.call(self);

  LazyBinaryOp<Tensor&> op1("aten::mul_", {new_mask, value_expand});
  op1.call(new_mask);

  Scalar alpha = 1.f;
  LazyBinaryOp<at::Tensor&> k{"aten::add_", {self, new_mask, alpha}};
  return k.call(self);
}
Tensor& masked_fill_scalar_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    Scalar value) {
  PT_LAZY_TRACE;
  auto mask_expand = mask;
  if (self.sizes() != mask.sizes()) {
    // this explicit broadcast can be removed when
    // binary kernels start supporting broadcase
    mask_expand = mask.expand(self.sizes());
  }

  TORCH_CHECK(
      self.sizes() == mask_expand.sizes(),
      "input & mask tensor shapes not matching");
  auto new_mask = mask_expand.to(self.dtype());
  // create a inverted mask
  auto zero_tensor = at::zeros_like(
      new_mask, new_mask.options(), new_mask.suggest_memory_format());
  auto inv_mask = at::eq(new_mask, zero_tensor).to(self.dtype());
  auto value_expand = torch::full(self.sizes(), value.toFloat());

  LazyBinaryOp<Tensor&> op("aten::mul_", {self, inv_mask});
  op.call(self);

  LazyBinaryOp<Tensor&> op1("aten::mul_", {new_mask, value_expand});
  op1.call(new_mask);

  Scalar alpha = 1.f;
  LazyBinaryOp<at::Tensor&> k{"aten::add_", {self, new_mask, alpha}};
  return k.call(self);
}
Tensor gather_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad) {
  HABANA_ASSERT(0);
  return gather_src_hpu(self, dim_, index, sparse_grad);
}
Tensor& scatter_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  HABANA_ASSERT(0);
  return scatter_inplace_src_hpu(self, dim_, index, src);
}

Tensor& scatter_inplace_value_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    Scalar value) {
  PT_LAZY_TRACE;

  auto hl_self = habana_lazy::GetHbLazyTensor(self);
  auto node =
      std::make_shared<habana_lazy::ir::ScatterValue>(self, dim_, index, value);

  // Create result tensor to store output of scatter node
  auto result = at::native::empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetOrCreateHbLazyTensor(result, c10::kHABANA);
  habana_lazy::ir::Value& res = hl_result.CurrentIrValue();
  res.m_index = 0;
  res.SetNode(node);

  // Add a control_edge node
  updateDstDependencies(hl_self, self, true);

  // Create MemCopy operator to copy value into self
  std::vector<at::Tensor> input_pt_vec;
  auto node2 = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::habana_d2d_memcpy_other"),
      {hl_result.GetIrValue(), hl_self.GetIrValue()});
  input_pt_vec.push_back(result);
  input_pt_vec.push_back(self);
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      self.device().index());
  context->MarkTensorRegistered(hl_self.getTensorUniqueId());
  habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node2);
  node2->AddInputPtTensors(input_pt_vec);

  flush_op(self);
  return self;
}
Tensor scatter_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  HABANA_ASSERT(0);
  return scatter_src_hpu(self, dim_, index, src);
}
Tensor scatter_add_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  HABANA_ASSERT(0);
  return scatter_add_src_hpu(self, dim_, index, src);
}
Tensor& scatter_add_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  HABANA_ASSERT(0);
  return scatter_add_inplace_src_hpu(self, dim_, index, src);
}
Tensor index_hpu_lazy(const at::Tensor& self, at::TensorList indices) {
  PT_LAZY_TRACE;
  // fallback to cpu for boolean indexing
  // https://jira.habana-labs.com/browse/SW-37171
  if (indices[0].scalar_type() == c10::ScalarType::Bool) {
    return AtenHpuTypeDefault::index(self, indices);
  }
  // cast input to fp32, int32 not supported yet
  habana_lazy::HbLazyTensor hl_self =
      habana_lazy::GetOrCreateHbLazyTensor(self, self.device());
  at::Tensor self_cast = self;
  // Remove this cast node once TPC kernel is available
  // JIRA <https://jira.habana-labs.com/browse/SW-37171>
  if (self.scalar_type() != c10::ScalarType::Float) {
    auto node = std::make_shared<habana_lazy::ir::Cast>(
        self, c10::ScalarType::Float, true);
    at::TensorOptions hb_options = self.options().dtype(c10::ScalarType::Float);
    self_cast = at::native::empty_hpu_lazy(
        self.sizes(), hb_options, self.suggest_memory_format(), false);
    auto hl_cast = habana_lazy::GetHbLazyTensor(self_cast);
    habana_lazy::ir::Value& out = hl_cast.CurrentIrValue();
    out.m_index = 0;
    out.SetNode(node);
  }
  LazyOp<at::Tensor> k{
      "aten::index",
      {self_cast, indices},
      {},
      {IndexOperator::compute_output_shape(self_cast, indices)}};
  auto result = k.call();

  if (self.scalar_type() != c10::ScalarType::Float) {
    at::TensorOptions hb_options = self.options();
    auto type = self.scalar_type();

    if (self.scalar_type() == c10::ScalarType::Long) {
      type = c10::ScalarType::Int;
      hb_options = hb_options.dtype(c10::ScalarType::Int);
    }
    auto node = std::make_shared<habana_lazy::ir::Cast>(result, type, true);
    auto result_cast = at::native::empty_hpu_lazy(
        result.sizes(), hb_options, result.suggest_memory_format(), false);

    auto hl_cast = habana_lazy::GetHbLazyTensor(result_cast);
    habana_lazy::ir::Value& out = hl_cast.CurrentIrValue();
    out.m_index = 0;
    out.SetNode(node);
    updateDstDependencies(hl_cast, result_cast);
    flush_op(result_cast);
    return result_cast;
  }
  return result;
}

Tensor& _index_put_impl_hpu_lazy_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate,
    const bool unsafe) {
  PT_LAZY_TRACE;
  // index backward is not supported on hpu, indices needs to be
  // bool, byte or long type for cpu fallback
  if (indices[0].scalar_type() == c10::ScalarType::Int) {
    auto tensorlist = indices.vec();
    std::vector<Tensor> indices_long;
    for (size_t i = 0; i < tensorlist.size(); i++) {
      auto long_tensor = habana_helpers::cast_tensor_to_long(tensorlist[i]);
      indices_long.push_back(long_tensor);
    }
    return AtenHpuTypeDefault::_index_put_impl_(
        self, indices_long, value, accumulate, unsafe);
  }
  return AtenHpuTypeDefault::_index_put_impl_(
      self, indices, value, accumulate, unsafe);
}

Tensor nonzero_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  auto input_shape = self.sizes();
  int dimensions = input_shape.size();
  int elements = self.numel();
  at::TensorOptions hb_options = self.options();
  hb_options = hb_options.dtype(c10::ScalarType::Int);

  // Handle case for empty tensor where we return empty tensor with size
  if (elements == 0) {
    auto shape = DimVector{0, dimensions};
    auto output = at::native::empty_hpu_lazy(
        shape, hb_options, self.suggest_memory_format(), true);
    auto hl_output = habana_lazy::GetHbLazyTensor(output);
    updateDstDependencies(hl_output, output);
    flush_op(output);
    return output;
  }

  // Add nonzero node
  std::vector<int64_t> output_shape{elements, dimensions};
  std::vector<int64_t> shape_tensor_shape{5};
  using T = std::tuple<at::Tensor, at::Tensor>;
  LazyOp<T> k("aten::nonzero", {self}, {}, {output_shape, shape_tensor_shape});
  // nonzero returns 2 output where and shape tensor
  auto result_nonzero = k.call();
  auto where_tensor = std::get<0>(result_nonzero);
  auto shape_tensor = std::get<1>(result_nonzero);

  // Select second element from shape tensor
  auto node_slice =
      std::make_shared<habana_lazy::ir::Slice>(shape_tensor, 0, 1);
  auto end_shape = DimVector{1};
  auto end_tensor = at::native::empty_hpu_lazy(
      end_shape, hb_options, self.suggest_memory_format(), false);
  auto hl_end = habana_lazy::GetHbLazyTensor(end_tensor);
  habana_lazy::ir::Value& end_out = hl_end.CurrentIrValue();
  end_out.m_index = 0;
  end_out.SetNode(node_slice);
  // Force an exections here to capture second element of shape tensor.
  // This element is required to determine shape of next node's output
  std::vector<HbLazyTensor> hl_flush_end = {hl_end};
  HbLazyTensor::SyncTensorsGraph(&hl_flush_end);
  auto end = end_tensor.item<int64_t>();

  // Handle case for all False where we return empty tensor with size
  if (end == 0) {
    auto shape = DimVector{0, dimensions};
    auto output = at::native::empty_hpu_lazy(
        shape, hb_options, self.suggest_memory_format(), true);
    auto hl_output = habana_lazy::GetHbLazyTensor(output);
    updateDstDependencies(hl_output, output);
    flush_op(output);
    return output;
  }

  // Add a slice node to capture relevent elements from nonzero node
  // in case we have relevant elements
  auto sliced_shape = DimVector{end, dimensions};
  auto node =
      std::make_shared<habana_lazy::ir::Slice>(where_tensor, 0, 0, end, 1);
  auto result = at::native::empty_hpu_lazy(
      sliced_shape, hb_options, self.suggest_memory_format(), true);
  auto hl_result = habana_lazy::GetOrCreateHbLazyTensor(result, c10::kHABANA);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  // Flushing to match eager mode execution
  std::vector<HbLazyTensor> hl_flush_result = {hl_result};
  HbLazyTensor::SyncTensorsGraph(&hl_flush_result);
  return result;
}
Tensor& index_add_hpu_lazy_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source) {
  PT_LAZY_TRACE;
  auto hl_result = habana_lazy::GetHbLazyTensor(self);
  updateDstDependencies(hl_result, self, true);
  auto node =
      std::make_shared<habana_lazy::ir::IndexAdd_>(self, dim_, indices, source);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self, indices, source};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(self);
  return self;
}
Tensor index_put_hpu_lazy(
    const Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  PT_LAZY_TRACE;
  // Remove CPU fallback once TPC kernel for index_backward is available
  // JIRA <https://jira.habana-labs.com/browse/SW-37171>
  if (indices[0].scalar_type() == c10::ScalarType::Int) {
    auto tensorlist = indices.vec();
    std::vector<Tensor> indices_long;
    for (size_t i = 0; i < tensorlist.size(); i++) {
      indices_long.push_back(
          habana_helpers::cast_tensor_to_long(tensorlist[i]));
    }
    return AtenHpuTypeDefault::index_put(self, indices_long, value, accumulate);
  }
  return AtenHpuTypeDefault::index_put(self, indices, value, accumulate);
}
Tensor& index_put_hpu_lazy_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  PT_LAZY_TRACE;
  // Remove CPU fallback once TPC kernel for index_backward is available
  // JIRA <https://jira.habana-labs.com/browse/SW-37171>
  if (indices[0].scalar_type() == c10::ScalarType::Int) {
    auto tensorlist = indices.vec();
    std::vector<Tensor> indices_long;
    for (size_t i = 0; i < tensorlist.size(); i++) {
      indices_long.push_back(
          habana_helpers::cast_tensor_to_long(tensorlist[i]));
    }
    return AtenHpuTypeDefault::index_put_(
        self, indices_long, value, accumulate);
  }
  return AtenHpuTypeDefault::index_put_(self, indices, value, accumulate);
}
Tensor index_select_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<habana_lazy::ir::IndexSelect>(self, dim, index);

  auto result = index_select_hpu(self, dim, index);

  auto hl_result = habana_lazy::GetOrCreateHbLazyTensor(result, c10::kHABANA);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{self, index};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}
Tensor gather2d_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  HABANA_ASSERT(0);
  return gather2d_hpu(input, indices, validCount);
}
Tensor slice_hpu_lazy(
    const Tensor& self_in,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  PT_LAZY_TRACE;
  if (self_in.dim() <= 1 && step == 1) {
    return at::native::slice(self_in, dim, start, end, step);
  }
  // WA for https://jira.habana-labs.com/browse/SW-37197
  auto self = self_in;
  auto dim_orig = dim;
  if ((dim == self_in.dim() - 1) && (step > 1)) {
    self = transpose_hpu_lazy(self_in, self_in.dim() - 1, self_in.dim() - 2);
    dim = self.dim() - 2;
  }
  auto node =
      std::make_shared<habana_lazy::ir::Slice>(self, dim, start, end, step);

  auto shape = SliceOperator::compute_output_shape(self, dim, start, end, step);
  auto result = at::native::empty_hpu_lazy(
      shape, self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);

  // Handling the case for empty self tensor
  // slice_hpu returns correct tensor shape of NULL result
  // so, don't add the node, just return the empty result
  if (self.numel() == 0) {
    updateDstDependencies(hl_result, result);
    flush_op(result);
    return result;
  }
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  auto output = result;
  // WA for https://jira.habana-labs.com/browse/SW-37197
  if ((dim_orig == self_in.dim() - 1) && (step > 1)) {
    output = transpose_hpu_lazy(result, result.dim() - 1, result.dim() - 2);
  }
  flush_op(output);
  return output;
}

Tensor select_hpu_lazy(const Tensor& self, int64_t dim, int64_t index) {
  PT_LAZY_TRACE;
  auto node = std::make_shared<habana_lazy::ir::Slice>(self, dim, index);

  // infer shape
  auto result = select_hpu(self, dim, index);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);

  // Handling the case for empty self tensor
  // select_hpu returns correct tensor shape of NULL result
  // so, don't add the node, just return the empty result
  if (self.numel() == 0) {
    updateDstDependencies(hl_result, result);
    flush_op(result);
    return result;
  }
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor& arange_hpu_lazy(Tensor& output, Scalar start, Scalar end, Scalar step) {
  PT_LAZY_TRACE;
  auto hl_result = habana_lazy::GetOrCreateHbLazyTensor(output, c10::kHABANA);
  auto hl_start = habana_lazy::GetIrValueForScalar(start);
  auto hl_end = habana_lazy::GetIrValueForScalar(end);
  auto hl_step = habana_lazy::GetIrValueForScalar(step);

  // resizing the output as it is coming as empty from model
  int out_depth = ArangeOperator::GetOutputSize(start, end, step);
  auto out_shape = DimVector({out_depth});
  auto out_reshaped = hl_result.getAttachedTensorImpl();
  THHTensor_resizeNd(out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  output.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::arange_out"),
      {hl_result.GetIrValue(), hl_start, hl_end, hl_step});

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  // updatet the view if any
  updateDstDependencies(hl_result, output);
  std::vector<at::Tensor> input_pt_vec{output};
  node->AddInputPtTensors(input_pt_vec);
  flush_op(output);
  return output;
}

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
  updateDstDependencies(hlresult, result);
  std::vector<at::Tensor> input_pt_vec{mat1, mat2};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

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
  updateDstDependencies((habana_lazy::HbLazyTensor&)hlresult, (Tensor&)result);
  std::vector<at::Tensor> input_pt_vec{self, mat1, mat2};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

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

  auto hlresult = habana_lazy::GetHbLazyTensor(out);
  habana_lazy::ir::Value& out_val = hlresult.CurrentIrValue();
  out_val.m_index = 0;
  out_val.SetNode(node);
  // updatet the view if any
  updateDstDependencies(hlresult, out);
  std::vector<at::Tensor> input_pt_vec{self, mat2};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(out);
  return out;
}

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
  updateDstDependencies((habana_lazy::HbLazyTensor&)hlresult, (Tensor&)result);
  std::vector<at::Tensor> input_pt_vec{self, mat2};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor dot_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  // TODO: Need to make 0-dim output.
  // With ... shape_out = {0}; following error reported
  // Assertion (dst.nbytes() >= src.nbytes()) is false!
  // ... /pytorch-integration/habana_kernels/basic_kernels.cpp:147 copy_hpu_
  // terminate called without an active exception
  // Aborted (core dumped)
  std::vector<int64_t> shape_out = {1};
  LazyOp<at::Tensor> k{"aten::dot", {self, other}, {}, {shape_out}};
  return k.call();
}

Tensor mv_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  std::vector<int64_t> shape_out = {self.size(0)};
  LazyOp<at::Tensor> k{"aten::mv", {self, other}, {}, {shape_out}};
  return k.call();
}

std::tuple<Tensor, Tensor> nll_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  PT_LAZY_TRACE;

  using T = std::tuple<at::Tensor, at::Tensor>;
  LazyOp<T> k(
      "aten::nll_loss_forward",
      {self, target, weight, reduction, ignore_index},
      {3, 4}, // metadata_indices
      {{1}, {}} // out_shapes
  );
  return k.call();
}

Tensor nll_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k(
      "aten::nll_loss_backward",
      {grad_output,
       self,
       target,
       weight,
       reduction,
       ignore_index,
       total_weight},
      {4, 5}, /* metadata_indices */
      {self.sizes().vec()} /* out_shapes*/);

  return k.call();
}

Tensor mse_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k(
      "aten::mse_loss",
      {self, target, reduction},
      {2}, // metadata_indices
      {MSELossFwdOperator::compute_output_shape(self, reduction)});
  return k.call();
}

Tensor mse_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> k(
      "aten::mse_loss_backward",
      {grad_output, self, target, reduction},
      {3}, // metadata_indices
      {mse_loss_backward_hpu(grad_output, self, target, reduction)
           .sizes()
           .vec()});
  return k.call();
}

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
  updateDstDependencies(hlresult, result);
  flush_op(result);
  return result;
}

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
  updateDstDependencies(hlresult, result);
  flush_op(result);
  return result;
}

Tensor binary_cross_entropy_with_logits_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight,
    const c10::optional<Tensor>& pos_weight,
    int64_t reduction) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr bce_loss_node =
      std::make_shared<habana_lazy::ir::BceLogitsLoss_forward>(
          self, target, weight, pos_weight, reduction);

  // allocate Output
  Tensor result;
  if (reduction == at::Reduction::Reduction::None) {
    result = at::native::empty_hpu_lazy(
        self.sizes(), self.options(), self.suggest_memory_format(), false);
  } else {
    result = at::native::empty_hpu_lazy(
        {1}, self.options(), self.suggest_memory_format(), false);
  }

  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(bce_loss_node);
  updateDstDependencies(hlresult, result);
  flush_op(result);
  return result;
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_hpu_lazy(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    bool training,
    double momentum,
    double eps) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node;

  if (training) {
    node = std::make_shared<habana_lazy::ir::BatchNormForward>(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps);
  } else {
    // weight and bias positions are swapped to match TPC kernel signature
    node = std::make_shared<habana_lazy::ir::BatchNormInf>(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps);
  }

  auto output_sizes = input.sizes().vec();
  auto sizes = std::make_tuple(
      output_sizes, running_mean.sizes().vec(), running_var.sizes().vec());
  auto mf = input.suggest_memory_format();

  // Get Output Image
  auto result_img = at::native::empty_hpu_lazy(
      std::get<0>(sizes), input.options(), mf, false);
  const auto hlresult0 = habana_lazy::GetHbLazyTensor(result_img);
  habana_lazy::ir::Value& out0 = hlresult0.CurrentIrValue();
  out0.m_index = 0;
  out0.SetNode(node);

  // TODO add support for track_running_stats = false
  HABANA_ASSERT(running_mean.defined());

  // set the running mean and variance as output nodes
  if (running_mean.defined() && training) {
    const auto hlresult1 = habana_lazy::GetHbLazyTensor(running_mean);
    habana_lazy::ir::Value& out1 = hlresult1.CurrentIrValue();
    out1.m_index = 1;
    out1.SetNode(node);

    const auto hlresult2 = habana_lazy::GetHbLazyTensor(running_var);
    habana_lazy::ir::Value& out2 = hlresult2.CurrentIrValue();
    out2.m_index = 2;
    out2.SetNode(node);
  }

  Tensor result_mean, result_var;
  if (training) {
    // Get output mean and var
    result_mean = at::native::empty_hpu_lazy(
        std::get<1>(sizes), running_mean.options(), mf, false);
    const auto hlresult3 = habana_lazy::GetHbLazyTensor(result_mean);
    habana_lazy::ir::Value& out3 = hlresult3.CurrentIrValue();
    out3.m_index = 3;
    out3.SetNode(node);

    result_var = at::native::empty_hpu_lazy(
        std::get<2>(sizes), running_var.options(), mf, false);
    const auto hlresult4 = habana_lazy::GetHbLazyTensor(result_var);
    habana_lazy::ir::Value& out4 = hlresult4.CurrentIrValue();
    out4.m_index = 4;
    out4.SetNode(node);
  }

  if (training) {
    flush_op({result_img, result_mean, result_var});
    return std::make_tuple(result_img, result_mean, result_var);
  } else {
    flush_op({result_img, running_mean, running_var});
    return std::make_tuple(result_img, running_mean, running_var);
  }
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_bwd_hpu_lazy(
    const Tensor& grad_out,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& running_mean,
    const Tensor& running_var,
    const Tensor& save_mean,
    const Tensor& save_invstd,
    bool train,
    double eps,
    UNUSED std::array<bool, 3> output_mask) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::BatchNormBackward>(
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

  auto output_sizes = input.sizes().vec();
  auto sizes = std::make_tuple(
      output_sizes, running_mean.sizes().vec(), running_var.sizes().vec());

  at::Tensor result_1, result_2, result_3;
  if (output_mask[0]) {
    result_1 = at::native::empty_hpu_lazy(
        std::get<0>(sizes),
        input.options(),
        input.suggest_memory_format(),
        false);
    const auto hlresult_1 = habana_lazy::GetHbLazyTensor(result_1);
    habana_lazy::ir::Value& out_1 = hlresult_1.CurrentIrValue();
    out_1.m_index = 0;
    out_1.SetNode(node);
  }

  if (output_mask[1]) {
    result_2 = at::native::empty_hpu_lazy(
        std::get<1>(sizes),
        weight.options(),
        weight.suggest_memory_format(),
        false);
    const auto hlresult_2 = habana_lazy::GetHbLazyTensor(result_2);
    habana_lazy::ir::Value& out_2 = hlresult_2.CurrentIrValue();
    out_2.m_index = 1;
    out_2.SetNode(node);
  }

  if (output_mask[2]) {
    result_3 = at::native::empty_hpu_lazy(
        std::get<2>(sizes),
        save_mean.options(),
        save_mean.suggest_memory_format(),
        false);
    const auto hlresult_3 = habana_lazy::GetHbLazyTensor(result_3);
    habana_lazy::ir::Value& out_3 = hlresult_3.CurrentIrValue();
    out_3.m_index = 2;
    out_3.SetNode(node);
  }

  flush_op({result_1, result_2, result_3});
  return std::make_tuple(result_1, result_2, result_3);
}

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
  updateDstDependencies((habana_lazy::HbLazyTensor&)hlresult, result_img);
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
  updateDstDependencies((habana_lazy::HbLazyTensor&)hlresult2, result_mean);
  auto result_var = at::native::empty_hpu_lazy(
      std::get<2>(sizes),
      input.options(),
      input.suggest_memory_format(),
      false);
  const auto hlresult3 = habana_lazy::GetHbLazyTensor(result_var);
  habana_lazy::ir::Value& out3 = hlresult3.CurrentIrValue();
  out3.m_index = 2;
  out3.SetNode(node);
  updateDstDependencies((habana_lazy::HbLazyTensor&)hlresult3, result_var);
  flush_op({result_img, result_mean, result_var});
  return std::make_tuple(result_img, result_mean, result_var);
}
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
  updateDstDependencies((habana_lazy::HbLazyTensor&)hlresult, result_dY);
  at::Tensor result2, result3;
  // Check if to return optional results
  if (grad_input_mask[1]) {
    result2 = at::native::empty_hpu_lazy(
        std::get<1>(sizes),
        gamma.options(),
        gamma.suggest_memory_format(),
        false);
    const auto hlresult2 = habana_lazy::GetHbLazyTensor(result2);
    habana_lazy::ir::Value& out2 = hlresult2.CurrentIrValue();
    out2.m_index = 1;
    out2.SetNode(node);
  }
  if (grad_input_mask[2]) {
    result3 = at::native::empty_hpu_lazy(
        std::get<2>(sizes),
        gamma.options(),
        gamma.suggest_memory_format(),
        false);
    const auto hlresult3 = habana_lazy::GetHbLazyTensor(result3);
    habana_lazy::ir::Value& out2 = hlresult3.CurrentIrValue();
    out2.m_index = 2;
    out2.SetNode(node);
  }
  flush_op({result_dY, result2, result3});
  return std::make_tuple(result_dY, result2, result3);
}

Tensor norm_scalar_hpu_lazy(const Tensor& self, Scalar p) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{
      "aten::norm",
      {self, p},
      {},
      {NormOperator::compute_output_shape(self, p)}};
  return k.call();
}

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

  // shape inferrence
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, false);

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
  updateDstDependencies(hlresult_0, result_0);

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
  updateDstDependencies(hlresult_1, result_1);

  flush_op({result_0, result_1});
  return {result_0, result_1};
}
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
}
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
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, false);

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
  updateDstDependencies(hlresult, result);
  flush_op(result);
  return result;
}

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

  // shape inference
  std::vector<int64_t> d{1, 1};
  IntArrayRef dilation(d.data(), d.size());
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, false);

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
  updateDstDependencies(hlresult, result);
  flush_op(result);
  return result;
}

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
}
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
  std::vector<int64_t> d{1, 1}; // setting dilation to 1 for avg pool
  IntArrayRef dilation(d.data(), d.size());
  auto opsize_nhwc = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, false);

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
  updateDstDependencies(hlresult, result);
  flush_op(result);
  return result;
}

Tensor adaptive_avg_pool2d_hpu_lazy(
    const Tensor& input,
    IntArrayRef output_size) {
  PT_LAZY_TRACE;
  auto opsize_nhwc =
      PoolHelper::compute_output_shape(input, output_size, false);
  // compute_output_shape return always nhwc. convert to nchw
  std::vector<long int> shape_out = {
      opsize_nhwc.at(0),
      opsize_nhwc.at(3),
      opsize_nhwc.at(1),
      opsize_nhwc.at(2)};

  LazyOp<Tensor> k{
      "aten::_adaptive_avg_pool2d", {input, output_size}, {1}, {shape_out}};
  return k.call();
}

Tensor adaptive_avg_pool2d_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<Tensor> k{
      "aten::_adaptive_avg_pool2d_backward",
      {grad_output, input},
      {},
      {input.sizes().vec()}};
  return k.call();
}

Tensor& uniform_hpu_lazy(
    Tensor& self,
    double from,
    double to,
    c10::optional<Generator> gen) {
  PT_LAZY_TRACE;
  LazyOp<Tensor&> op{
      "aten::uniform_", {self, from, to, std::move(gen)}, {1, 2, 3}};
  return op.call(self);
}
Tensor& normal_hpu_lazy(
    Tensor& self,
    double mean,
    double std,
    c10::optional<Generator> gen) {
  PT_LAZY_TRACE;
  LazyOp<Tensor&> op{
      "aten::normal_", {self, mean, std, std::move(gen)}, {1, 2, 3}};
  return op.call(self);
}
Tensor& randperm_hpu_lazy(
    Tensor& output,
    int64_t n,
    c10::optional<Generator> gen) {
  PT_LAZY_TRACE;

  // resizing the output as it is coming as empty from model
  auto hl_result = habana_lazy::GetOrCreateHbLazyTensor(output, c10::kHABANA);
  auto out_shape = DimVector({n});
  auto out_reshaped = hl_result.getAttachedTensorImpl();
  THHTensor_resizeNd(out_reshaped, out_shape.size(), out_shape.data(), nullptr);
  output.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));

  LazyOp<Tensor&> op{
      "hpu::randperm_out",
      {output, Scalar((int32_t)n), std::move(gen)},
      {2},
      {{n}}};

  return op.call(output);
}
Tensor bernoulli_hpu_lazy(const Tensor& self, c10::optional<Generator> gen) {
  HABANA_ASSERT(0);
  return bernoulli_hpu(self, gen);
}
Tensor& bernoulli_scalar_hpu_lazy(
    Tensor& self,
    double p,
    c10::optional<Generator> gen) {
  HABANA_ASSERT(0);
  return bernoulli_scalar_hpu(self, p, gen);
}

std::tuple<Tensor, Tensor> fused_dropout_hpu_lazy(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen) {
  PT_LAZY_TRACE;
  auto res = fused_dropout_hpu(self, p, gen);
  using T = std::tuple<at::Tensor, at::Tensor>;
  FusedDropout<T> k(
      {self, p, std::move(gen)}, {1, 2} // metadata_indices
  );

  return k.call();
}

at::Tensor repeat_hpu_lazy(const at::Tensor& self, at::IntArrayRef repeats) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{
      "aten::repeat",
      {self, repeats},
      {1},
      {RepeatOperator::compute_output_shape(self, repeats)}};
  return k.call();
}

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
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor& sum_IntList_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  HABANA_ASSERT(0);
  return sum_IntList_out_hpu(output, self, dim, keepdim, dtype);
}
Tensor mean_dim_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;

  std::bitset<64> dim_mask;
  if (dim.empty()) {
    dim_mask = std::bitset<64>().flip();
  } else {
    size_t ndims = self.dim();
    for (int64_t k : dim) {
      size_t dim = c10::maybe_wrap_dim(k, ndims);
      dim_mask[dim] = true;
    }
  }
  std::vector<int64_t> shape = self.sizes().vec();
  for (int64_t dim = shape.size() - 1; dim >= 0; dim--) {
    if (dim_mask[dim]) {
      if (keepdim) {
        shape[dim] = 1;
      } else {
        shape.erase(shape.begin() + dim);
      }
    }
  }

  LazyOp<at::Tensor> k(
      "aten::mean",
      {self, dim, keepdim, std::move(dtype)},
      {1, 2, 3}, // metadata_indices
      {shape});
  return k.call();
}
Tensor& mean_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  HABANA_ASSERT(0);
  return mean_dim_out_hpu(output, self, dim, keepdim, dtype);
}

Tensor sum_hpu_lazy(const Tensor& self_in, c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  auto self = self_in;
  // Cast Boolean/Char (I8) inputs to Float since TPC kernel supports only f32
  if (self_in.scalar_type() == c10::ScalarType::Bool ||
      self_in.scalar_type() == c10::ScalarType::Char) {
    c10::ScalarType dst_dtype = c10::ScalarType::Float;
    self = at::native::empty_hpu_lazy(
        self_in.sizes(),
        self_in.options().dtype(dst_dtype),
        self_in.suggest_memory_format(),
        false);
    self = copy_hpu_lazy_(self, self_in, true);
  }

  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Sum>(self, dtype);
  auto result = at::native::empty_hpu_lazy(
      {0}, self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  flush_op(result);
  result.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  return result;
}

Tensor mean_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  HABANA_ASSERT(0);
  return mean_hpu(self, dtype);
}

Tensor prod_dim_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::ProdDimInt>(self, dim, keepdim, dtype);

  // Infer Output shape
  auto shape_out = self.sizes().vec();
  if (keepdim == true) {
    shape_out[dim] = 1;
  } else {
    shape_out.erase(shape_out.begin() + dim);
  }

  auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor prod_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Prod>(self, dtype);

  // Output of Prod is product of all elements
  std::vector<int64_t> shape_out{1};
  auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor& any_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  HABANA_ASSERT(0);
  return any_dim_out_hpu(output, self, dim, keepdim);
}
Tensor any_dim_hpu_lazy(const Tensor& self, int64_t dim, bool keepdim) {
  HABANA_ASSERT(0);
  return any_dim_hpu(self, dim, keepdim);
}
Tensor any_hpu_lazy(const Tensor& self) {
  HABANA_ASSERT(0);
  return any_hpu(self);
}
Tensor argmax_hpu_lazy(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::ArgMax>(self, dim, keepdim);
  std::vector<int64_t> shape_out;

  if (dim.has_value()) {
    shape_out = self.sizes().vec();
    if (keepdim == true) {
      shape_out[dim.value()] = 1;
    } else {
      shape_out.erase(shape_out.begin() + dim.value());
    }
  } else {
    shape_out.push_back(1);
  }

  auto result = at::native::empty_hpu_lazy(
      shape_out,
      self.options().dtype(c10::ScalarType::Int),
      self.suggest_memory_format(),
      false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}
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
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}
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
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

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
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

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
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}
} // namespace habana
namespace at {
namespace native {

Tensor empty_hpu_lazy(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format,
    bool create_storage) {
  PT_LAZY_TRACE;
  c10::optional<MemoryFormat> mem_format = optional_memory_format.has_value()
      ? optional_memory_format
      : options.memory_format_opt();
  auto original_dtype = options.dtype();
  auto type = typeMetaToScalarType(original_dtype);
  // Dont allocate 8 bytes for double/long as we are anyway going to cast at
  // CPU and then copy to device @ 4byts per element
  type = type == c10::ScalarType::Long ? c10::ScalarType::Int : type;
  type = type == c10::ScalarType::Double ? c10::ScalarType::Float : type;
  auto new_dtype = scalarTypeToTypeMeta(type);

  if (create_storage) {
    c10 ::Allocator* allocator;
    if (options.pinned_memory()) {
      TORCH_CHECK(false, "habana allocator doesn't supported pinned memory");
    } else {
      allocator = habana::getHABANADeviceAllocator();
    }
    int64_t nelements = prod_intlist(size);
    int elem_size = new_dtype.itemsize();
    auto storage_impl = c10::make_intrusive<StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nelements,
        allocator->allocate(nelements * elem_size),
        allocator,
        /*resizeable=*/true);
    Tensor at_internal_tensor =
        habana_lazy::AtenInternalHbTensor(std::move(storage_impl), new_dtype);
    // Setup the tensor sizes & strides for tensor with dim = 4, else for now
    // assuming contiguous
    if ((4 == size.size()) && mem_format.has_value()) {
      at_internal_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          size, CalculateStrides(size, mem_format.value()));
    } else {
      at_internal_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }

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
              size,
              0,
              options.device(),
              c10::typeMetaToScalarType(original_dtype));
      at_tensor = habana_lazy::AtenFromHbLazyTensor(hb_tensor);

      // The lazy tensor will have a reference to the internal tensor
      hb_tensor.SetTensorData(at_internal_tensor);

      // Setup the tensor sizes & strides for tensor with dim = 4, else for
      // now assuming contiguous
      if ((4 == size.size()) && mem_format.has_value()) {
        at_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
            size, CalculateStrides(size, mem_format.value()));
      } else {
        at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
      }

      // Keep a pointer to the storageless tensor from the internal tensor
      auto at_internal_impl =
          habana_lazy::GetHbInternalTensorImpl(at_internal_tensor);
      HABANA_ASSERT(at_internal_impl != nullptr);

      // As its an inplace op and we want this op to execute
      // we want to wind back status of this tensor to registered
      // so that when post order is created, we actually execute it
      // auto context =
      //    habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      //        options.device().index());
      // context->MarkTensorStatus(
      //    hb_tensor.getTensorUniqueId(), LazyTensorExecutionStatus::kINPUT);
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
            size,
            0,
            options.device(),
            c10::typeMetaToScalarType(original_dtype));
    Tensor at_tensor = habana_lazy::AtenFromHbLazyTensor(hb_tensor);
    // Setup the tensor sizes & strides for tensor with dim = 4, else for now
    // assuming contiguous
    if ((4 == size.size()) && mem_format.has_value()) {
      at_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
          size, CalculateStrides(size, mem_format.value()));
    } else {
      at_tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    }
    return at_tensor;
  }
}
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
}
} // namespace native
} // namespace at
Tensor clone_hpu_lazy(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format) {
  PT_LAZY_TRACE;
  static_cast<void>(memory_format);
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
  updateDstDependencies(hlresult, result);
  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);
  flush_op(result);
  return result;
}
Tensor& zero_hpu_lazy(Tensor& self) {
  return fill_hpu_lazy_(self, 0);
}
Tensor cat_hpu_lazy(const TensorList tensors, int64_t dim_) {
  PT_LAZY_TRACE;

  struct Kernel : public LazyOp<at::Tensor> {
    explicit Kernel(const at::TensorList tensors, int64_t dim)
        : LazyOp<at::Tensor>("aten::cat", {tensors, dim}, {1}, {}, -1),
          tensors{tensors},
          dim{dim} {}
    at::Tensor get_result_overrideable() override {
      auto first_tensor = tensors[0];

      auto shape_out = first_tensor.sizes().vec();
      shape_out[dim] = 0;
      auto tensor_count = tensors.size();
      for (unsigned i = 0; i < tensor_count; i++) {
        shape_out[dim] += tensors[i].sizes()[dim];
      }
      return at::native::empty_hpu_lazy(
          shape_out,
          first_tensor.options(),
          first_tensor.suggest_memory_format(),
          false);
    }
    const TensorList tensors;
    int64_t dim;
  };
  Kernel k{tensors, dim_};
  return k.call();
}

Tensor& cat_hpu_lazy_out(
    Tensor& result,
    const TensorList tensors,
    int64_t dim_) {
  HABANA_ASSERT(0);
  return cat_hpu_out(result, tensors, dim_);
}

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
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor& transpose_hpu_lazy_(Tensor& self, int64_t dim0_, int64_t dim1_) {
  return AtenHpuTypeDefault::transpose_(self, dim0_, dim1_);
}

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
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor& t_hpu_lazy_(Tensor& self) {
  HABANA_ASSERT(0);
  return t_hpu_(self);
}

void adjustPTSizesLazy(Tensor& t) {
  // PT expects metadata like sizes and strides same as in NCHW,
  // but data permuted for channel last, so change the size and stride
  // NCHW
  auto sizes = t.sizes().vec();
  std::vector<int> out_pos = {0, 3, 1, 2};
  std::vector<long int> swapped_sizes = {
      sizes[out_pos[0]],
      sizes[out_pos[1]],
      sizes[out_pos[2]],
      sizes[out_pos[3]]};
  t.unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes);
  // For 4D tensors we need to make sure that we generate the PT channel last
  // strides. Also as its a front end tensor, there may be a backend tensor
  // already if so, change dims for that tensor too.
  if (t.dim() == 4) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        c10::MemoryFormat::ChannelsLast);
    auto hl_result = habana_lazy::GetHbLazyTensor(t);
    if (hl_result.getAttachedTensorImpl()) {
      hl_result.getAttachedTensorImpl()->empty_tensor_restride(
          c10::MemoryFormat::ChannelsLast);
    }
  }
}

Tensor permute_cl_hpu_lazy(const Tensor& self, IntArrayRef dims_) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::PermuteCL>(self, dims_);
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      PermuteOperator::compute_output_shape(self, dims_.vec());
  auto result = at::native::empty_strided_hpu_lazy(
      new_sizes, new_strides, self.options(), false);
  adjustPTSizesLazy(result);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  flush_op(result);
  return result;
}

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
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}

Tensor expand_hpu_lazy(const Tensor& self, IntArrayRef size, bool implicit) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Expand>(self, size, implicit);

  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) =
      at::inferExpandGeometry(self.sizes(), self.strides(), size);

  // expandedStrides will be set to 0 by inferExpandGeometry.
  // Since we give back a contiguous tensor, we will set strides
  // to proper values.
  habana_helpers::recalc_strides(expandedStrides, expandedSizes);

  auto result = at::native::empty_strided_hpu_lazy(
      expandedSizes, expandedStrides, self.options(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  flush_op(result);
  return result;
}
std::vector<Tensor> split_with_sizes_hpu_lazy(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  HABANA_ASSERT(0);
  return split_with_sizes_hpu(self, split_sizes, dim);
}
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
  updateDstDependencies(hlresult, result);
  std::vector<at::Tensor> input_pt_vec{grad_output, self};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
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
}
std::tuple<Tensor, Tensor> topk_hpu_lazy(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  PT_LAZY_TRACE;

  using T = std::tuple<at::Tensor, at::Tensor>;
  class Kernel : public LazyOp<T> {
   public:
    Kernel(
        const Tensor& self,
        int64_t k,
        int64_t dim,
        bool largest,
        bool sorted)
        : LazyOp<T>(
              "aten::topk",
              {self, k, dim, largest, sorted},
              {1, 2, 3, 4},
              {},
              -1),
          self(self),
          k(k),
          dim(dim) {}

   private:
    T get_result_overrideable() override {
      auto shape_out = self.sizes().vec();
      int64_t dim_ = c10::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
      shape_out[dim_] = k;
      auto type =
          self.scalar_type() == c10::ScalarType::BFloat16 ? kShort : kInt;

      auto result_0 = at::native::empty_hpu_lazy(
          shape_out, self.options(), self.suggest_memory_format(), false);
      auto result_1 = at::native::empty_hpu_lazy(
          shape_out,
          self.options().dtype(type),
          self.suggest_memory_format(),
          false);
      return {result_0, result_1};
    }
    at::Tensor self;
    int64_t k;
    int64_t dim;
  };

  Kernel kernel{self, k, dim, largest, sorted};
  return kernel.call();
}

std::tuple<Tensor, Tensor> sort_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  PT_LAZY_TRACE;

  habana_lazy::ir::NodePtr node = std::make_shared<habana_lazy::ir::TopK>(
      self, self.size(dim), dim, descending, true);

  auto shape_out = self.sizes().vec();

  // out 0
  auto result_0 = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);

  auto hlresult_0 = habana_lazy::GetHbLazyTensor(result_0);
  habana_lazy::ir::Value& out_0 = hlresult_0.CurrentIrValue();
  out_0.m_index = 0;
  out_0.SetNode(node);

  // out 1
  auto result_1 = at::native::empty_hpu_lazy(
      shape_out,
      self.options().dtype(kInt),
      self.suggest_memory_format(),
      false);
  auto hlresult_1 = habana_lazy::GetHbLazyTensor(result_1);
  habana_lazy::ir::Value& out_1 = hlresult_1.CurrentIrValue();
  out_1.m_index = 1;
  out_1.SetNode(node);

  flush_op({result_0, result_1});
  return std::make_tuple(result_0, result_1);
}

at::Tensor elu_hpu_lazy(
    const at::Tensor& self,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::elu", {self, alpha, scale, input_scale}};
  return k.call();
}

at::Tensor& elu_hpu_lazy_(
    at::Tensor& self,
    at::Scalar alpha,
    at::Scalar scale,
    at::Scalar input_scale) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::elu_", {self, alpha, scale, input_scale}};
  return k.call(self);
}

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
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);
  flush_op(result);
  return result;
}

Tensor& relu_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::relu_", {self}};
  return k.call(self);
}

at::Tensor& leaky_relu_lazy_(at::Tensor& self, at::Scalar negative_slope) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::leaky_relu_", {self, negative_slope}};
  return k.call(self);
}

at::Tensor leaky_relu_backward_lazy(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Scalar negative_slope,
    bool self_is_result) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{
      "aten::leaky_relu_backward",
      {grad_output, self, negative_slope, self_is_result},
      {2, 3}};
  return k.call();
}

at::Tensor leaky_relu_lazy(const at::Tensor& self, at::Scalar negative_slope) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::leaky_relu", {self, negative_slope}};
  return k.call();
}

Tensor sign_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::sign", {input}};
  return k.call();
}

Tensor& sign_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::sign", {input}};
  return k.call(input);
}

Tensor sgn_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  TORCH_CHECK(!input.is_complex(), "Unsupported complex data type provided");
  return sign_hpu_lazy(input);
}

Tensor& sgn_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  TORCH_CHECK(!input.is_complex(), "Unsupported complex data type provided");
  return sign_hpu_lazy_(input);
}

at::Tensor one_hot_hpu_lazy(const Tensor& self, int64_t num_classes) {
  PT_LAZY_TRACE;
  auto shape = self.sizes().vec();

  // empty tensor could be converted to one hot representation,
  // but shape inference is not possible.
  if (self.numel() == 0) {
    if (num_classes <= 0) {
      AT_ERROR("Can not infer total number of classes from empty tensor.");
    } else {
      shape.push_back(num_classes);
      return at::empty(shape, self.options());
    }
  }

  if (num_classes == -1) {
    num_classes = self.max().item().toLong() + 1;
  }

  shape.push_back(num_classes);
  Tensor ret = at::zeros(shape, self.options());
  ret.scatter_(-1, self.unsqueeze(-1), 1);
  return ret;
}

Tensor floor_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::floor"), {hl_input.GetIrValue()});
  auto result = at::native::empty_hpu_lazy(
      input.sizes(), input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor& floor_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);
  auto hl_result = habana_lazy::GetHbLazyTensor(input);
  updateDstDependencies(hl_result, input, true);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::floor"), {hl_input.GetIrValue()});

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

  flush_op(input);
  return input;
}

Tensor log_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::log"), {hl_input.GetIrValue()});
  auto result = at::native::empty_hpu_lazy(
      input.sizes(), input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor& log_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);
  auto hl_result = habana_lazy::GetHbLazyTensor(input);
  updateDstDependencies(hl_result, input, true);
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::log"), {hl_input.GetIrValue()});

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

  flush_op(input);
  return input;
}

Tensor log2_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::log2"), {hl_input.GetIrValue()});
  auto result = at::native::empty_hpu_lazy(
      input.sizes(), input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor& log2_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);
  auto hl_result = habana_lazy::GetHbLazyTensor(input);
  updateDstDependencies(hl_result, input, true);
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::log2"), {hl_input.GetIrValue()});

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

  flush_op(input);
  return input;
}

Tensor upsample_nearest2d_hpu_lazy(
    const Tensor& input,
    c10::optional<at::IntArrayRef> output_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::UpsampleNearest2d>(
          input, output_size, scale_factors);

  auto memory_format = input.suggest_memory_format();
  auto shape_out = UpsampleOperator::compute_output_shape(
      input.sizes().vec(), output_size, scale_factors, memory_format);
  auto result = at::native::empty_hpu_lazy(
      shape_out, input.options(), memory_format, false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  // updatet the view if any
  updateDstDependencies(hlresult, result);
  flush_op(result);
  return result;
}

Tensor upsample_nearest2d_backward_hpu_lazy(
    const Tensor& grad_output,
    c10::optional<at::IntArrayRef> output_size,
    at::IntArrayRef input_size,
    c10::optional<at::ArrayRef<double>> scale_factors) {
  PT_LAZY_TRACE;
  auto memory_format = grad_output.suggest_memory_format();
  std::vector<int64_t> permuted_sizes = input_size.vec();
  permuted_sizes[0] = input_size[0];
  permuted_sizes[1] = input_size[2];
  permuted_sizes[2] = input_size[3];
  permuted_sizes[3] = input_size[1];

  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::UpsampleNearest2dBackward>(
          grad_output, output_size, permuted_sizes, scale_factors);

  auto result = at::native::empty_hpu_lazy(
      input_size, grad_output.options(), memory_format, false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  // updatet the view if any
  updateDstDependencies(hlresult, result);
  flush_op(result);
  return result;
}

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
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

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
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{grad_in, input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

at::Tensor& hardsigmoid_hpu_lazy_(at::Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::hardsigmoid_", {self}};
  return k.call(self);
}

Tensor hardsigmoid_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::hardsigmoid", {input}};
  return k.call();
}

Tensor hardsigmoid_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::hardsigmoid_backward", {grad_output, self}};
  return k.call();
}

// make sqrt as inplace op for workaround in SW-26172
Tensor sqrt_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);
  updateDstDependencies(hl_input, input, true);
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

  flush_op(input);
  return input;
}
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
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

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
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor& tanh_out_hpu_lazy(Tensor& out, const Tensor& self) {
  HABANA_ASSERT(0);
  return tanh_out_hpu(out, self);
}

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
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{grad_in, input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor gelu_hpu_lazy(const Tensor& self) {
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::gelu"), {hl_input.GetIrValue()});

  auto result = at::native::empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor gelu_backward_hpu_lazy(const Tensor& grad, const Tensor& self) {
  auto hl_grad = habana_lazy::GetOrCreateHbLazyTensor(grad, c10::kHABANA);
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::gelu_backward"),
      {hl_grad.GetIrValue(), hl_self.GetIrValue()});

  auto result = at::native::empty_hpu_lazy(
      self.sizes(), self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{grad, self};
  node->AddInputPtTensors(input_pt_vec);
  flush_op(result);
  return result;
}

Tensor& erf_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::erf_", {self}};
  return k.call(self);
}
Tensor erf_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::erf", {self}};
  return k.call();
}
Tensor& exp_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::exp_", {self}};
  return k.call(self);
}

Tensor exp_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::exp", {self}};
  return k.call();
}

Tensor& neg_out_hpu_lazy(Tensor& result, const Tensor& input) {
  HABANA_ASSERT(0);
  return neg_out_hpu(result, input);
}

Tensor& reciprocal_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::reciprocal_", {self}};
  return k.call(self);
}

Tensor reciprocal_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::reciprocal", {self}};
  return k.call();
}

Tensor& reciprocal_out_hpu_lazy(Tensor& result, const Tensor& self) {
  HABANA_ASSERT(0);
  return reciprocal_out_hpu(result, self);
}
Tensor clamp_min_hpu_lazy(const Tensor& self, Scalar min) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::clamp_min", {self, min}};
  return k.call();
}
Tensor& clamp_hpu_lazy_(
    Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  auto hl_result = habana_lazy::GetHbLazyTensor(self);
  updateDstDependencies(hl_result, self, true);
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Clamp>(self, min, max);

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  // As its an inplace op and we want this op to execute
  // we want to wind back status of this tensor to registered
  // so that when post order is created, we actually execute it
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      self.device().index());
  context->MarkTensorRegistered(hl_result.getTensorUniqueId());

  flush_op(self);
  return self;
}

Tensor round_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::round"), {hl_input.GetIrValue()});
  auto result = at::native::empty_hpu_lazy(
      input.sizes(), input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor& round_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);
  auto hl_result = habana_lazy::GetHbLazyTensor(input);
  updateDstDependencies(hl_result, input, true);
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::round"), {hl_input.GetIrValue()});

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      input.device().index());
  context->MarkTensorStatus(
      hl_input.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);

  flush_op(input);
  return input;
}

Tensor rsqrt_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::rsqrt"), {hl_input.GetIrValue()});
  auto result = at::native::empty_hpu_lazy(
      input.sizes(), input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor& rsqrt_hpu_lazy_(Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);
  auto hl_result = habana_lazy::GetHbLazyTensor(input);
  updateDstDependencies(hl_result, input, true);
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::rsqrt"), {hl_input.GetIrValue()});

  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(
      input.device().index());
  context->MarkTensorStatus(
      hl_input.getTensorUniqueId(), LazyTensorExecutionStatus::kREGISTERED);

  flush_op(input);
  return input;
}

Tensor isfinite_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::isfinite"), {hl_input.GetIrValue()});
  auto result = at::native::empty_hpu_lazy(
      input.sizes(),
      input.options().dtype(c10::ScalarType::Bool),
      input.suggest_memory_format(),
      false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  std::vector<at::Tensor> input_pt_vec{input};
  node->AddInputPtTensors(input_pt_vec);

  flush_op(result);
  return result;
}

Tensor clamp_hpu_lazy(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::clamp", {self, min, max}, {1, 2}};
  return k.call();
}

Tensor abs_hpu_lazy(const Tensor& input) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::abs", {input}};
  return k.call();
}

Tensor& abs_hpu_lazy_(Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor&> k{"aten::abs_", {self}};
  return k.call(self);
}

Tensor neg_hpu_lazy(const Tensor& self) {
  PT_LAZY_TRACE;
  LazyOp<at::Tensor> k{"aten::neg", {self}};
  return k.call();
}
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
  flush_op({weights_in, moments_in});
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

  flush_op({weights_in, moments_in});
  return std::tie(weights_in, moments_in);
}

void optimizer_adamw_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& exp_avg,
    TensorList& exp_avg_sq,
    at::Tensor& lr_t,
    at::Tensor& neg_step_t,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay) {
  PT_LAZY_TRACE;
  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = habana_lazy::GetHbLazyTensor(weights[i]);
    updateDstDependencies(hlweight, weights[i], true);
    auto hlexpavg = habana_lazy::GetHbLazyTensor(exp_avg[i]);
    updateDstDependencies(hlexpavg, exp_avg[i], true);
    auto hlexpavgsq = habana_lazy::GetHbLazyTensor(exp_avg_sq[i]);
    updateDstDependencies(hlexpavgsq, exp_avg_sq[i], true);
  }

  auto hl_lr_t = habana_lazy::GetOrCreateHbLazyTensor(lr_t, c10::kHABANA);
  auto hl_neg_step_t =
      habana_lazy::GetOrCreateHbLazyTensor(neg_step_t, c10::kHABANA);

  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::OptimizerFusedAdamw>(
          gradients,
          weights,
          exp_avg,
          exp_avg_sq,
          lr_t,
          neg_step_t,
          beta1,
          beta2,
          epsilon,
          weight_decay);

  int64_t out_index = 0;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hl_exp_avg = habana_lazy::GetHbLazyTensor(exp_avg[i]);
    habana_lazy::ir::Value& out1 = hl_exp_avg.CurrentIrValue();
    out1.m_index = out_index++;
    out1.SetNode(node);

    auto hl_exp_avg_1 = habana_lazy::GetHbLazyTensor(exp_avg[i]);
    habana_lazy::ir::Value& out2 = hl_exp_avg_1.CurrentIrValue();
    out2.m_index = out_index++;
    out2.SetNode(node);

    auto hl_exp_avg_sq = habana_lazy::GetHbLazyTensor(exp_avg_sq[i]);
    habana_lazy::ir::Value& out3 = hl_exp_avg_sq.CurrentIrValue();
    out3.m_index = out_index++;
    out3.SetNode(node);

    auto hl_exp_avg_sq_1 = habana_lazy::GetHbLazyTensor(exp_avg_sq[i]);
    habana_lazy::ir::Value& out4 = hl_exp_avg_sq_1.CurrentIrValue();
    out4.m_index = out_index++;
    out4.SetNode(node);

    auto hl_weight = habana_lazy::GetHbLazyTensor(weights[i]);
    habana_lazy::ir::Value& out5 = hl_weight.CurrentIrValue();
    out5.m_index = out_index++;
    out5.SetNode(node);
  }

  return;
}

Tensor fused_norm_hpu_lazy(
    std::vector<Tensor>& grad,
    const Tensor& max_norm,
    float norm_type) {
  PT_LAZY_TRACE;
  for (size_t i = 0; i < grad.size(); i++) {
    auto hlweight = habana_lazy::GetHbLazyTensor(grad[i]);
    updateDstDependencies(hlweight, grad[i], true);
  }
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::FusedNorm>(grad, max_norm, norm_type);
  int64_t out_index = 0;
  auto result = at::native::empty_hpu_lazy(
      {1}, grad[0].options(), grad[0].suggest_memory_format(), false);

  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = out_index++;
  out.SetNode(node);

  for (size_t i = 0; i < grad.size(); i++) {
    auto hlweight = habana_lazy::GetHbLazyTensor(grad[i]);
    habana_lazy::ir::Value& out1 = hlweight.CurrentIrValue();
    out1.m_index = out_index++;
    out1.SetNode(node);
  }

  flush_op(result);
  return result;
}

Tensor optimizer_lamb_fused_norm_hpu_lazy(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm) {
  PT_LAZY_TRACE;
  auto clip_norm = get_tensor_for_scalar(1.0);
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::LambFusedNorm>(
          grad, max_grad_norm, clip_norm);

  auto result = at::native::empty_hpu_lazy(
      {1}, grad[0].options(), grad[0].suggest_memory_format(), false);
  auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  updateDstDependencies(hlresult, result);

  return result;
}

std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
optimizer_lamb_phase1_hpu_lazy(
    const std::vector<at::Tensor>& gradients,
    std::vector<at::Tensor>& weights,
    std::vector<at::Tensor>& exp_avg,
    std::vector<at::Tensor>& exp_avg_sq,
    const at::Tensor& clip_global_grad_norm,
    const int grad_averaging,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay) {
  PT_LAZY_TRACE;
  static_cast<void>(lr);

  /*
  Assuming mark step is present before optimizer is invoked
  for (size_t i = 0; i < weights.size(); i++) {
    auto hl_grad = habana_lazy::GetHbLazyTensor(gradients[i]);
    updateDstDependencies(hl_grad, gradients[i], true);

    auto hl_wts = habana_lazy::GetHbLazyTensor(weights[i]);
    updateDstDependencies(hl_wts, weights[i], true);

    auto hlexpavg = habana_lazy::GetHbLazyTensor(exp_avg[i]);
    updateDstDependencies(hlexpavg, exp_avg[i], true);

    auto hlexpavgsq = habana_lazy::GetHbLazyTensor(exp_avg_sq[i]);
    updateDstDependencies(hlexpavgsq, exp_avg_sq[i], true);
  }
  */

  auto hl_clip_global = habana_lazy::GetHbLazyTensor(clip_global_grad_norm);
  updateDstDependencies(hl_clip_global, clip_global_grad_norm, true);

  float bias_correction1 = 1.0, bias_correction2 = 1.0;
  if (bias_correction) {
    bias_correction1 = 1.0 - std::pow(beta1, step);
    bias_correction2 = 1.0 - std::pow(beta2, step);
  }

  float beta3 = 1.0;
  if (grad_averaging) {
    beta3 = 1 - beta1;
  }

  auto bias_correction1_t = get_tensor_for_scalar(bias_correction1);
  auto bias_correction2_t = get_tensor_for_scalar(bias_correction2);

  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::OptimizerFusedLambPhase1>(
          gradients,
          weights,
          exp_avg,
          exp_avg_sq,
          clip_global_grad_norm,
          beta1,
          beta2,
          beta3,
          epsilon,
          bias_correction1_t,
          bias_correction2_t,
          weight_decay);

  int64_t out_index = 0;

  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);

  std::vector<Tensor> weight_norm_vec, adam_norm_vec, adam_step_vec;
  for (size_t i = 0; i < weights.size(); i++) {
    auto adam_step = at::native::empty_hpu_lazy(
        weights[i].sizes(),
        weights[i].options(),
        weights[i].suggest_memory_format(),
        false);
    auto hl_adam_step = habana_lazy::GetHbLazyTensor(adam_step);
    habana_lazy::ir::Value& out1 = hl_adam_step.CurrentIrValue();
    out1.m_index = out_index++;
    out1.SetNode(node);

    context->m_retained_tensor_list.emplace_back(adam_step);
    adam_step_vec.push_back(adam_step);

    auto adam_norm = at::native::empty_hpu_lazy(
        {1}, weights[i].options(), weights[i].suggest_memory_format(), false);
    auto hl_adam_norm = habana_lazy::GetHbLazyTensor(adam_norm);
    habana_lazy::ir::Value& out2 = hl_adam_norm.CurrentIrValue();
    out2.m_index = out_index++;
    out2.SetNode(node);

    context->m_retained_tensor_list.emplace_back(adam_norm);
    adam_norm_vec.push_back(adam_norm);

    auto weight_norm = at::native::empty_hpu_lazy(
        {1}, weights[i].options(), weights[i].suggest_memory_format(), false);
    auto hl_weight_norm = habana_lazy::GetHbLazyTensor(weight_norm);
    habana_lazy::ir::Value& out3 = hl_weight_norm.CurrentIrValue();
    out3.m_index = out_index++;
    out3.SetNode(node);

    context->m_retained_tensor_list.emplace_back(weight_norm);
    weight_norm_vec.push_back(weight_norm);

    // add the tensors that are updated inplace
    auto exp_avg_temp = at::native::empty_hpu_lazy(
        exp_avg[i].sizes(),
        exp_avg[i].options(),
        exp_avg[i].suggest_memory_format(),
        false);
    auto hl_exp_avg_temp = habana_lazy::GetHbLazyTensor(exp_avg_temp);
    habana_lazy::ir::Value& out4 = hl_exp_avg_temp.CurrentIrValue();
    out4.m_index = out_index++;
    out4.SetNode(node);
    context->m_retained_tensor_list.emplace_back(exp_avg_temp);

    auto hl_exp_avg = habana_lazy::GetHbLazyTensor(exp_avg[i]);
    habana_lazy::ir::Value& out5 = hl_exp_avg.CurrentIrValue();
    out5.m_index = out_index++;
    out5.SetNode(node);
    context->m_retained_tensor_list.emplace_back(exp_avg[i]);

    auto exp_avg_sq_temp = at::native::empty_hpu_lazy(
        exp_avg_sq[i].sizes(),
        exp_avg_sq[i].options(),
        exp_avg_sq[i].suggest_memory_format(),
        false);
    auto hl_exp_avg_sq_temp = habana_lazy::GetHbLazyTensor(exp_avg_sq_temp);
    habana_lazy::ir::Value& out6 = hl_exp_avg_sq_temp.CurrentIrValue();
    out6.m_index = out_index++;
    out6.SetNode(node);
    context->m_retained_tensor_list.emplace_back(exp_avg_sq_temp);

    auto hl_exp_avg_sq = habana_lazy::GetHbLazyTensor(exp_avg_sq[i]);
    habana_lazy::ir::Value& out7 = hl_exp_avg_sq.CurrentIrValue();
    out7.m_index = out_index++;
    out7.SetNode(node);
    context->m_retained_tensor_list.emplace_back(exp_avg_sq[i]);
  }

  return std::tie(weight_norm_vec, adam_norm_vec, adam_step_vec);
}

void optimizer_lamb_phase2_hpu_lazy(
    std::vector<at::Tensor>& weights,
    const std::vector<at::Tensor>& adam_norm,
    const std::vector<at::Tensor>& weight_norm,
    const std::vector<at::Tensor>& adam_step,
    const std::vector<at::Tensor>& trust_ratio,
    const float step,
    const float weight_decay,
    const int use_lamb) {
  PT_LAZY_TRACE;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hl_weights = habana_lazy::GetHbLazyTensor(weights[i]);
    updateDstDependencies(hl_weights, weights[i], true);
  }

  auto nstep_t = at::tensor(-step).to(c10::kHABANA, true);

  // Build Params for the graph
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::OptimizerFusedLambPhase2>(
          weights,
          adam_norm,
          weight_norm,
          adam_step,
          trust_ratio,
          nstep_t,
          weight_decay,
          use_lamb);

  int64_t out_index = 0;
  for (size_t i = 0; i < weights.size(); i++) {
    auto hl_weight = habana_lazy::GetHbLazyTensor(weights[i]);
    habana_lazy::ir::Value& out1 = hl_weight.CurrentIrValue();
    out1.m_index = out_index++;
    out1.SetNode(node);
  }
}

Tensor& optimizer_adagrad_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  PT_LAZY_TRACE;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = habana_lazy::GetHbLazyTensor(weights[i]);
    updateDstDependencies(hlweight, weights[i], true);

    auto hlvariance = habana_lazy::GetHbLazyTensor(variances[i]);
    updateDstDependencies(hlvariance, variances[i], true);
  }

  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::OptimizerFusedAdagrad>(
          gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);

  int64_t out_index = 0;
  HABANA_ASSERT(weights.size() == variances.size());

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = habana_lazy::GetHbLazyTensor(weights[i]);
    habana_lazy::ir::Value& out1 = hlweight.CurrentIrValue();
    out1.m_index = out_index++;
    out1.SetNode(node);

    auto hlvariance = habana_lazy::GetHbLazyTensor(variances[i]);
    habana_lazy::ir::Value& out2 = hlvariance.CurrentIrValue();
    out2.m_index = out_index++;
    out2.SetNode(node);
  }

  flush_op(lr);
  return lr;
}

Tensor& optimizer_sgd_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_LAZY_TRACE;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = habana_lazy::GetHbLazyTensor(weights[i]);
    updateDstDependencies(hlweight, weights[i], true);
  }

  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::OptimizerFusedSGD>(
          gradients, weights, lr, wd, mom, damp, nesterov);

  int64_t out_index = 0;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = habana_lazy::GetHbLazyTensor(weights[i]);
    habana_lazy::ir::Value& out1 = hlweight.CurrentIrValue();
    out1.m_index = out_index++;
    out1.SetNode(node);
  }

  flush_op(lr);
  return lr;
}

Tensor& optimizer_sgd_momentum_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_LAZY_TRACE;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = habana_lazy::GetHbLazyTensor(weights[i]);
    updateDstDependencies(hlweight, weights[i], true);

    auto hlmomentum = habana_lazy::GetHbLazyTensor(momentum[i]);
    updateDstDependencies(hlmomentum, momentum[i], true);
  }

  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::OptimizerFusedSGDMomentum>(
          gradients, weights, momentum, epoch_num, lr, wd, mom, damp, nesterov);

  int64_t out_index = 0;
  HABANA_ASSERT(weights.size() == momentum.size());

  auto hlweight = habana_lazy::GetHbLazyTensor(weights[0]);
  habana_lazy::ir::Value& out = hlweight.CurrentIrValue();
  node->set_as_output_tensor_list();
  out.m_index = 0;
  out.SetNode(node);

  habana_lazy::ir::NodePtr node_unpack =
      std::make_shared<habana_lazy::ir::ListUnpack>(out);

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = habana_lazy::GetHbLazyTensor(weights[i]);
    habana_lazy::ir::Value& out1 = hlweight.CurrentIrValue();
    out1.m_index = out_index++;
    out1.SetNode(node_unpack);

    auto hlmomentum = habana_lazy::GetHbLazyTensor(momentum[i]);
    habana_lazy::ir::Value& out2 = hlmomentum.CurrentIrValue();
    out2.m_index = out_index++;
    out2.SetNode(node_unpack);
  }

  flush_op(lr);
  return lr;
}

at::Tensor ones_like_hpu_lazy(
    const Tensor& self,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> memory_format) {
  // Note that currently we are not lowering parameters as per the ones_like
  // schema. This works for the ones_like usage in MNIST (where it is used
  // only for filling grad_out tensor with 1's), but we may need to revisit
  // this in future.
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> op(
      "aten::ones_like",
      {self, dtype, layout, device, pin_memory, memory_format},
      {1, 2, 3, 4, 5});
  return op.call();
}

Tensor& bitwise_and_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;
  auto hl_out = habana_lazy::GetOrCreateHbLazyTensor(out, c10::kHABANA);
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(other, c10::kHABANA);
  auto out_shape = BitwiseOutOperator::compute_output_shape(self, other);
  // Resize output tensor(s) to correct shape if required
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = hl_out.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));
  }
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::bitwise_and_Tensor_out"),
      {hl_out.GetIrValue(), hl_self.GetIrValue(), hl_other.GetIrValue()});
  habana_lazy::ir::Value& output = hl_out.CurrentIrValue();
  output.m_index = 0;
  output.SetNode(node);
  // updatet the view if any
  updateDstDependencies(hl_out, out);
  std::vector<at::Tensor> input_pt_vec{out, self, other};
  node->AddInputPtTensors(input_pt_vec);
  flush_op(out);
  return out;
}

Tensor& bitwise_and_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    Scalar other) {
  HABANA_ASSERT(false && "Not implemented yet");
  static_cast<void>(out);
  static_cast<void>(self);
  static_cast<void>(other);
}

Tensor& bitwise_or_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;
  auto hl_out = habana_lazy::GetOrCreateHbLazyTensor(out, c10::kHABANA);
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(other, c10::kHABANA);
  auto out_shape = BitwiseOutOperator::compute_output_shape(self, other);
  // Resize output tensor(s) to correct shape if required
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = hl_out.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));
  }
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("hpu::bitwise_or_Tensor_out"),
      {hl_out.GetIrValue(), hl_self.GetIrValue(), hl_other.GetIrValue()});
  auto hl_result = habana_lazy::GetHbLazyTensor(out);
  habana_lazy::ir::Value& output = hl_result.CurrentIrValue();
  output.m_index = 0;
  output.SetNode(node);
  // updatet the view if any
  updateDstDependencies(hl_result, out);
  std::vector<at::Tensor> input_pt_vec{out, self, other};
  node->AddInputPtTensors(input_pt_vec);
  return out;
};

Tensor& bitwise_xor_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;

  auto hl_out = habana_lazy::GetOrCreateHbLazyTensor(out, c10::kHABANA);
  auto out_shape = BitwiseOutOperator::compute_output_shape(self, other);
  // Resize output tensor(s) to correct shape if required
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = hl_out.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));
  }

  LazyOp<at::Tensor&> k{
      "hpu::bitwise_xor_Tensor_out", {out, self, other}, {}, {out_shape}};
  return k.call(out);
};

Tensor& bitwise_not_out_hpu_lazy(Tensor& out, const Tensor& self) {
  PT_LAZY_TRACE;

  auto hl_out = habana_lazy::GetOrCreateHbLazyTensor(out, c10::kHABANA);
  auto out_shape = self.sizes().vec();
  // Resize output tensor(s) to correct shape if required
  if (out.sizes().vec() != out_shape) {
    auto out_reshaped = hl_out.getAttachedTensorImpl();
    THHTensor_resizeNd(
        out_reshaped, out_shape.size(), out_shape.data(), nullptr);
    out.unsafeGetTensorImpl()->set_sizes_contiguous(IntArrayRef(out_shape));
  }

  LazyOp<at::Tensor&> k{
      "hpu::bitwise_not_Tensor_out", {out, self}, {}, {out_shape}};
  return k.call(out);
};

std::tuple<Tensor, Tensor, Tensor> unique2_hpu_lazy(
    const Tensor& self,
    bool sorted,
    bool return_inverse,
    bool return_counts) {
  PT_LAZY_TRACE;

  int elements = self.numel();
  std::vector<int64_t> feature_map_shape{elements};
  std::vector<int64_t> valid_count_shape{1};
  // Add unique_2 node
  using T = std::tuple<at::Tensor, at::Tensor>;
  Unique<T> k(
      {IValue(self),
       IValue(sorted),
       IValue(return_inverse),
       IValue(return_counts)},
      {1, 2, 3},
      {feature_map_shape, valid_count_shape});
  // unique2 returns 2 output feature_map and valid tensor
  auto output = k.call();
  auto feature_map = std::get<0>(output);
  auto valid_count = std::get<1>(output);
  auto hl_feature_map = habana_lazy::GetHbLazyTensor(feature_map);
  auto hl_valid = habana_lazy::GetHbLazyTensor(valid_count);

  // Force an execution here because "unique" is a non shape inferable op.
  std::vector<HbLazyTensor> hl_flush = {hl_feature_map, hl_valid};
  HbLazyTensor::SyncTensorsGraph(&hl_flush);
  auto end = valid_count.item<int64_t>();

  // Add a slice node to capture relevent elements from feature_map
  auto sliced_shape = DimVector{end};
  auto node_slice =
      std::make_shared<habana_lazy::ir::Slice>(feature_map, 0, 0, end, 1);
  auto result = at::native::empty_hpu_lazy(
      sliced_shape,
      self.options().dtype(self.scalar_type()),
      self.suggest_memory_format(),
      false);
  auto hl_result = habana_lazy::GetOrCreateHbLazyTensor(result, c10::kHABANA);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node_slice);

  // These are optional tensors which shall be populated only when we
  // start supporting return_inverse and return_counts
  Tensor inverse_indices;
  Tensor counts;

  updateDstDependencies(hl_result, result);
  flush_op(result);
  return std::make_tuple(result, inverse_indices, counts);
};

std::tuple<at::Tensor, at::Tensor> max_dim_hpu_lazy(
    const at::Tensor& self,
    int64_t dim,
    bool keepdim) {
  PT_LAZY_TRACE;
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::MaxDim>(self, dim, keepdim);

  // Infer Output shape
  auto shape_out = MaxDimOperator::compute_output_shape(self, dim, keepdim);

  auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto index = at::native::empty_hpu_lazy(
      shape_out,
      self.options().dtype(c10::ScalarType::Int),
      self.suggest_memory_format(),
      false);
  auto hl_result1 = habana_lazy::GetHbLazyTensor(result);
  auto hl_result2 = habana_lazy::GetHbLazyTensor(index);
  habana_lazy::ir::Value& out1 = hl_result1.CurrentIrValue();
  habana_lazy::ir::Value& out2 = hl_result2.CurrentIrValue();
  out1.m_index = 0;
  out1.SetNode(node);
  out2.m_index = 1;
  out2.SetNode(node);
  updateDstDependencies(hl_result1, result);
  updateDstDependencies(hl_result2, index);
  flush_op({result, index});
  return std::make_tuple(result, index);
}

at::Tensor max_hpu_lazy(const at::Tensor& self) {
  PT_LAZY_TRACE;
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::max"), {hl_self.GetIrValue()});

  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);

  // Infer Output shape
  auto shape_out = MaxOperator::compute_output_shape();

  auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  flush_op(result);
  return result;
}
Tensor masked_scale_hpu_lazy(
    const Tensor& self,
    const Tensor& mask,
    double scale) {
  PT_LAZY_TRACE;
  // scale changed to support dropout backward based on what we pass for dropout
  scale = scale / (scale - 1);
  auto masked = mul_tensor_hpu_lazy(self, mask);
  auto scaled = mul_scalar_hpu_lazy(masked, scale);
  return scaled;
}

Tensor matmul_hpu_lazy(const Tensor& self, const Tensor& other) {
  PT_LAZY_TRACE;
  LazyOp<Tensor> k(
      "aten::matmul",
      {self, other},
      {},
      {MatMulOperator::compute_output_shape(self, other)});
  return k.call();
}

std::tuple<Tensor, Tensor> matmul_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& other) {
  PT_LAZY_TRACE;
  LazyOp<std::tuple<Tensor, Tensor>> k(
      "hpu::matmul_backward",
      {grad_output, self, other},
      {},
      {self.sizes().vec(), other.sizes().vec()});
  return k.call();
}

Tensor habana_nms_hpu_lazy(
    const Tensor& boxes,
    const Tensor& scores,
    float iou_threshold,
    float score_threshold) {
  PT_LAZY_TRACE;

  std::vector<int64_t> box_id_out_shape{scores.sizes()[0]};
  std::vector<int64_t> valid_box_id_out_shape{1};
  std::vector<int64_t> shape_tensor_shape{5};
  using T = std::tuple<at::Tensor, at::Tensor, at::Tensor>;
  HabanaNMSLazy<T> k(
      {boxes, scores, Scalar(iou_threshold), Scalar(score_threshold)},
      {box_id_out_shape, valid_box_id_out_shape, shape_tensor_shape});
  auto result_nms = k.call();
  auto box_id_out = std::get<0>(result_nms);
  auto valid_box_id_out = std::get<1>(result_nms);
  auto shape_tensor = std::get<2>(result_nms);
  auto hl_box = habana_lazy::GetHbLazyTensor(box_id_out);
  auto hl_valid = habana_lazy::GetHbLazyTensor(valid_box_id_out);
  auto hl_shape = habana_lazy::GetHbLazyTensor(shape_tensor);

  // Force an execution here to capture valid_box_id_out.
  // This element is required to determine shape of next node's output
  std::vector<HbLazyTensor> hl_flush = {hl_box, hl_valid, hl_shape};
  HbLazyTensor::SyncTensorsGraph(&hl_flush);
  auto end = valid_box_id_out.item<int64_t>();
  // Extract correct output using shape information.
  // Add a slice node to capture relevent elements
  auto sliced_shape = DimVector{end};
  auto node =
      std::make_shared<habana_lazy::ir::Slice>(box_id_out, 0, 0, end, 1);
  auto result = at::native::empty_hpu_lazy(
      sliced_shape,
      scores.options().dtype(c10::ScalarType::Int),
      scores.suggest_memory_format(),
      false);
  auto hl_result = habana_lazy::GetOrCreateHbLazyTensor(result, c10::kHABANA);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  updateDstDependencies(hl_result, result);
  std::vector<HbLazyTensor> hl_flush_result = {hl_result};
  HbLazyTensor::SyncTensorsGraph(&hl_flush_result);
  return result;
}
