/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_helpers/logging.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/conv_kernels.h"
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_kernels/pool_kernels.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ops/cat.h"
#include "habana_lazy/ops/convolution.h"
#include "habana_lazy/ops/mse_loss.h"
#include "habana_lazy/ops/pool.h"
#include "habana_lazy/ops/reduce_ops.h"
#include "habana_lazy/ops/softmax.h"
#include "habana_lazy/ops/tensor_shape.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/synapse_helpers/util.h"

/**
 * By default the tensors are created with storage.
 * When a lazy kernel is called for op accumulation, this
 * is set to false
 */
static thread_state_map threadTensorCreateWithStorage({{pthread_self(), true}});

/**
 * When a tensor is created with storage, it could come from
 * the framework (.to operation), or from lowering kernel.
 * By default this is false.
 * When it is called from lowring kernel, only the internal tensor
 * is created with storage.
 * When it is called from other places like .to, the internal tensor
 * is created as well as a lazy tensor. PyTorch gets back a storageless
 * tensor on top of the lazy tensor and the lazy tensor holds a reference
 * to the internal tensor.
 */
static thread_state_map threadInLoweringContext({{pthread_self(), false}});

void AllocateWithStorage() {
  threadTensorCreateWithStorage[pthread_self()] = true;
}

void AllocateWithoutStorage() {
  threadTensorCreateWithStorage[pthread_self()] = false;
}

void SetLoweringContext(bool ctx) {
  threadInLoweringContext[pthread_self()] = ctx;
  synapse_helpers::SetSynHelperLoweringContext(ctx);
}

bool CreateThreadTensorWithStorage() {
  auto ptid = pthread_self();
  if (threadTensorCreateWithStorage.find(ptid) ==
      threadTensorCreateWithStorage.end()) {
    threadTensorCreateWithStorage[ptid] = true;
  }
  return threadTensorCreateWithStorage[ptid];
}

bool IsThreadInLoweringContext() {
  auto ptid = pthread_self();
  if (threadInLoweringContext.find(ptid) == threadInLoweringContext.end()) {
    threadInLoweringContext[ptid] = false;
  }
  return threadInLoweringContext[ptid];
}

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
  // We need to add device to device copy kernel here
  // As d2D copies may not mean trigger execution, we just need to add the
  // nodes like cast to our lazy graph that we are creating
  habana_lazy::HbLazyTensor hb_tensor =
      habana_lazy::GetOrCreateHbLazyTensor(src, src.device());
  // TODO : we can give a more detailed cast info in node name later
  // Will need to move the name generation in a utility(will need to
  // modify kernel too, not touching right now)
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::cast"), {hb_tensor.GetIrValue()});
  self = habana_helpers::hpu_cast_tensor(src, self.dtype());
  auto hlresult = habana_lazy::GetHbLazyTensor(self);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return self;
}

Tensor& copy_hpu_lazy_D2H(Tensor& self, const Tensor& src, bool non_blocking) {
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
      // We need to check the memory deletion of the original tensor created by
      // PT
      self = copy_hpu_(self, tensor_data.value(), non_blocking);
      self = self.to(type);
    } else {
      self = copy_hpu_(self, tensor_data.value(), non_blocking);
    }
    self = habana_lazy::CreateHbLazyTensor(
        self, habana_lazy::GetHblazyDevice(self));
    return self;
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
  bool processed = false;
  auto new_tensor = preProcessIfLongorDouble(src, self, processed);

  // Get the internal tensor for copy kernel
  // First get the lazy tensor
  auto self_hb_tensor = habana_lazy::GetHbLazyTensor(self);
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
  // Return the self tensor, as copy_hpu_ doesn't create a new tensor and
  // returns the dst
  return self;
}

// calling eager mode kernels as a temporary placeholder to avoid warnings
Tensor& copy_hpu_lazy_(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_LAZY_BEGIN;
  if (habana_lazy::IsHbLazyTensor(src)) {
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
  } else if (!src.has_storage()) {
    TORCH_CHECK(
        false,
        "Habana copy_hpu_lazy_: trying to copy from a storage less tensor");
  }
  TORCH_CHECK(self.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");

  const auto src_device = src.device().type();
  const auto dst_device = self.device().type();

  bool is_d2d_copy = false;
  if (src_device == c10::DeviceType::HABANA &&
      dst_device == c10::DeviceType::HABANA) {
    is_d2d_copy = true;
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

  PT_LAZY_END;
  return self;
}
Tensor as_strided_hpu_lazy(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  return as_strided_hpu(self, size, stride, storage_offset);
};
Tensor& set_hpu_lazy_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  return set_hpu_(self, source, storage_offset, size, stride);
};
Tensor view_hpu_lazy(const Tensor& self, IntArrayRef size) {
  return view_hpu(self, size);
};
Tensor addcmul_hpu_lazy(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  return addcmul_hpu(self, tensor1, tensor2, alpha);
};
Tensor& addcmul_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  return addcmul_hpu_(self, tensor1, tensor2, alpha);
};
Tensor addcdiv_hpu_lazy(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  return addcdiv_hpu(self, tensor1, tensor2, alpha);
};
Tensor& addcdiv_hpu_lazy_(
    Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    Scalar alpha) {
  return addcdiv_hpu_(self, tensor1, tensor2, alpha);
};

Tensor add_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
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

  return result;
}

Tensor add_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  return add_scalar_hpu(self, other, alpha);
};
Tensor& add_scalar_hpu_lazy_(Tensor& self, Scalar other, Scalar alpha) {
  return add_scalar_hpu_(self, other, alpha);
};

Tensor& add_tensor_hpu_lazy_(Tensor& self, const Tensor& other, Scalar alpha) {
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_other = habana_lazy::GetOrCreateHbLazyTensor(other, c10::kHABANA);
  auto hl_alpha = habana_lazy::GetIrValueForScalar(alpha);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::add_"),
      {hl_self.GetIrValue(), hl_other.GetIrValue(), hl_alpha});

  habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return self;
};

Tensor sub_tensor_hpu_lazy(
    const Tensor& self,
    const Tensor& other,
    Scalar alpha) {
  return sub_tensor_hpu(self, other, alpha);
};
Tensor& sub_tensor_hpu_lazy_(Tensor& self, const Tensor& other, Scalar alpha) {
  return sub_tensor_hpu_(self, other, alpha);
};
Tensor sub_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  return sub_scalar_hpu(self, other, alpha);
};
Tensor& sub_scalar_hpu_lazy_(Tensor& self, Scalar other, Scalar alpha) {
  return sub_scalar_hpu_(self, other, alpha);
};
Tensor rsub_scalar_hpu_lazy(const Tensor& self, Scalar other, Scalar alpha) {
  return rsub_scalar_hpu(self, other, alpha);
};
Tensor& mul_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  return mul_tensor_hpu_(self, other);
};
Tensor mul_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
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

  return result;
};
Tensor mul_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  return mul_scalar_hpu(self, other);
};
Tensor& mul_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  return mul_scalar_hpu_(self, other);
};
Tensor div_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  return div_tensor_hpu(self, other);
};
Tensor& div_tensor_hpu_lazy_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  return div_tensor_hpu_out(result, self, other);
};
Tensor& div_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  return div_tensor_hpu_(self, other);
};
Tensor div_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  return div_scalar_hpu(self, other);
};
Tensor& div_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  return div_scalar_hpu_(self, other);
};
Tensor pow_tensor_tensor_hpu_lazy(const Tensor& self, const Tensor& other) {
  return pow_tensor_tensor_hpu(self, other);
};
Tensor& pow_tensor_tensor_hpu_lazy_(Tensor& self, const Tensor& other) {
  return pow_tensor_tensor_hpu_(self, other);
};
Tensor pow_tensor_scalar_hpu_lazy(const Tensor& self, Scalar other) {
  return pow_tensor_scalar_hpu(self, other);
};
Tensor& pow_tensor_scalar_hpu_lazy_(Tensor& self, Scalar other) {
  return pow_tensor_scalar_hpu_(self, other);
};
Tensor pow_scalar_tensor_hpu_lazy(Scalar other, const Tensor& self) {
  return pow_scalar_tensor_hpu(other, self);
};
Tensor gt_hpu_lazy(Tensor& self, Tensor& other) {
  return gt_hpu(self, other);
};

void eq_tensor_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    const Tensor& other) {
  return eq_tensor_out_hpu(output, self, other);
};
Tensor eq_tensor_hpu_lazy(Tensor& self, Tensor& other) {
  return eq_tensor_hpu(self, other);
};
Tensor eq_tensor_scalar_hpu_lazy(Tensor& self, Scalar other) {
  return eq_tensor_scalar_hpu(self, other);
};
Tensor lt_scalar_hpu_lazy(Tensor& self, Scalar other) {
  return lt_scalar_hpu(self, other);
};
Tensor lt_tensor_hpu_lazy(Tensor& self, Tensor& other) {
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

  // shape inference
  // convert tensors to synapse memory format nchw to nhwc
  std::vector<int64_t> ipsize_nhwc = {input.sizes().vec().at(0),
                                      input.sizes().vec().at(2),
                                      input.sizes().vec().at(3),
                                      input.sizes().vec().at(1)};
  auto opsize_nhwc = ConvOperator::compute_output_shape(
      ipsize_nhwc,
      weight.sizes().vec(),
      padding.vec(),
      stride.vec(),
      false,
      input.suggest_memory_format());
  // convert from synapse memory format nhwc to nchw
  std::vector<int64_t> shape_out = {opsize_nhwc.at(0),
                                    opsize_nhwc.at(3),
                                    opsize_nhwc.at(1),
                                    opsize_nhwc.at(2)};

  auto result = at::native::empty_hpu_lazy(
      shape_out, input.options(), input.suggest_memory_format(), false);
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
      weight.sizes(), grad_output.options(), memory_format);
  auto grad_input = at::native::empty_hpu_lazy(
      input.sizes(), grad_output.options(), memory_format);
  auto grad_bias = at::native::empty_hpu_lazy(
      {grad_output.size(3)}, grad_output.options(), memory_format);

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

  auto conv_out = std::make_tuple(grad_weight, grad_input, grad_bias);

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
  return constant_pad_hpu(self, pad, value);
};
Tensor embedding_hpu_lazy(
    const Tensor& weight,
    const Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {
  return embedding_hpu(
      weight, indices, padding_idx, scale_grad_by_freq, sparse);
};
Tensor embedding_dense_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  return embedding_dense_backward_hpu(
      grad, indices, num_weights, padding_idx, scale_grad_by_freq);
};
Tensor embedding_bag_sum_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  return embedding_bag_sum_hpu(
      input, indices, offsets, valid_count, kernel_mode);
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
  return result;
};
Tensor& embedding_bag_sum_bwd_out_hpu_lazy(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices_bwd,
    const Tensor& offsets_bwd,
    const Tensor& valid_count_bwd) {
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
  return out;
};
Tensor& fill_hpu_lazy_(Tensor& self, Scalar value) {
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_alpha = habana_lazy::GetIrValueForScalar(value);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::fill_"), {hl_self.GetIrValue(), hl_alpha});

  habana_lazy::ir::Value& out = hl_self.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return self;
};
Tensor& masked_fill_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    const Tensor& value) {
  return masked_fill_hpu_(self, mask, value);
};
Tensor& masked_fill_scalar_hpu_lazy_(
    Tensor& self,
    const Tensor& mask,
    Scalar value) {
  return masked_fill_scalar_hpu_(self, mask, value);
};
Tensor gather_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    bool sparse_grad) {
  return gather_src_hpu(self, dim_, index, sparse_grad);
};
Tensor& scatter_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  return scatter_inplace_src_hpu(self, dim_, index, src);
};
Tensor scatter_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  return scatter_src_hpu(self, dim_, index, src);
};
Tensor scatter_add_src_hpu_lazy(
    const Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  return scatter_add_src_hpu(self, dim_, index, src);
};
Tensor& scatter_add_inplace_src_hpu_lazy(
    Tensor& self,
    int64_t dim_,
    const Tensor& index,
    const Tensor& src) {
  return scatter_add_inplace_src_hpu(self, dim_, index, src);
};
Tensor& index_add_hpu_lazy_(
    Tensor& self,
    int64_t dim_,
    const Tensor& indices,
    const Tensor& source) {
  return index_add_hpu_(self, dim_, indices, source);
};
Tensor index_put_hpu_lazy(
    const Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  return index_put_hpu(self, indices, value, accumulate);
};
Tensor& index_put_hpu_lazy_(
    Tensor& self,
    TensorList indices,
    const Tensor& value,
    bool accumulate) {
  return index_put_hpu_(self, indices, value, accumulate);
};
Tensor index_select_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  return index_select_hpu(self, dim, index);
};
Tensor gather2d_hpu_lazy(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  return gather2d_hpu(input, indices, validCount);
};
Tensor slice_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t end,
    int64_t step) {
  return slice_hpu(self, dim, start, end, step);
};
Tensor select_hpu_lazy(const Tensor& self, int64_t dim, int64_t index) {
  return select_hpu(self, dim, index);
};
Tensor& arange_hpu_lazy(Tensor& output, Scalar start, Scalar end, Scalar step) {
  return arange_hpu(output, start, end, step);
};
Tensor mm_hpu_lazy(const at::Tensor& mat1, const at::Tensor& mat2) {
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

  return result;
};

Tensor addmm_hpu_lazy(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta,
    Scalar alpha) {
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
  const IntArrayRef shape_out = {mat1.size(0), mat2.size(1)};
  const auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format(), false);
  const auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};

Tensor& batch_gemm_out_hpu_lazy(
    Tensor& out,
    const Tensor& self,
    const Tensor& mat2) {
  const auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  const auto hl_mat2 = habana_lazy::GetOrCreateHbLazyTensor(mat2, c10::kHABANA);

  const auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::bmm"),
      {hl_self.GetIrValue(), hl_mat2.GetIrValue()});

  const auto hlresult = habana_lazy::GetHbLazyTensor(out);
  habana_lazy::ir::Value& out_val = hlresult.CurrentIrValue();
  out_val.m_index = 0;
  out_val.SetNode(node);

  return out;
};

Tensor batch_gemm_hpu_lazy(const Tensor& self, const Tensor& mat2) {
  const auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  const auto hl_mat2 = habana_lazy::GetOrCreateHbLazyTensor(mat2, c10::kHABANA);

  const auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::bmm"),
      {hl_self.GetIrValue(), hl_mat2.GetIrValue()});

  auto shape_out = BmmOperator::compute_output_shape(self, mat2);
  const auto result = at::native::empty_hpu_lazy(
      shape_out, self.options(), self.suggest_memory_format());
  const auto hlresult = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};

Tensor dot_hpu_lazy(const Tensor& self, const Tensor& other) {
  return dot_hpu(self, other);
};
Tensor mv_hpu_lazy(const Tensor& self, const Tensor& other) {
  return mv_hpu(self, other);
};
std::tuple<Tensor, Tensor> nll_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  return nll_loss_forward_hpu(self, target, weight, reduction, ignore_index);
};
Tensor nll_loss_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    UNUSED const Tensor& total_weight) {
  return nll_loss_backward_hpu(
      grad_output, self, target, weight, reduction, ignore_index, total_weight);
};

Tensor mse_loss_forward_hpu_lazy(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
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
  return binary_cross_entropy_hpu(self, target, weight, reduction);
};
Tensor binary_cross_entropy_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction) {
  return binary_cross_entropy_backward_hpu(
      grad_output, self, target, weight, reduction);
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
  return layer_norm_hpu(input, weight, bias, m, n, eps);
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
  return layer_norm_backward_hpu(
      dY, X, mean, rstd, gamma, M, N, grad_input_mask);
};
Tensor norm_scalar_hpu_lazy(const Tensor& self, Scalar p) {
  return norm_scalar_hpu(self, p);
};
std::tuple<Tensor, Tensor> max_pool2d_with_indices_hpu_lazy(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  habana_lazy::ir::NodePtr maxpool_node =
      std::make_shared<habana_lazy::ir::MaxPool>(
          input, kernel_size, stride, padding, dilation, ceil_mode);

  // shaper inferrence
  auto shape_out = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, false);

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

  // shaper inferrence
  auto out_shape = PoolHelper::compute_output_shape(
      input, kernel_size, stride, padding, dilation, ceil_mode, false);
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
  return avg_pool2d_hpu(
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
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
  return avg_pool2d_backward_hpu(
      grad_output,
      input,
      kernel_size,
      stride,
      padding,
      ceil_mode,
      count_include_pad,
      divisor_override);
};
void uniform_hpu_lazy(
    const Tensor& self,
    double from,
    double to,
    CPUGenerator* gen) {
  uniform_hpu(self, from, to, gen);
};
void normal_hpu_lazy(
    const Tensor& self,
    double mean,
    double std,
    CPUGenerator* gen) {
  normal_hpu(self, mean, std, gen);
};
Tensor bernoulli_hpu_lazy(const Tensor& self, CPUGenerator* gen) {
  return bernoulli_hpu(self, gen);
};
Tensor& bernoulli_scalar_hpu_lazy(Tensor& self, double p, CPUGenerator* gen) {
  return bernoulli_scalar_hpu(self, p, gen);
};

Tensor sum_dim_IntList_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
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
  return sum_IntList_out_hpu(output, self, dim, keepdim, dtype);
};
Tensor mean_dim_hpu_lazy(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  return mean_dim_hpu(self, dim, keepdim, dtype);
};
Tensor& mean_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> dtype) {
  return mean_dim_out_hpu(output, self, dim, keepdim, dtype);
};

Tensor sum_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Sum>(self, dtype);
  auto result = sum_hpu(self, dtype);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};

Tensor mean_hpu_lazy(const Tensor& self, c10::optional<ScalarType> dtype) {
  return mean_hpu(self, dtype);
};
Tensor& any_dim_out_hpu_lazy(
    Tensor& output,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  return any_dim_out_hpu(output, self, dim, keepdim);
};
Tensor any_dim_hpu_lazy(const Tensor& self, int64_t dim, bool keepdim) {
  return any_dim_hpu(self, dim, keepdim);
};
Tensor any_hpu_lazy(const Tensor& self) {
  return any_hpu(self);
};
namespace habana {
Tensor log_softmax_hpu_lazy(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  auto node =
      std::make_shared<habana_lazy::ir::LogSoftMax>(self, dim, half_to_float);
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
  auto node = std::make_shared<habana_lazy::ir::LogSoftMaxBackward>(
      grad, output, dim, input);
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
    int64_t dim,
    const bool half_to_float) {
  return softmax_hpu(self, dim, half_to_float);
};
Tensor softmax_backward_hpu_lazy(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  return softmax_backward_hpu(grad, output, dim, input);
};
} // namespace habana
namespace at {
namespace native {
Tensor empty_hpu_lazy(
    IntArrayRef size,
    const TensorOptions& options,
    c10::optional<MemoryFormat> optional_memory_format,
    bool create_storage) {
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

    // This call could have come from a .to call and not from a lowering
    // context. In such case, create the lazt tensor.
    if (!IsThreadInLoweringContext()) {
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
      at_internal_impl->set_tensor(&at_tensor);
    }

    // If we are not from lowering context, return the storageless one.
    if (!IsThreadInLoweringContext()) {
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
    const TensorOptions& options) {
  at::Tensor empty_tensor = empty_hpu_lazy(
      size, options, c10::nullopt, CreateThreadTensorWithStorage());
  empty_tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
  return empty_tensor;
};
} // namespace native
} // namespace at
Tensor clone_hpu_lazy(
    const Tensor& self,
    c10::optional<MemoryFormat> memory_format) {
  return clone_hpu(self, memory_format);
};
Tensor& zero_hpu_lazy(Tensor& self) {
  return zero_hpu(self);
};
Tensor cat_hpu_lazy(const TensorList tensors, int64_t dim_) {
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Cat>(tensors, dim_);

  auto result = cat_hpu(tensors, dim_);

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
  return cat_hpu_out(result, tensors, dim_);
};
Tensor transpose_hpu_lazy(const Tensor& self, int64_t dim0_, int64_t dim1_) {
  return transpose_hpu(self, dim0_, dim1_);
};
Tensor& transpose_hpu_lazy_(Tensor& self, int64_t dim0_, int64_t dim1_) {
  return transpose_hpu_(self, dim0_, dim1_);
};

Tensor t_hpu_lazy(const Tensor& self) {
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::t"), {hl_self.GetIrValue()});
  auto result = t_hpu(self);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};

Tensor& t_hpu_lazy_(Tensor& self) {
  return t_hpu_(self);
};

Tensor permute_hpu_lazy(const Tensor& self, IntArrayRef dims_) {
  habana_lazy::ir::NodePtr node =
      std::make_shared<habana_lazy::ir::Permute>(self, dims_);
  auto result = permute_hpu(self, dims_);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);
  return result;
};

Tensor expand_hpu_lazy(const Tensor& self, IntArrayRef size, bool implicit) {
  return expand_hpu(self, size, implicit);
};
std::vector<Tensor> split_with_sizes_hpu_lazy(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim) {
  return split_with_sizes_hpu(self, split_sizes, dim);
};
Tensor threshold_backward_hpu_lazy(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold) {
  auto hl_grad =
      habana_lazy::GetOrCreateHbLazyTensor(grad_output, c10::kHABANA);
  auto hl_self = habana_lazy::GetOrCreateHbLazyTensor(self, c10::kHABANA);
  auto hl_threshold = habana_lazy::GetIrValueForScalar(threshold);

  habana_lazy::ir::ValueList ir_vlaues{
      hl_grad.GetIrValue(), hl_self.GetIrValue(), hl_threshold};

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::threshold_backward"), ir_vlaues);

  at::Tensor result = threshold_backward_hpu(grad_output, self, threshold);

  auto hlresult = habana_lazy::GetHbLazyTensor(result);

  habana_lazy::ir::Value& out = hlresult.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

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
  return topk_out_hpu(values, indices, self, k, dim_, largest, sorted);
};
std::tuple<Tensor, Tensor> topk_hpu_lazy(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  return topk_hpu(self, k, dim, largest, sorted);
};
std::tuple<Tensor, Tensor> sort_hpu_lazy(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  return sort_hpu(self, dim, descending);
};
Tensor unary_op_hpu_lazy(
    const Tensor& input,
    std::string& node_type,
    UnaryOperator* Op) {
  return unary_op_hpu(input, node_type, Op);
};
Tensor unary_backward_op_hpu_lazy(
    const Tensor& grad_in,
    const Tensor& input,
    std::string& node_type,
    UnaryBackwardOperator* Op) {
  return unary_backward_op_hpu(grad_in, input, node_type, Op);
};
Tensor relu_hpu_lazy(const Tensor& input) {
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::relu"), {hl_input.GetIrValue()});
  auto shape_out = input.sizes();
  auto result = at::native::empty_hpu_lazy(
      shape_out, input.options(), input.suggest_memory_format(), false);
  auto hl_result = habana_lazy::GetHbLazyTensor(result);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return result;
};
Tensor& relu_hpu_lazy_(Tensor& input) {
  auto hl_input = habana_lazy::GetOrCreateHbLazyTensor(input, c10::kHABANA);

  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("aten::relu_"), {hl_input.GetIrValue()});
  auto hl_result = habana_lazy::GetHbLazyTensor(input);
  habana_lazy::ir::Value& out = hl_result.CurrentIrValue();
  out.m_index = 0;
  out.SetNode(node);

  return input;
};
Tensor sigmoid_hpu_lazy(const Tensor& input) {
  return sigmoid_hpu(input);
};
Tensor sigmoid_backward_hpu_lazy(const Tensor& grad_in, const Tensor& input) {
  return sigmoid_backward_hpu(grad_in, input);
};
Tensor sqrt_hpu_lazy(const Tensor& input) {
  return sqrt_hpu(input);
};
Tensor tanh_hpu_lazy(const Tensor& input) {
  return tanh_hpu(input);
};
Tensor& tanh_hpu_lazy_(Tensor& self) {
  return tanh_hpu_(self);
};
Tensor& tanh_out_hpu_lazy(Tensor& out, Tensor& self) {
  return tanh_out_hpu(out, self);
};
Tensor tanh_backward_hpu_lazy(const Tensor& grad_in, const Tensor& input) {
  return tanh_backward_hpu(grad_in, input);
};
Tensor gelu_hpu_lazy(const Tensor& self) {
  return gelu_hpu(self);
};
Tensor gelu_backward_hpu_lazy(const Tensor& grad, const Tensor& self) {
  return gelu_backward_hpu(grad, self);
};
Tensor& erf_hpu_lazy_(Tensor& self) {
  return erf_hpu_(self);
};
Tensor erf_hpu_lazy(const Tensor& self) {
  return erf_hpu(self);
};
Tensor& exp_hpu_lazy_(Tensor& self) {
  return exp_hpu_(self);
};
Tensor exp_hpu_lazy(const Tensor& self) {
  return exp_hpu(self);
};
Tensor& neg_out_hpu_lazy(Tensor& result, const Tensor& input) {
  return neg_out_hpu(result, input);
};
Tensor& reciprocal_hpu_lazy_(Tensor& self) {
  return reciprocal_hpu_(self);
};
Tensor reciprocal_hpu_lazy(const Tensor& self) {
  return reciprocal_hpu(self);
};
Tensor& reciprocal_out_hpu_lazy(Tensor& result, const Tensor& self) {
  return reciprocal_out_hpu(result, self);
};
Tensor clamp_min_hpu_lazy(const Tensor& self, Scalar min) {
  return clamp_min_hpu(self, min);
};
Tensor& clamp_hpu_lazy_(
    Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  return clamp_hpu_(self, min, max);
};
Tensor clamp_hpu_lazy(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  return clamp_hpu(self, min, max);
};
Tensor abs_hpu_lazy(const Tensor& self) {
  return abs_hpu(self);
};
Tensor neg_hpu_lazy(const Tensor& self) {
  return neg_hpu(self);
};
namespace at {
namespace native {
Scalar _local_scalar_dense_hpu_lazy(const Tensor& self) {
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
std::tuple<torch::Tensor, torch::Tensor>
optimizer_sparse_sgd_with_valid_count_hpu_lazy(
    const Tensor& gradients,
    const Tensor& weights_in,
    const Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor,
    float mom,
    bool nesterov) {
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

  auto weights_out = at::native::empty_hpu_lazy(
      weights_in.sizes(),
      weights_in.options(),
      weights_in.suggest_memory_format(),
      false);
  auto moments_out = at::native::empty_hpu_lazy(
      moments_in.sizes(),
      moments_in.options(),
      moments_in.suggest_memory_format(),
      false);

  auto hlweights = habana_lazy::GetHbLazyTensor(weights_out);
  habana_lazy::ir::Value& out1 = hlweights.CurrentIrValue();
  out1.m_index = 0;
  out1.SetNode(node);
  auto hlmoments = habana_lazy::GetHbLazyTensor(moments_out);
  habana_lazy::ir::Value& out2 = hlmoments.CurrentIrValue();
  out2.m_index = 1;
  out2.SetNode(node);
  return std::tie(weights_out, moments_out);
}
std::tuple<torch::Tensor, torch::Tensor>
optimizer_sparse_adagrad_with_valid_count_hpu_lazy(
    const Tensor& gradients,
    const Tensor& weights_in,
    const Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor) {
  auto node = habana_lazy::ir::Node::Create(
      Symbol::fromQualString("::habanaOptimizerSparseAdagrad"), {});

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

  auto weights_out = at::native::empty_hpu_lazy(
      weights_in.sizes(),
      weights_in.options(),
      weights_in.suggest_memory_format(),
      false);
  auto moments_out = at::native::empty_hpu_lazy(
      moments_in.sizes(),
      moments_in.options(),
      moments_in.suggest_memory_format(),
      false);

  auto hlweights = habana_lazy::GetHbLazyTensor(weights_out);
  habana_lazy::ir::Value& out1 = hlweights.CurrentIrValue();
  out1.m_index = 0;
  out1.SetNode(node);
  auto hlmoments = habana_lazy::GetHbLazyTensor(moments_out);
  habana_lazy::ir::Value& out2 = hlmoments.CurrentIrValue();
  out2.m_index = 1;
  out2.SetNode(node);
  return std::tie(weights_out, moments_out);
}
