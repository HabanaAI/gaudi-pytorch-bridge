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

// Add new src->dst cast mappings to this
std::map<c10::ScalarType, std::vector<c10::ScalarType>> const
    d2d_copy_supported_casts{
        {c10::ScalarType::Float,
         {c10::ScalarType::BFloat16, c10::ScalarType::Int}},
        {c10::ScalarType::BFloat16, {c10::ScalarType::Float}},
        {c10::ScalarType::Char,
         {c10::ScalarType::Float, c10::ScalarType::BFloat16}},
        {c10::ScalarType::Int, {c10::ScalarType::Float}}};

bool copy_transpose_valid(const Tensor& self, const Tensor& src) {
  return (
      self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast &&
      src.numel() != 0 && self.dim() == 4 &&
      self.scalar_type() == src.scalar_type() &&
      src.is_contiguous(c10::MemoryFormat::Contiguous));
}

void adjustPTSizes(Tensor& t) {
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
  // strides
  if (t.dim() == 4) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        c10::MemoryFormat::ChannelsLast);
  }
}

void do_copy_transpose(Tensor& dst, const Tensor& src) {
  int64_t dim_chl_pos[] = {0, 2, 3, 1};
  at::IntArrayRef chl_pos = dim_chl_pos;
  dst = src.permute(chl_pos);
  adjustPTSizes(dst);
}

static void do_d2d_copy(Tensor& dst, const Tensor& src_in, bool non_blocking) {
  // Nothing to do if copy is triggered with same src & dst addresses
  // it actually triggers an assert on func_sim if we trigger this DMA
  // therefore return without doing anything. (this case seen with Mask R-CNN
  // Detectron2 model when copying check-point weights)
  if (dst.data_ptr() == src_in.data_ptr()) {
    return;
  }

  // No direct support for Long in device
  bool same_type = (src_in.scalar_type() == dst.scalar_type());
  auto src = ((src_in.scalar_type() != c10::ScalarType::Long) || same_type)
      ? src_in
      : habana_helpers::cast_tensor_to_integer(src_in);
  auto src_iter = d2d_copy_supported_casts.find(src.scalar_type());
  auto src_scalar_type = src.scalar_type();
  auto dst_scalar_type = dst.scalar_type();
  bool cast_supported = false;
  // cast is possible if src and dst type mapping present in
  // d2d_copy_supported_casts
  if ((src_iter != d2d_copy_supported_casts.end()) &&
      (std::find(
           src_iter->second.begin(), src_iter->second.end(), dst_scalar_type) !=
       src_iter->second.end()))
    cast_supported = true;

  if (cast_supported) { // if supported src->dst mapping
    dst = habana_helpers::hpu_cast_tensor(
        src, at::scalarTypeToTypeMeta(dst_scalar_type));

  } else { // special cases
    if (dst_scalar_type == c10::ScalarType::Long &&
        src_scalar_type == c10::ScalarType::Int) {
      TORCH_CHECK(
          dst.nbytes() >= src.nbytes(),
          "Unsupported device to device copy: dst size needs to be >= src size");
    } else {
      TORCH_CHECK(
          dst.nbytes() == src.nbytes(), "Unsupported device to device copy");
    }
    if (copy_transpose_valid(dst, src)) {
      do_copy_transpose(dst, src);
    } else {
      habana_helpers::copy_data_within_device(src, dst, non_blocking);
    }
  }
}

// cpu->hpu and hpu->cpu copy implementation
Tensor& copy_hpu_(Tensor& self, const Tensor& src, bool non_blocking) {
  PT_KERNEL_BEGIN;
  Tensor& dst = self;
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");
  const auto src_device = src.device().type();
  const auto dst_device = dst.device().type();

  if (src_device == c10::DeviceType::CPU &&
      dst_device == c10::DeviceType::HABANA) {
    HABANA_ASSERT(dst.nbytes() >= src.nbytes());
    habana_helpers::copy_data_to_device(src, dst, non_blocking);
  } else if (
      src_device == c10::DeviceType::HABANA &&
      dst_device == c10::DeviceType::CPU) {
    HABANA_ASSERT(dst.nbytes() >= src.nbytes());
    if (dst.nbytes() > src.nbytes()) {
      // special handling for int to long cast. Needed in saving checkpoints for
      // RN50 lazy The long integer tensor that is used by PT for BN exp
      // averaging (num_batches_tracked) is converted into int in lazy mode. It
      // needs to be converted back to long while saving the checkpoint
      if ((src.scalar_type() == c10::ScalarType::Int) &&
          (dst.scalar_type() == c10::ScalarType::Long)) {
        Tensor dst_tmp = at::empty(
            dst.sizes(),
            at::CPU(at::kInt).options(),
            dst.suggest_memory_format());
        habana_helpers::copy_data_to_host(src, dst_tmp, non_blocking);
        dst = dst_tmp.to(c10::ScalarType::Long);
      } else {
        HABANA_ASSERT(dst.nbytes() != src.nbytes());
      }
    } else {
      habana_helpers::copy_data_to_host(src, dst, non_blocking);
    }
  } else if (
      src_device == c10::DeviceType::HABANA &&
      dst_device == c10::DeviceType::HABANA) {
    do_d2d_copy(dst, src, non_blocking);
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
    if (self_storage) {
      c10::raw::intrusive_ptr::incref(source_storage);
      THTensor_stealAndSetStoragePtr(self_, source_storage);
    } else {
      auto THHStorage_new = []() -> THStorage* {
        THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                                 c10::StorageImpl::use_byte_size_t(),
                                 0,
                                 at::habana::getHABANADeviceAllocator(),
                                 true)
                                 .release();
        return storage;
      };
      THTensor_stealAndSetStoragePtr(self_, THHStorage_new());
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
  // (2) Tensor to.dtype(Tensor self, ScalarType dtype, bool
  // non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) ->
  // Tensor
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
      self.dtype() == c10::ScalarType::Float &&
      type == c10::ScalarType::BFloat16) {
    node_type = "cast_f32_to_bf16";
  } else if (
      self.dtype() == c10::ScalarType::Int && type == c10::ScalarType::Float) {
    node_type = "cast_i32_to_f32";
  } else if (
      (self.dtype() == c10::ScalarType::Bool ||
       self.dtype() == c10::ScalarType::Char) &&
      type == c10::ScalarType::Float) {
    node_type = "cast_i8_to_f32";
  } else if (
      (self.dtype() == c10::ScalarType::Bool ||
       self.dtype() == c10::ScalarType::Char) &&
      type == c10::ScalarType::BFloat16) {
    node_type = "cast_i8_to_bf16";
  } else if (
      self.dtype() == c10::ScalarType::Float &&
      (type == c10::ScalarType::Bool || type == c10::ScalarType::Char)) {
    node_type = "cast_f32_to_i8";
  } else if (self.dtype() == type) {
    // Cases where a simple copy is being done (input_new = input) come as .to
    // call with same input & output data types. we add a identity node to
    // graph to handle this
    IdentityOperator memcopyOp(self.device().index(), self.scalar_type());
    auto& syn_arg0 =
        memcopyOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));

    torch::jit::Stack stack = {IValue(self)};
    memcopyOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    p_context_->syn_inputs_[0] = std::move(syn_arg0);
    p_context_->syn_outputs_.emplace_back(
        std::move(memcopyOp.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(memcopyOp.GetOutputs()[0]));
    return;
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

void CastLazyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for cast operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for toDtype operator");

  auto self = inputs[0].toTensor();
  // auto output = inputs[1].toTensor();
  auto type = inputs[1].toScalarType();

  std::string node_type;
  if (self.dtype() == c10::ScalarType::BFloat16 &&
      type == c10::ScalarType::Float) {
    node_type = "cast_bf16_to_f32";
  } else if (
      type == c10::ScalarType::BFloat16 &&
      self.dtype() == c10::ScalarType::Float) {
    node_type = "cast_f32_to_bf16";
  } else if (
      type == c10::ScalarType::Int && self.dtype() == c10::ScalarType::Float) {
    node_type = "cast_i32_to_f32";
  } else if (
      type == c10::ScalarType::Float && self.dtype() == c10::ScalarType::Int) {
    node_type = "cast_f32_to_i32";
  } else if (
      type == c10::ScalarType::Char && self.dtype() == c10::ScalarType::Float) {
    node_type = "cast_i8_to_f32";
  } else if (
      type == c10::ScalarType::Float && self.dtype() == c10::ScalarType::Char) {
    node_type = "cast_f32_to_i8";
  } else if (
      type == c10::ScalarType::Bool && self.dtype() == c10::ScalarType::Float) {
    node_type = "cast_i8_to_f32";
  } else if (
      type == c10::ScalarType::Float && self.dtype() == c10::ScalarType::Bool) {
    node_type = "cast_f32_to_i8";
  }

  /*
   TODO: This is the Original implementation for Cast Operator
        where we pass the output of the cast as part of the inputs,
        making this as inplace operator. For Resnet we decided
        to make the inplace cast operator as cast out operator.
        The down side of cast out operator is it cannot give us
        the effect of eager mode, cast out operator would create
        a new pytorch output and use that instead of the one that
        is altready created by the .to operator from pytorch.

        Will eventually enable this class as needed going further.

  SetGuid(node_type);

  ns_CastKernel::Params params = synapse_cast_params_builder();
  p_context_->params_.emplace<ns_CastKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  HABANA_ASSERT(p_context_->syn_inputs_.size() == 2);
  synapse_helpers::tensor_or_ref& input_tensor = p_context_->syn_inputs_.back();
  p_context_->syn_outputs_.emplace_back(std::move(input_tensor));
  p_context_->pt_outputs_.emplace_back(output);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));*/

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
  auto self = inputs[0].toTensor();
  at::Tensor output;
  if (inputs.size() == 2) {
    output = inputs[1].toTensor();
    p_context_->syn_outputs_.emplace_back(
        std::move(p_context_->syn_inputs_[1]));
    p_context_->pt_outputs_.emplace_back(output);
  } else {
    output = habana_helpers::createPTTensor(self, is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  }
  p_context_->params_size_ = 0;
  AddNodeToSynapseGraph(graph, NULL, 0);
}

void IdentityOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  auto self = inputs[0].toTensor();
  at::Tensor output;
  if (inputs.size() == 2) {
    output = inputs[1].toTensor();
  } else {
    output = habana_helpers::createPTTensor(self, is_output_persistent);
  }
  p_context_->params_size_ = 0;
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, NULL, 0);
}

/*************************************************************************
 * @brief Kernel implementation for dummy, used for graph ordering
 ************************************************************************/
void DummyOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  static_cast<void>(graph);
  static_cast<void>(is_output_persistent);
  at::Tensor output;
  int out_index = inputs.size() - 1;
  output = inputs[out_index].toTensor();
  p_context_->params_size_ = 0;
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[out_index]));
  p_context_->pt_outputs_.emplace_back(output);
}

Tensor as_strided_hpu(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset) {
  // DeviceGuard omitted
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}

Tensor view_hpu(const Tensor& self, IntArrayRef size) {
  // DeviceGuard omitted
  return at::native::view(self, size);
}

static auto& KernelRegistry =
    ::habana::KernelRegistry()
        .add(
            "hpu::habana_d2d_memcpy",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<MemCopyOperator>(device_id, node_type);
            })
        .add(
            "hpu::habana_d2d_memcpy_other",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<MemCopyOperator>(device_id, node_type);
            })
        .add(
            "hpu::cast",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<CastLazyOperator>(device_id, node_type);
            })
        .add(
            "aten::to",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ToDtypeOperator>(device_id, node_type);
            })
        .add(
            "hpu::control_edge_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<DummyOperator>(device_id, node_type);
            })
        .add(
            "hpu::control_edge_other_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<DummyOperator>(device_id, node_type);
            });
