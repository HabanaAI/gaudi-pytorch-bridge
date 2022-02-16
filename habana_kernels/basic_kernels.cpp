/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <synapse_api.h>
#include <torch/script.h>

#include <habana_device/PinnedMemoryAllocator.h>
#include "habana_bridge/kernel/hpu_shape_inference.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"

using namespace torch;

static void print_stride_warning(const Tensor& src, const Tensor& dst) {
  if (src.strides() != dst.strides())
    PT_KERNEL_DEBUG(
        "src device: ",
        src.device(),
        " src.strides(): ",
        src.strides(),
        " src.sizes(): ",
        src.sizes(),
        "\ndst device: ",
        dst.device(),
        " dst.strides(): ",
        dst.strides(),
        " dst.sizes(): ",
        dst.sizes(),
        "\nData will be copied with with basic memcopy so you can expect wrong results");
}

// Add new src->dst cast mappings to this
std::unordered_map<c10::ScalarType, std::vector<c10::ScalarType>> const
    d2d_copy_supported_casts{
        {c10::ScalarType::Byte, {c10::ScalarType::Int, c10::ScalarType::Float}},
        {c10::ScalarType::Float,
         {c10::ScalarType::BFloat16,
          c10::ScalarType::Int,
          c10::ScalarType::Bool}},
        {c10::ScalarType::BFloat16, {c10::ScalarType::Float}},
        {c10::ScalarType::Char,
         {c10::ScalarType::Float, c10::ScalarType::BFloat16}},
        {c10::ScalarType::Bool,
         {c10::ScalarType::Float,
          c10::ScalarType::BFloat16,
          c10::ScalarType::Int}},
        {c10::ScalarType::Int, {c10::ScalarType::Float}}};

bool copy_transpose_valid(const Tensor& self, const Tensor& src) {
  auto is_valid_4d =
      self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast &&
      src.numel() != 0 && self.dim() == 4 &&
      self.scalar_type() == src.scalar_type() &&
      src.is_contiguous(c10::MemoryFormat::Contiguous);

  auto is_valid_5d =
      self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d &&
      src.numel() != 0 && self.dim() == 5 &&
      self.scalar_type() == src.scalar_type() &&
      src.is_contiguous(c10::MemoryFormat::Contiguous);

  return is_valid_4d || is_valid_5d;
}

void adjustPTSizes(Tensor& t) {
  // PT expects metadata like sizes and strides same as in NCHW,
  // but data permuted for channel last, so change the size and stride
  // NCHW
  auto sizes = t.sizes().vec();
  std::vector<int> out_pos = {0, 3, 1, 2};
  std::vector<int> out_pos_5d = {0, 4, 1, 2, 3};
  std::vector<long int> swapped_sizes = {
      sizes[out_pos[0]],
      sizes[out_pos[1]],
      sizes[out_pos[2]],
      sizes[out_pos[3]]};
  std::vector<long int> swapped_sizes_5d = {
      sizes[out_pos_5d[0]],
      sizes[out_pos_5d[1]],
      sizes[out_pos_5d[2]],
      sizes[out_pos_5d[3]],
      sizes[out_pos_5d[4]]};
  if (t.dim() == 5) {
    t.unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes_5d);
  } else {
    t.unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes);
  }
  // For 4D tensors we need to make sure that we generate the PT channel last
  // strides
  if (t.dim() == 4) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        c10::MemoryFormat::ChannelsLast);
  }
  if (t.dim() == 5) {
    t.unsafeGetTensorImpl()->empty_tensor_restride(
        c10::MemoryFormat::ChannelsLast3d);
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
  PT_OTHER_OPS_BEGIN; // this macro is used because this kernel is used from
                      // other Lazy kernels
  Tensor& dst = self;
  TORCH_CHECK(dst.defined(), "dst is undefined");
  TORCH_CHECK(src.defined(), "src is undefined");
  const auto src_device = src.device().type();
  const auto dst_device = dst.device().type();

  Tensor src_contiguous;
  if (src_device == c10::DeviceType::CPU &&
      dst_device == c10::DeviceType::HPU) {
    // CPU/source tensor should have same dtype as dst & should be contiguous
    // before H2D DMA is triggered
    // Backend kernels should not trigger contiguous call
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
      TORCH_CHECK(src.is_contiguous(src.suggest_memory_format()));
      src_contiguous = src.to(dst.scalar_type());
    } else {
      src_contiguous =
          src.to(dst.scalar_type()).contiguous(src.suggest_memory_format());
    }

    TORCH_CHECK(dst.nbytes() >= src_contiguous.nbytes());
    habana_helpers::copy_data_to_device(src_contiguous, dst, non_blocking);
    print_stride_warning(src_contiguous, dst);
  } else if (
      src_device == c10::DeviceType::HPU &&
      dst_device == c10::DeviceType::CPU) {
    // HPU/source tensor should be contiguous before D2H DMA is triggered
    // Backend kernels should not trigger contiguous call
    if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
      TORCH_CHECK(src.is_contiguous(src.suggest_memory_format()));
      src_contiguous = src;
    } else {
      src_contiguous = src.contiguous(src.suggest_memory_format());
    }

    if (src_contiguous.scalar_type() != dst.scalar_type()) {
      // if src & dst dtypes different, create an intermediate CPU tensor of
      // same dtype as src
      // Note this also covers special handling for int -> long or float ->
      // double casts. Needed in saving checkpoints for RN50 lazy The long
      // integer tensor that is used by PT for BN exp averaging
      // (num_batches_tracked) is converted into int in lazy mode. It needs to
      // be converted back to long while saving the checkpoint
      auto dst_intermediate = at::empty_like(
          dst,
          dst.options().dtype(src_contiguous.scalar_type()),
          dst.suggest_memory_format());
      // Is there any reason why this check cannot be strict equality?
      TORCH_CHECK(dst_intermediate.nbytes() >= src_contiguous.nbytes());
      habana_helpers::copy_data_to_host(
          src_contiguous, dst_intermediate, non_blocking);
      dst = dst_intermediate.to(dst.scalar_type());
    } else {
      // Is there any reason why this check cannot be strict equality?
      TORCH_CHECK(dst.nbytes() >= src_contiguous.nbytes());
      habana_helpers::copy_data_to_host(src_contiguous, dst, non_blocking);
    }
    print_stride_warning(src_contiguous, dst);
  } else if (
      src_device == c10::DeviceType::HPU &&
      dst_device == c10::DeviceType::HPU) {
    do_d2d_copy(dst, src, non_blocking);
    print_stride_warning(src, dst);
  } else {
    PT_KERNEL_FATAL(
        "copy_hpu_ doesn't support ", src_device, " to ", dst_device, "copy");
  }

  PT_OTHER_OPS_END;
  return dst;
}

Tensor& set_hpu_(
    Tensor& self,
    Storage source,
    int64_t storage_offset,
    IntArrayRef size,
    IntArrayRef stride) {
  PT_OTHER_OPS_BEGIN;
  if (stride.data()) {
    TORCH_CHECK(size.size() == stride.size(), "inconsistent size/stride sizes");
  }

  auto scalar_type = self.scalar_type();
  auto self_ = checked_dense_tensor_unwrap(
      self, "self", 1, "_th_set_", false, DeviceType::HPU, scalar_type);
// TODO: remove this commented section
// part of 1.5 migration related change - revert once not needed
#if 0
  auto source_ = checked_storage(
      source,
      "source",
      2,
      DeviceType::HPU,
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
      AT_ASSERT(source_storage);

      TORCH_CHECK(
          self_->storage().device() == source_storage->device(),
          "Attempted to set the storage of a tensor on device \"",
          self_->storage().device(),
          "\" to a storage on different device \"",
          source_storage->device(),
          "\".  This is no longer allowed; the devices must match.");
      self_->set_storage_keep_dtype(
          at::Storage(c10::intrusive_ptr<THStorage>::reclaim(source_storage)));
    } else {
      auto THHStorage_new = []() -> THStorage* {
        THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                                 c10::StorageImpl::use_byte_size_t(),
                                 0,
                                 habana::getHABANADeviceAllocator(),
                                 true)
                                 .release();
        return storage;
      };
      AT_ASSERT(THHStorage_new());

      TORCH_CHECK(
          self_->storage().device() == THHStorage_new()->device(),
          "Attempted to set the storage of a tensor on device \"",
          self_->storage().device(),
          "\" to a storage on different device \"",
          THHStorage_new()->device(),
          "\".  This is no longer allowed; the devices must match.");
      self_->set_storage_keep_dtype(at::Storage(
          c10::intrusive_ptr<THStorage>::reclaim(THHStorage_new())));
    }
  }

  TORCH_CHECK(storage_offset >= 0, "Invalid storage offset: ", storage_offset);
  self_->set_storage_offset(storage_offset);

  /* size and stride */
  THHTensor_resizeNd(self_, stride.size(), size.data(), stride.data());

  PT_OTHER_OPS_END;
  return self;
}

void ToDtypeOperator::AllocateAndAddSynapseNode_Helper(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs) {
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

  // Determine cast node_type to use based on src & dst dtypes
  std::string node_type;
  auto is_output_persistent = GetOutputPersistence()[0];

  if ((type != self.scalar_type()) &&
      !((type == c10::ScalarType::Char &&
         self.scalar_type() == c10::ScalarType::Bool) ||
        (type == c10::ScalarType::Bool &&
         self.scalar_type() == c10::ScalarType::Char))) {
    std::pair<c10::ScalarType, c10::ScalarType> type_key{
        self.scalar_type(), type};
    auto iter = habana_helpers::cast_map.find(type_key);
    if (iter != habana_helpers::cast_map.end()) {
      node_type = iter->second;
    } else {
      HABANA_ASSERT(
          0 &&
              "Unsupported Cast operation requested in ToDtypeOperator::AllocateAndAddSynapseNode",
          self.scalar_type(),
          " -> ",
          type);
    }
  } else {
    // Cases where a simple copy is being done (input_new = input) come as .to
    // call with same input & output data types. we add a identity node to
    // graph to handle this
    auto memcopyOp = make_operator<IdentityOperator>(
        self.device().index(), self.scalar_type());
    memcopyOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    memcopyOp->SetOutputMetadata(output_metadata_);

    torch::jit::Stack stack = {IValue(self)};
    memcopyOp->SetOutputPersistence({is_output_persistent});
    memcopyOp->AllocateAndAddSynapseNode_Helper(graph, stack);

    p_context_->syn_outputs_.emplace_back(
        std::move(memcopyOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(memcopyOp->GetOutputs()[0]));
    return;
  }

  // we do not care about last 3 entries dtype conversion, so throw them away
  inputs.pop_back();
  inputs.pop_back();
  inputs.pop_back();

  auto Op = make_operator<CastOperator>(self.device().index(), node_type);
  Op->SetSynapseInput(p_context_->syn_inputs_[0]);
  Op->SetOutputMetadata(output_metadata_);
  Op->SetOutputPersistence({is_output_persistent});
  Op->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
  p_context_->syn_outputs_.emplace_back(std::move(Op->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(Op->GetOutputs()[0]));
}

void CastLazyOperator::AllocateAndAddSynapseNode_Helper(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for cast operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for toDtype operator");

  auto self = inputs[0].toTensor();
  auto is_output_persistent = GetOutputPersistence()[0];

  // auto output = inputs[1].toTensor();
  auto type = inputs[1].toScalarType();

  std::string node_type;
  if (self.scalar_type() != type) {
    std::pair<c10::ScalarType, c10::ScalarType> type_key{
        self.scalar_type(), type};
    auto iter = habana_helpers::cast_map.find(type_key);
    if (iter != habana_helpers::cast_map.end()) {
      node_type = iter->second;
    } else {
      HABANA_ASSERT(
          0 &&
              "Unsupported Cast operation requested in CastLazyOperator::AllocateAndAddSynapseNode: ",
          self.scalar_type(),
          " -> ",
          type);
    }
  } else {
    node_type = "cast_identity";
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

  // Insert the cast node - in case cast is to same type alias, insert an
  // identity op
  if (node_type.compare("cast_identity")) {
    if (self.scalar_type() == c10::ScalarType::BFloat16 &&
        type == c10::ScalarType::Int) {
      auto bf_to_floatOp = make_operator<CastOperator>(
          self.device().index(), "cast_bf16_to_f32");
      auto float_to_intOp =
          make_operator<CastOperator>(self.device().index(), "cast_f32_to_i32");
      bf_to_floatOp->SetSynapseInput(p_context_->syn_inputs_[0]);
      bf_to_floatOp->SetOutputPersistence({false});
      bf_to_floatOp->AllocateAndAddSynapseNode(graph, inputs, false);
      float_to_intOp->SetSynapseInput(bf_to_floatOp->GetSynOutputs()[0]);
      float_to_intOp->SetOutputMetadata(output_metadata_);
      float_to_intOp->SetOutputPersistence({is_output_persistent});
      float_to_intOp->AllocateAndAddSynapseNode(
          graph, inputs, is_output_persistent);
      p_context_->syn_outputs_.emplace_back(
          std::move(float_to_intOp->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(
          std::move(float_to_intOp->GetOutputs()[0]));
    } else if (
        self.scalar_type() == c10::ScalarType::Byte &&
        type == c10::ScalarType::BFloat16) {
      auto byte_to_floatOp =
          make_operator<CastOperator>(self.device().index(), "cast_u8_to_f32");
      auto float_to_bfOp = make_operator<CastOperator>(
          self.device().index(), "cast_f32_to_bf16");
      byte_to_floatOp->SetSynapseInput(p_context_->syn_inputs_[0]);
      byte_to_floatOp->SetOutputPersistence({false});
      byte_to_floatOp->AllocateAndAddSynapseNode(graph, inputs, false);
      float_to_bfOp->SetSynapseInput(byte_to_floatOp->GetSynOutputs()[0]);
      float_to_bfOp->SetOutputMetadata(output_metadata_);
      float_to_bfOp->SetOutputPersistence({is_output_persistent});
      float_to_bfOp->AllocateAndAddSynapseNode(
          graph, inputs, is_output_persistent);
      p_context_->syn_outputs_.emplace_back(
          std::move(float_to_bfOp->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(
          std::move(float_to_bfOp->GetOutputs()[0]));
    } else {
      auto Op = make_operator<CastOperator>(self.device().index(), node_type);
      Op->SetSynapseInput(p_context_->syn_inputs_[0]);
      Op->SetOutputMetadata(output_metadata_);
      Op->SetOutputPersistence({is_output_persistent});
      Op->AllocateAndAddSynapseNode(graph, inputs, is_output_persistent);
      p_context_->syn_outputs_.emplace_back(std::move(Op->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(std::move(Op->GetOutputs()[0]));
    }
  } else {
    auto identityOp = make_operator<IdentityOperator>(
        self.device().index(), self.scalar_type());
    identityOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    identityOp->SetOutputMetadata(output_metadata_);
    torch::jit::Stack stack = {IValue(self)};
    identityOp->SetOutputPersistence({is_output_persistent});
    identityOp->AllocateAndAddSynapseNode_Helper(graph, stack);

    p_context_->syn_outputs_.emplace_back(
        std::move(identityOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(identityOp->GetOutputs()[0]));
  }
}

/*************************************************************************
 * @brief Kernel implementation for memcpy, used for D2D mem transfers
 * @param self - input which needs to be transferred
 * @param dest - Destination tensor
 ************************************************************************/
void MemCopyOperator::AllocateAndAddSynapseNode_Helper(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto is_output_persistent = GetOutputPersistence()[0];
  at::Tensor output;
  if (inputs.size() == 2) {
    output = inputs[1].toTensor();
    synapse_helpers::tensor& output_tensor = p_context_->syn_inputs_[1];
    p_context_->syn_outputs_.emplace_back(output_tensor);
    p_context_->pt_outputs_.emplace_back(output);
  } else {
    output = habana_helpers::createPTTensor(self, is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  }
  p_context_->params_size_ = 0;
  AddNodeToSynapseGraph(graph, NULL, 0);
}

void IdentityOperator::AllocateAndAddSynapseNode_Helper(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto is_output_persistent = GetOutputPersistence()[0];
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
void DummyOperator::AllocateAndAddSynapseNode_Helper(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  auto is_output_persistent = GetOutputPersistence()[0];
  static_cast<void>(graph);
  static_cast<void>(is_output_persistent);
  at::Tensor output;
  int out_index = inputs.size() - 1;
  output = inputs[out_index].toTensor();
  p_context_->params_size_ = 0;
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[out_index], graph));
  p_context_->pt_outputs_.emplace_back(output);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> AsStridedOperator::
    compute_output_shape(
        const Tensor& self,
        IntArrayRef size,
        IntArrayRef stride) {
  std::vector<int64_t> out_size_vec;
  std::vector<int64_t> out_stride_vec;

  if ((self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast) ||
      (self.suggest_memory_format() == c10::MemoryFormat::ChannelsLast3d)) {
    if (size.size() == 4) {
      // NCHW -> NHWC
      const int64_t dim_pos_in[4] = {0, 2, 3, 1};
      for (size_t idx = 0; idx < size.size(); idx++) {
        out_size_vec.emplace_back(size[dim_pos_in[idx]]);
        out_stride_vec.emplace_back(stride[dim_pos_in[idx]]);
      }
    } else if (size.size() == 5) {
      // NCDHW -> NDHWC
      const int64_t dim_pos_in[5] = {0, 2, 3, 4, 1};
      for (size_t idx = 0; idx < size.size(); idx++) {
        out_size_vec.emplace_back(size[dim_pos_in[idx]]);
        out_stride_vec.emplace_back(stride[dim_pos_in[idx]]);
      }
    } else {
      for (size_t idx = 0; idx < size.size(); idx++) {
        out_size_vec.emplace_back(size[idx]);
        out_stride_vec.emplace_back(stride[idx]);
      }
    }
  } else {
    for (size_t idx = 0; idx < size.size(); idx++) {
      out_size_vec.emplace_back(size[idx]);
      out_stride_vec.emplace_back(stride[idx]);
    }
  }

  return std::make_tuple(out_size_vec, out_stride_vec);
}

/*************************************************************************
 * @brief Kernel implementation for As strided, used for tensor views
 * @param self - input which needs to be viewed
 ************************************************************************/
void AsStridedOperator::AllocateAndAddSynapseNode_Helper(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto is_output_persistent = GetOutputPersistence()[0];

  static_cast<void>(graph);
  static_cast<void>(is_output_persistent);
  TORCH_CHECK(
      inputs[1].isIntList(), "Input arg 1 needs to be of Int List type");
  TORCH_CHECK(
      inputs[2].isIntList(), "Input arg 2 needs to be of Int List type");
  TORCH_CHECK(inputs[3].isScalar(), "Input arg 3 fneeds to be of scalar type");
  auto size = inputs[1].toIntVector();
  auto strides = inputs[2].toIntVector();
  auto offset = inputs[3].toInt();
  auto opt_offset = c10::make_optional(offset);
  at::Tensor output;

  output = at::as_strided(self, size, strides, opt_offset);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section_with_size(
          p_context_->syn_inputs_[0],
          graph,
          size,
          strides,
          offset * self.itemsize()));
  p_context_->pt_outputs_.emplace_back(output);
}

/*************************************************************************
 * @brief Kernel implementation for As strided, used for tensor views
 * @param self - input which needs to be viewed
 ************************************************************************/
void AsStridedLayoutOperator::AllocateAndAddSynapseNode_Helper(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto is_output_persistent = GetOutputPersistence()[0];
  static_cast<void>(graph);
  static_cast<void>(is_output_persistent);
  TORCH_CHECK(
      inputs[1].isIntList(), "Input arg 1 needs to be of Int List type");

  at::Tensor output;
  int64_t offset = 0;
  c10::optional<int64_t> opt_offset = c10::make_optional((int64_t)0);
  auto sizes = self.sizes().vec();
  auto dims = inputs[1].toIntVector();
  auto is_5d_layout = dims.size() == 5 ? true : false;
  std::vector<int64_t> swapped_sizes = {
      sizes[dims[0]], sizes[dims[1]], sizes[dims[2]], sizes[dims[3]]};
  if (is_5d_layout) {
    swapped_sizes.push_back(sizes[dims[4]]);
  }

  std::vector<long int> new_strides = {
      swapped_sizes[1] * swapped_sizes[2] * swapped_sizes[3],
      swapped_sizes[3] * swapped_sizes[2],
      swapped_sizes[3],
      1};

  if (is_5d_layout) {
    new_strides.clear();
    new_strides.push_back(
        swapped_sizes[4] * swapped_sizes[3] * swapped_sizes[2] *
        swapped_sizes[1]);
    new_strides.push_back(
        swapped_sizes[4] * swapped_sizes[3] * swapped_sizes[2]);
    new_strides.push_back(swapped_sizes[4] * swapped_sizes[3]);
    new_strides.push_back(swapped_sizes[4]);
    new_strides.push_back(1);
  }

  output = at::as_strided(self, swapped_sizes, new_strides, opt_offset);
  output.unsafeGetTensorImpl()->set_sizes_contiguous(swapped_sizes);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section_with_size(
          p_context_->syn_inputs_[0],
          graph,
          swapped_sizes,
          new_strides,
          offset * self.itemsize()));
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

static inline Device ensure_has_index(c10::optional<at::Device> device) {
  const c10::impl::DeviceGuardImplInterface* impl =
      c10::impl::getDeviceGuardImpl((*device).type());
  return impl->getDevice();
}

bool is_pinned_hpu(const Tensor& self, c10::optional<at::Device> device) {
  ensure_has_index(device);
  return habana::PinnedMemoryAllocator_is_pinned(self.storage().data());
}

Tensor pin_memory_hpu(
    const at::Tensor& self,
    c10::optional<at::Device> device) {
  ensure_has_index(device);
  auto* allocator = habana::getPinnedMemoryAllocator();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  auto tensor = at::cpu::empty({0}, self.options())
                    .set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

void StridedInsertOperator::compute_params(
    synStridedOpParams& params,
    Stack& inputs,
    synapse_helpers::graph& graph) {
  auto orig_t = inputs[0].toTensor();
  auto insert_t = inputs[1].toTensor();
  std::vector<int64_t> strides;
  int64_t offset = 0;

  bool have_shape_tensors = inputs[2].isTensor();
  if (have_shape_tensors) {
    TORCH_CHECK(p_context_->syn_inputs_[2].ref().is_shape_tensor());
    TORCH_CHECK(p_context_->syn_inputs_[3].ref().is_shape_tensor());
    strides = p_context_->syn_inputs_[2].ref().pt_shape();
    offset = p_context_->syn_inputs_[3].ref().pt_shape()[0];
  } else {
    strides = inputs[2].toIntVector();
    offset = inputs[3].toInt();
  }

  if (!have_shape_tensors) {
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      AllocateSynapseShapeTensor(graph, orig_t);
      // For Dynamic case fill strides/offset params with max size
      if (!graph.is_dry_run()) {
        synapse_helpers::tensor& stride_tensor = p_context_->syn_inputs_[2];
        std::vector<int64_t> min, max;
        std::tie(min, max) =
            habana::ShapeInference::GetMinMaxShape(stride_tensor.id());
        strides = max;
        synapse_helpers::tensor& offset_tensor = p_context_->syn_inputs_[3];
        std::tie(min, max) =
            habana::ShapeInference::GetMinMaxShape(offset_tensor.id());
        if (max.size()) {
          offset = max[0];
        }
      }
    }
  }

  params.baseOffset = static_cast<uint64_t>(offset);

  size_t idx = 0;
  // synapse expects strides in reverse order
  for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
    params.strides[idx++] = static_cast<uint64_t>(*it);
  }
}

void StridedInsertOperator::AllocateAndAddSynapseNode_Helper(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect number of arguments for strided insert op");

  synStridedOpParams params;
  compute_params(params, inputs, graph);

  auto orig_t = inputs[0].toTensor();
  auto is_output_persistent = GetOutputPersistence()[0];
  auto output = habana_helpers::createPTTensor(
      orig_t,
      orig_t.sizes(),
      orig_t.options(),
      orig_t.suggest_memory_format(),
      is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  bool have_shape_tensors = inputs[2].isTensor();
  if (have_shape_tensors) {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

void StridedInsertOperator::ReuseMemoryAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect number of arguments for strided insert op");
  auto graph_input = inputs[4].toTensor();
  auto orig_t = inputs[0].toTensor();
  TORCH_CHECK(graph_input.sizes() == orig_t.sizes(), "incorrect graph input");

  struct synStridedOpParams params;
  compute_params(params, inputs, graph);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(syn_t_vec[0], graph));
  p_context_->pt_outputs_.emplace_back(graph_input);

  bool have_shape_tensors = inputs[2].isTensor();
  if (have_shape_tensors) {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  } else {
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

bool StridedViewOperator::verifiyViewMemoryAccess(
    at::Tensor& real,
    at::Tensor& view,
    at::Tensor& strides,
    at::Tensor& offset) {
  auto rv = real.sizes().vec();
  const uint64_t realTensorElements =
      std::accumulate(rv.begin(), rv.end(), 1, std::multiplies<unsigned>());
  if (realTensorElements == 0) {
    return true;
  }
  uint64_t lastElementOffset = 0;
  for (unsigned d = 0; d < view.dim(); d++) {
    if (view.sizes()[d] == 0) {
      return true;
    }
    lastElementOffset += strides.sizes()[d] * (view.sizes()[d] - 1);
  }
  if (offset.sizes()[0] + lastElementOffset >= realTensorElements) {
    return false;
  }
  return true;
}

/*************************************************************************
 * @brief Kernel implementation for strided view , used for tensor views
 * @param self - input which needs to be viewed
 ************************************************************************/
void StridedViewOperator::AllocateAndAddSynapseNode_Helper(
    synapse_helpers::graph& graph,
    Stack& inputs) {
  auto self = inputs[0].toTensor();
  std::vector<int64_t> size;
  std::vector<int64_t> strides;
  int64_t offset = 0;

  bool have_shape_tensors = inputs[1].isTensor();
  if (have_shape_tensors) {
    TORCH_CHECK(p_context_->syn_inputs_[1].ref().is_shape_tensor());
    TORCH_CHECK(p_context_->syn_inputs_[2].ref().is_shape_tensor());
    TORCH_CHECK(p_context_->syn_inputs_[3].ref().is_shape_tensor());

    size = p_context_->syn_inputs_[1].ref().pt_shape();
    strides = p_context_->syn_inputs_[2].ref().pt_shape();
    offset = p_context_->syn_inputs_[3].ref().pt_shape()[0];
    // For dynamic min-max inference, validate the mem access of
    // elements. If the calculation dosen't match, fail here for inference
    // fallback to kick in. if GC compile fails, the fallback penalty is huge.
    bool memAccessCheck = verifiyViewMemoryAccess(
        inputs[0].toTensor(),
        inputs[1].toTensor(),
        inputs[2].toTensor(),
        inputs[3].toTensor());
    TORCH_CHECK(
        self.numel() == 0 || memAccessCheck,
        "Strided View will access memory outside of original tensor range!");
  } else {
    TORCH_CHECK(
        inputs[1].isIntList(), "Input arg 1 needs to be of Int List type");
    TORCH_CHECK(
        inputs[2].isIntList(), "Input arg 2 needs to be of Int List type");
    TORCH_CHECK(inputs[3].isScalar(), "Input arg 3 needs to be of scalar type");
    size = inputs[1].toIntVector();
    strides = inputs[2].toIntVector();
    offset = inputs[3].toInt();
  }

  auto is_output_persistent = GetOutputPersistence()[0];
  auto output = habana_helpers::createPTTensor(
      self,
      size,
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);

  // For Dynamic case fill strides/offset params with max size
  if (graph.is_dynamic_graph()) {
    synapse_helpers::tensor& stride_tensor = p_context_->syn_inputs_[2];
    std::vector<int64_t> min, max;
    std::tie(min, max) =
        habana::ShapeInference::GetMinMaxShape(stride_tensor.id());
    strides = max;
    synapse_helpers::tensor& offset_tensor = p_context_->syn_inputs_[3];
    std::tie(min, max) =
        habana::ShapeInference::GetMinMaxShape(offset_tensor.id());
    if (max.size()) {
      offset = max[0];
    }
  }

  // If shape tensors are not created at frontend we need to create
  // Shape tensor at backend and also pass the params. Otherwise no params are
  // required.
  if (!have_shape_tensors) {
    // Allocate Shape tensor
    if (graph.is_dynamic_graph()) {
      AllocateSynapseShapeTensor(graph, output);
    }
    struct synStridedOpParams params;
    params.baseOffset = static_cast<uint64_t>(offset);
    size_t idx = 0;
    // synapse expects strides in reverse order
    for (auto it = strides.rbegin(); it != strides.rend(); ++it) {
      params.strides[idx++] = static_cast<uint64_t>(*it);
    }
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  } else {
    AddNodeToSynapseGraph(graph, nullptr, 0);
  }
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add("hpu::habana_d2d_memcpy", KERNEL_FN_GLOBAL(MemCopyOperator))
        .add("hpu::habana_d2d_memcpy_other", KERNEL_FN_GLOBAL(MemCopyOperator))
        .add("hpu::cast", KERNEL_FN_GLOBAL(CastLazyOperator))
        .add("aten::to.dtype", KERNEL_FN_GLOBAL(ToDtypeOperator))
        .add("hpu::control_edge_", KERNEL_FN_GLOBAL(DummyOperator))
        .add("hpu::control_edge_other_", KERNEL_FN_GLOBAL(DummyOperator))
        .add("hpu::as_strided_lazy_", KERNEL_FN_GLOBAL(AsStridedOperator))
        .add("hpu::as_strided_lazy_cl_", KERNEL_FN_GLOBAL(AsStridedClOperator))
        .add("hpu::strided_view", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_cl", KERNEL_FN_GLOBAL(StridedViewClOperator))
        .add("hpu::strided_view_ds", KERNEL_FN_GLOBAL(StridedViewOperator))
        .add("hpu::strided_view_cl_ds", KERNEL_FN_GLOBAL(StridedViewClOperator))
        .add("hpu::strided_insert", KERNEL_FN_GLOBAL(StridedInsertOperator))
        .add("hpu::strided_insert_ds", KERNEL_FN_GLOBAL(StridedInsertOperator))
        .add(
            "hpu::strided_insert_cl",
            KERNEL_FN_GLOBAL(StridedInsertClOperator))
        .add(
            "hpu::strided_insert_cl_ds",
            KERNEL_FN_GLOBAL(StridedInsertClOperator))
        .add(
            "hpu::as_strided_layout",
            KERNEL_FN_GLOBAL(AsStridedLayoutOperator));
