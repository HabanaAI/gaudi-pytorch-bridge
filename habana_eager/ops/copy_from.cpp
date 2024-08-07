/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "habana_eager/ops/copy_from.h"
#include "backend/backend_meta.h"
#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/eager_pipeline.h"
#include "backend/helpers/tensor_utils.h"
#include "common/utils.h"
#include "habana_eager/eager_context.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_eager/ops/view.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "hpu_ops/op_logger.h"
#include "pytorch_helpers/habana_helpers/thread_pool/thread_pool.h"

namespace {

std::vector<int64_t> translateSynapsePermuteToPt(
    const std::vector<uint8_t>& synapse_permuate) {
  // first reverse vector and then change idx to mirror
  // NHWC 2013 -> 3102 -> 0231
  // RSCK 3201 -> 1023 -> 2310
  std::vector<int64_t> pt_permute(
      synapse_permuate.rbegin(), synapse_permuate.rend());
  for (size_t i = 0; i < synapse_permuate.size(); i++) {
    pt_permute[i] = pt_permute.size() - pt_permute[i] - 1;
  }
  return pt_permute;
}

std::vector<int64_t> calcNewStrides(
    const torch::Tensor& permutedTensor,
    const std::vector<int64_t>& pt_permute) {
  // compute strides based on new sizes after permute
  // permtue the strides
  std::vector<int64_t> new_sizes, new_strides;
  std::tie(new_sizes, new_strides) =
      PermuteOperator::compute_output_shape(permutedTensor, pt_permute);

  std::vector<int64_t> new_strides_perm(new_strides.size());
  for (size_t i = 0; i < new_strides.size(); i++) {
    new_strides_perm[pt_permute[i]] = new_strides[i];
  }
  return new_strides_perm;
}

// Backend it treating Long(int64) as Int(int32) and Double as Float.
// In Lazy, we track this under the hood, and implicitly up/down cast data.
// In Eager, we cannot track it, so we 'unpack' tensors received from HPU
// with simple reinterpret_cast trick.
template <class From, class To>
void unpackData(const at::Tensor& t) {
  static_assert(sizeof(From) <= sizeof(To));
  From* from_ptr = reinterpret_cast<From*>(t.data_ptr());
  To* to_ptr = reinterpret_cast<To*>(t.data_ptr());
  for (int idx = t.numel() - 1; idx >= 0; --idx) {
    to_ptr[idx] = from_ptr[idx];
  }
}
} // namespace

namespace habana {
namespace eager {

at::Tensor _copy_from_and_resize(
    const at::Tensor& self,
    const at::Tensor& dst) {
  auto sizes = self.sizes().vec();
  if (self.sizes() != dst.sizes()) {
    dst.resize_(self.sizes());
  }
  return dst.copy_(self);
}

bool is_view_op_needed(const at::Tensor& t) {
  return (habana::is_view_lowering(t) || (!t.is_contiguous()));
}

void _assert_tensors_sizes(const at::Tensor& src, const at::Tensor& dst) {
  HABANA_ASSERT(
      dst.nbytes() >= src.nbytes(),
      "dst size: ",
      dst.nbytes(),
      " src size: ",
      src.nbytes());
}

void _assert_tensors_dtypes(const at::Tensor& src, const at::Tensor& dst) {
  HABANA_ASSERT(
      dst.dtype() == src.dtype(),
      "dst dtype: ",
      dst.dtype(),
      " src dtype: ",
      src.dtype());
}

at::Tensor _hpu_cast(const at::Tensor& dst, const at::Tensor& src) {
  habana::eager::EagerOp<at::Tensor&> hpu_op{
      "hpu::_copy_from", {src, dst}, {dst.sizes().vec()}, 1};
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::InplaceOut, "hpu::_copy_from", 1});
  return hpu_op.call(const_cast<at::Tensor&>(dst));
}

at::Tensor _hpu_cast(
    const at::Tensor& src,
    const at::TensorOptions& options,
    const c10::MemoryFormat& mem_format) {
  auto dst = at::empty(src.sizes(), options, mem_format);
  return _hpu_cast(dst, src);
}

at::Tensor add_strided_insert(at::Tensor dst, at::Tensor insert) {
  auto strides = dst.strides().vec();
  auto offset = dst.unsafeGetTensorImpl()->storage_offset();
  auto base = habana::eager::create_base(dst);
  base.unsafeGetTensorImpl()->set_storage_keep_dtype(dst.storage());

  habana::eager::EagerOp<at::Tensor&> hpu_op{
      "hpu::strided_insert_", {base, insert, strides, offset}};

  hpu_op.set_eager_op_info(
      {eager::eagerOpKind::Inplace,
       "hpu::strided_insert",
       decltype(eager::EagerOpMetaData::out_indices_){0}});

  hpu_op.call(base);
  return dst;
}

at::Tensor _copy_from_d2h(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  _assert_tensors_sizes(self, dst);
  _assert_tensors_dtypes(self, dst);

  auto self_ = self;
  auto self_strides = self_.strides();

  // If synapse permutation is applicable
  // calculate permuted strides w.r.t hpu tensor
  habana::eager::JoinPendingPipelineThreads();
  synapse_helpers::layouts::MemoryPermutation permutation;
  std::tie(permutation, std::ignore) =
      habana_helpers::get_tensor_memory_permutation(self_);
  if (permutation.size() != 0) {
    // translate synapse permtue to pt permute
    auto pt_permute = translateSynapsePermuteToPt(permutation);
    // if view tensor and not grad view tensor, then get the base tensor
    auto tmeta{habana::get_tensor_extra_meta(self_)};
    auto t = (tmeta->is_view_tensor() && !tmeta->is_maybe_grad_view())
        ? habana::eager::create_base(self_)
        : self_;
    // calculate new strides according to the synapse permutation
    auto strides_vec = calcNewStrides(t, pt_permute);
    self_strides = strides_vec;

    if (tmeta->is_maybe_grad_view()) {
      self_.unsafeGetTensorImpl()->set_sizes_and_strides(
          self_.sizes(), strides_vec);
    }
  }

  bool same_mem_format = (self_strides == dst.strides());
  if (!same_mem_format) {
    auto d2h_dst = at::empty_like(dst, dst.options().device(self.device()));
    self_ = add_strided_insert(d2h_dst, self_);
    habana::eager::JoinPendingPipelineThreads();
  }

  // To Do - join pending not required here once copy d2h
  // also comes through pipeline SW-126657
  // we also need Thread join before invoking is_view_op_needed()
  if (dst.is_contiguous() && is_view_op_needed(self_)) {
    auto base = habana::eager::create_base(self_);
    constexpr int out_index = 0;
    habana::eager::EagerOp<at::Tensor> hpu_op{
        "aten::as_strided",
        {base, self_.sizes(), self_.strides(), self_.storage_offset()},
        {self_.sizes().vec()},
        out_index};
    self_ = hpu_op.call();
    habana::eager::JoinPendingPipelineThreads();

    // restride cpu tensor since synapse will always return contiguous tensor
    dst.unsafeGetTensorImpl()->set_sizes_contiguous(dst.sizes());
    dst.unsafeGetTensorImpl()->set_storage_offset(0);
  }
  if (!common::IsInt64Supported() &&
      self_.scalar_type() == c10::ScalarType::Long) {
    habana_helpers::copy_data_to_host(self_, dst, false);
    unpackData<int32_t, int64_t>(dst);
  } else if (self_.scalar_type() == c10::ScalarType::Double) {
    habana_helpers::copy_data_to_host(self_, dst, false);
    unpackData<float, double>(dst);
  } else {
    habana_helpers::copy_data_to_host(self_, dst, non_blocking);
  }
  habana_helpers::print_tensor_debug(self_);
  habana_helpers::print_tensor_debug(dst);
  return dst;
}

void Execute_Copy(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking,
    c10::hpu::HPUStream stream,
    void* host_ptr = nullptr) {
  habana_helpers::copy_data_to_device(
      std::move(src), std::move(dst), non_blocking, stream, host_ptr);
}

void Copy_Compile_Empty_Task(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking,
    c10::hpu::HPUStream stream,
    void* host_ptr) {
  habana_helpers::Singleton_ExecThreadPool::getInstance().Enqueue(
      Execute_Copy,
      std::move(src),
      std::move(dst),
      std::move(non_blocking),
      std::move(stream),
      std::move(host_ptr));

  if (not GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
    habana_helpers::Singleton_ExecThreadPool::getInstance().JoinPendingThread();
  }
}

static void clear_permutation_info(const at::Tensor& tensor) {
  auto smeta{get_storage_extra_meta(tensor)};
  if (smeta) {
    auto synapse_permute = smeta->get_memory_permutation();
    if (synapse_permute.size() != 0) {
      PT_LAYOUTS_DEBUG("clearing memory permute ", VecToString(synapse_permute))
      smeta->set_memory_permutation({});
    }
  }
}

void Copy_Empty_Lowering_Task(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking,
    c10::hpu::HPUStream stream,
    void* host_ptr) {
  // Note: Here we clear the permutation info in lowering thread when the
  // pipeline is enabled, to avoid race condition.
  // Because src tensor is always not permuted and we will directly copy src
  // data to dst. So if dst is permuted, we need to clear the permutation info
  // in dst.
  clear_permutation_info(dst);
  habana_helpers::Singleton_CompileThreadPool::getInstance().Enqueue(
      Copy_Compile_Empty_Task,
      std::move(src),
      std::move(dst),
      non_blocking,
      std::move(stream),
      std::move(host_ptr));
  if (not GET_ENV_FLAG_NEW(PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE)) {
    habana_helpers::Singleton_CompileThreadPool::getInstance()
        .JoinPendingThread();
  }
}

void Register_Copy_In_Pipeline(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking,
    c10::hpu::HPUStream stream) {
  // Set pipeline metadata on the dst hpu tensor
  auto dst_hb_tmeta{habana::get_tensor_extra_meta(dst)};
  dst_hb_tmeta->set_tensor_pipelined();

  void* host_ptr;
  // Set cpu host memory metadata on the src cpu tensor (if non-pinned memory)
  if (!habana::PinnedMemoryAllocator_is_pinned(src.data_ptr())) {
    // Allocate host memory and do std::copy in the main thread
    // host memory will be freed after dma memcopy at copy_data_to_device
    const size_t total_bytes = habana_helpers::GetNBytes(src);
    synStatus status =
        habana::HPURegistrar::get_device().get_host_memory().malloc(
            &host_ptr, total_bytes);
    TORCH_CHECK(
        status == synStatus::synSuccess,
        Logger::formatStatusMsg(status),
        "Host malloc failed !");
    PT_EAGER_DEBUG("Host memory : ", host_ptr, " bytes :", total_bytes);

    std::copy(
        reinterpret_cast<uint8_t*>(src.data_ptr()),
        reinterpret_cast<uint8_t*>(src.data_ptr()) + total_bytes,
        reinterpret_cast<uint8_t*>(host_ptr));
  }

  habana::eager::SingleTonEagerContext::getInstance()
      .ScheduleWorkAndUpdateLoweringThreadHandle(
          Copy_Empty_Lowering_Task,
          std::move(src),
          std::move(dst),
          non_blocking,
          std::move(stream),
          std::move(host_ptr));
}

void Pipeline_Or_Direct_Copy(
    const at::Tensor& src,
    const at::Tensor& dst,
    bool non_blocking) {
  bool pipeline_flag =
      GET_ENV_FLAG_NEW(PT_HPU_EAGER_PIPELINE_ENABLE) && non_blocking;
  if (pipeline_flag) {
    // Check if the CPU src tensor is allocated at the pinned memory.
    // non-blocking copy with pinned memory allocation should not be pipelined.
    // as there can be a race condition with CPU tensor inplace operation.
    pipeline_flag &= (!habana::PinnedMemoryAllocator_is_pinned(src.data_ptr()));
  }

  if (pipeline_flag) {
    auto src_backend = HbEagerTensorPool::get_backend_tensor(src);
    auto dst_backend = HbEagerTensorPool::get_backend_tensor(dst);

    Register_Copy_In_Pipeline(
        src_backend,
        dst_backend,
        non_blocking,
        c10::hpu::getCurrentHPUStream());
  } else {
    habana::eager::JoinPendingPipelineThreads();
    // Because src tensor is always not permuted and we will directly copy src
    // data to dst. So if dst is permuted, we need to clear the permutation info
    // in dst.
    clear_permutation_info(dst);
    Execute_Copy(src, dst, non_blocking, c10::hpu::getCurrentHPUStream());
  }
}

at::Tensor _copy_from_h2d(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  _assert_tensors_sizes(self, dst);
  _assert_tensors_dtypes(self, dst);

  at::Tensor result;

  // Special handling for Long/Double tensors
  // Downcast sent data (implicitly backend will treat it as Int/Float anyway)
  auto temp_self = self;
  if (!common::IsInt64Supported() &&
      self.scalar_type() == c10::ScalarType::Long) {
    temp_self = self.to(c10::ScalarType::Int);
  } else if (self.scalar_type() == c10::ScalarType::Double) {
    temp_self = self.to(c10::ScalarType::Float);
  }

  // TODO optimize user configured CH last cpu tensor
  temp_self = temp_self.contiguous();
  if (dst.is_contiguous(dst.suggest_memory_format())) {
    dst.unsafeGetTensorImpl()->set_sizes_contiguous(dst.sizes());
  }

  if (!dst.is_contiguous()) {
    // This is done in two steps. First copy the contiguous Host tensor to
    // device. Then invoke strided_insert
    auto insert_t = at::empty(
        temp_self.sizes(), dst.options(), c10::MemoryFormat::Contiguous);
    Pipeline_Or_Direct_Copy(temp_self, insert_t, non_blocking);

    result = add_strided_insert(dst, insert_t);
  } else {
    Pipeline_Or_Direct_Copy(temp_self, dst, non_blocking);
    result = dst;
  }

  return result;
}

at::Tensor _copy_from_d2d(const at::Tensor& self, const at::Tensor& dst) {
  at::Tensor result;
  bool is_zst_view = false;
  if (dst.sizes() == 0) {
    auto is_tensor_pipelined = false;
    auto tmeta{habana::get_tensor_extra_meta(dst)};
    if (tmeta) {
      if (auto hb_tmeta = dynamic_cast<habana::TensorExtraMeta*>(tmeta)) {
        is_tensor_pipelined = hb_tmeta->is_tensor_pipelined();
      }
    }
    if (is_tensor_pipelined) {
      habana::eager::JoinPendingPipelineThreads();
    }
    is_zst_view = habana::is_view_lowering(dst);
  }

  if (!dst.is_contiguous() || is_zst_view) {
    auto self_ = self;
    bool same_data_type = (dst.scalar_type() == self.scalar_type());
    // If dtype is same, post_process_eager_graph() will take care
    // of strided-view node insertion when source is non-contiguous.
    // But, if dtype is different, additional cast operation is also
    // needed. As there is no explicit hpu::cast kind of eager-op, we
    // use hpu::_copy_from, which takes care of cast in the back-end.
    if (!same_data_type) {
      self_ = _hpu_cast(self, dst.options(), c10::MemoryFormat::Contiguous);
    }

    auto dstShape = dst.sizes();
    if (dstShape.vec() != self_.sizes().vec())
      self_ = self_.broadcast_to(dstShape);
    habana::eager::EagerOp<at::Tensor&> hpu_op{
        "hpu::_copy_from", {self_, dst}, {dst.sizes().vec()}, 1};
    hpu_op.set_eager_op_info(
        {habana::eager::eagerOpKind::InplaceOut,
         "hpu::_copy_from_strided_insert",
         decltype(eager::EagerOpMetaData::out_indices_){0}});
    result = hpu_op.call(const_cast<at::Tensor&>(self_));
  } else {
    // Since _copy_from is neither inplace nor an out variant but pytorch
    // expects to copy to dst, we treat _copy_from as an out variant in the
    // backend with "dst" as the out tensor
    habana::eager::EagerOp<at::Tensor&> hpu_op{
        "hpu::_copy_from", {self, dst}, {dst.sizes().vec()}, 1};
    hpu_op.set_eager_op_info(
        {habana::eager::eagerOpKind::InplaceOut, "hpu::_copy_from", 1});
    result = hpu_op.call(const_cast<at::Tensor&>(dst));
  }
  return result;
}

at::Tensor _copy_from(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking) {
  const auto src_device = self.device().type();
  const auto dst_device = dst.device().type();

  PT_EAGER_DEBUG(
      "_copy_from: src_sizes: ",
      self.sizes(),
      " src_strides: ",
      self.strides(),
      " src_dtype: ",
      self.scalar_type(),
      " src_device: ",
      src_device,
      " dst_sizes: ",
      dst.sizes(),
      " dst_strides: ",
      dst.strides(),
      " dst_dtype: ",
      dst.scalar_type(),
      " dst_device: ",
      dst_device);

  at::Tensor result;
  auto src = self;
  bool same_data_type = (dst.scalar_type() == self.scalar_type());

  // Special handling for Long/Double tensors
  // Unpack received data (implicitly received as Int/Float half of buffer)
  if (dst_device == at::kCPU) {
    if (!same_data_type) {
      src = _hpu_cast(
          self,
          dst.options().device(src.device()),
          c10::MemoryFormat::Contiguous);
    }
    result = _copy_from_d2h(src, dst, non_blocking);
  } else if (src_device == at::kCPU) {
    if (!same_data_type) {
      auto tmp = at::empty_like(src, src.options().device(dst.device()));
      tmp = _copy_from_h2d(src, tmp, non_blocking);
      result = _hpu_cast(dst, tmp);
    } else {
      result = _copy_from_h2d(src, dst, non_blocking);
    }
  } else {
    result = _copy_from_d2d(src, dst);
  }

  return result;
}

TORCH_LIBRARY_FRAGMENT(hpu, m) {
  m.def("_copy_from(Tensor self, Tensor dst) -> Tensor");
  m.def("identity(Tensor self) -> Tensor");
  m.def(
      "strided_insert(Tensor self, Tensor other, int[] stride, int offset) -> (Tensor)");
  m.def(
      "strided_insert_(Tensor(a!) self, Tensor other, int[] stride, int offset) -> (Tensor(a!))");
}
} // namespace eager
} // namespace habana
