/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "habana_lazy/view_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/lazy_executor.h"
#include "habana_lazy/ops/index.h"
#include "habana_lazy/ops/shape_ops.h"
#include "habana_lazy/ops/tensor_shape.h"
#include "habana_lazy/sbs_debug.h"

using namespace habana;
using namespace at;

namespace habana_lazy {

void StridedViewContext::ReplaceViewBase(int64_t id, Tensor& new_base_t) {
  HbContext* devctx = habana_lazy::HbContextArena::Get()->GetHbContext(
      GetHbLazyTensor(new_base_t).getDataPtr()->device);
  for (auto& uid_wptr : devctx->tensors_data) {
    std::shared_ptr<Data> data = uid_wptr.second.lock();
    auto& params_opt = data->stride_params;
    if (params_opt.has_value()) {
      auto& params = params_opt.value();
      auto recent_base = HbLazyTensorViews::get_recent_base_tensor(params.base);
      auto base_id = GetHbLazyTensorId(recent_base);
      if (id == base_id) {
        params.optype = kStridedOpDefault;
        params.base = new_base_t;
      }
    }
  }
}

bool IsStridesRatioZero(
    std::vector<int64_t>& self_strides,
    std::vector<int64_t>& stride_sizes) {
  auto len = self_strides.size();
  for (uint64_t i = 0; i < len; i++) {
    if (stride_sizes[i] < self_strides[i]) {
      return true;
    }
  }
  return false;
}

at::Tensor add_slice_insert_node(
    const at::Tensor& orig_t,
    const at::Tensor& insert_t,
    const std::vector<StridedOpSliceParams>& params) {
  std::vector<int64_t> paramsvec;
  std::for_each(
      params.begin(), params.end(), [&](const StridedOpSliceParams& n) {
        auto start = n.start;
        auto end = n.end;
        paramsvec.insert(paramsvec.end(), {n.dim, start, end, n.step});
      });
  auto node = std::make_shared<ir::SliceInsert>(orig_t, insert_t, paramsvec);
  auto result = empty_hpu_lazy(
      orig_t.sizes(), orig_t.options(), orig_t.suggest_memory_format(), false);
  habana::get_and_set_tensor_const(orig_t, result);
  auto hl_result = GetHbLazyTensor(result);
  hl_result.IrSetNode(node);
  flush_op(1);
  return result;
}

std::vector<uint64_t> get_strided_insert_stride_data(
    std::vector<int64_t>& stride,
    int64_t& offset) {
  std::vector<uint64_t> stride_data_vec;

  stride_data_vec.push_back(static_cast<uint64_t>(stride.size()));
  stride_data_vec.push_back(static_cast<uint64_t>(offset));
  for (auto it = stride.rbegin(); it != stride.rend(); ++it) {
    stride_data_vec.push_back(static_cast<uint64_t>(*it));
  }
  size_t fill_dim = (SYN_MAX_TENSOR_DIM + 1) - stride.size();
  for (size_t i = 0; i < fill_dim; i++) {
    stride_data_vec.push_back(static_cast<uint64_t>(0));
  }

  return stride_data_vec;
}

ir::NodePtr strided_insert_h2d(
    const Tensor& orig_t,
    const Tensor& insert_t,
    Tensor& offset_st,
    IntArrayRef strides,
    int64_t& offset,
    std::string& node_str) {
  ir::NodePtr node;
  auto self_strides = orig_t.strides().vec();
  auto stride_sizes = strides.vec();

  std::vector<uint64_t> stride_data_vec =
      get_strided_insert_stride_data(stride_sizes, offset);

  auto stride_st = empty_hpu_lazy(
      stride_data_vec.size() * 2,
      orig_t.options(),
      orig_t.suggest_memory_format(),
      false,
      HOST_TO_DEVICE_TENSOR);
  auto hl_stride_st = GetOrCreateHbLazyTensor(stride_st, c10::kHPU);
  auto hl_stride_internal = hl_stride_st.CurrentTensorAttached().value();
  auto tmeta{get_tensor_extra_meta(hl_stride_internal)};

  tmeta->set_host_data(
      stride_data_vec.data(),
      stride_data_vec.size(),
      sizeof(uint64_t),
      HostDataType::UINT64_T);

  if (orig_t.sizes().size() != strides.size() ||
      IsStridesRatioZero(self_strides, stride_sizes)) {
    tmeta->set_H2D_data_for_bucketing();
    node_str = "hpu::strided_insert_orig_ds_h2d";
    node = std::make_shared<ir::StridedInsert>(
        orig_t, insert_t, stride_st, node_str);
  } else {
    auto lazy_ten = GetHbLazyTensor(offset_st);
    auto tensor_offset = lazy_ten.CurrentTensorAttached().value();
    auto impl_offset = habana_lazy::GetHbInternalTensorImpl(tensor_offset);
    HABANA_ASSERT(impl_offset, "impl_offset is invalid");

    // Mark this front end shape tensor as it does not need synapse tensor.
    // It carries stride_ratios info for BE lowering kernel.
    impl_offset->setH2DFrontEndShapeTensor();

    std::vector<int64_t> stride_ratios;
    auto len = stride_sizes.size();
    for (uint64_t i = 0; i < len; i++) {
      stride_ratios.push_back(stride_sizes[i] / self_strides[i]);
    }
    impl_offset->get_shape_struct().set_strides_tensor_shape(strides.vec());
    impl_offset->get_shape_struct().set_stride_ratio(stride_ratios);
    node_str = "hpu::strided_insert_orig_ds";
    node = std::make_shared<ir::StridedInsert>(
        orig_t, insert_t, stride_st, offset_st, node_str);
  }

  return node;
}

Tensor add_strided_insert_node(
    const Tensor& orig_t,
    const Tensor& insert_t,
    IntArrayRef strides,
    int64_t offset,
    bool is_flush) {
  auto mf = orig_t.suggest_memory_format();
  ir::NodePtr node;
  PT_DYNAMIC_SHAPE_DEBUG(
      "Strided insert orig size = ",
      orig_t.sizes().vec(),
      " insert_t sizes = ",
      insert_t.sizes().vec(),
      " strides = ",
      strides.vec(),
      " offset = ",
      offset);
  if (habana_helpers::GetRefineDynamicShapeStatus()) {
    std::string node_str = ((mf == c10::MemoryFormat::ChannelsLast) ||
                            (mf == c10::MemoryFormat::ChannelsLast3d))
        ? "hpu::strided_insert_cl_ds"
        : "hpu::strided_insert_ds";

    std::vector<int64_t> offset_vec = {offset};
    IntArrayRef offset_ref(offset_vec.data(), offset_vec.size());
    auto offset_st = empty_hpu_lazy(
        offset_ref,
        orig_t.options(),
        c10::MemoryFormat::Contiguous,
        false,
        SHAPE_TENSOR);
    auto self_strides = orig_t.strides().vec();
    auto stride_sizes = strides.vec();
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_AS_STRIDED)) {
      node = strided_insert_h2d(
          orig_t, insert_t, offset_st, strides, offset, node_str);
    } else {
      if (orig_t.sizes().size() != strides.size() ||
          IsStridesRatioZero(self_strides, stride_sizes)) {
        node_str = "hpu::strided_insert_orig_ds";
        auto out_stride_st = empty_hpu_lazy(
            strides,
            orig_t.options(),
            c10::MemoryFormat::Contiguous,
            false,
            SHAPE_TENSOR);
        node = std::make_shared<ir::StridedInsert>(
            orig_t, insert_t, out_stride_st, offset_st, node_str);
      } else {
        auto lazy_ten = GetHbLazyTensor(offset_st);
        auto tensor_offset_st = lazy_ten.CurrentTensorAttached().value();
        auto impl_st = habana_lazy::GetHbInternalTensorImpl(tensor_offset_st);
        HABANA_ASSERT(impl_st, "impl_st is invalid");

        std::vector<int64_t> stride_ratios;
        auto len = stride_sizes.size();
        for (uint64_t i = 0; i < len; i++) {
          stride_ratios.push_back(stride_sizes[i] / self_strides[i]);
        }
        impl_st->get_shape_struct().set_strides_tensor_shape(strides.vec());
        impl_st->get_shape_struct().set_stride_ratio(stride_ratios);
        node = std::make_shared<ir::StridedInsert>(
            orig_t, insert_t, offset_st, node_str);
      }
    }
  } else {
    std::string node_str = ((mf == c10::MemoryFormat::ChannelsLast) ||
                            (mf == c10::MemoryFormat::ChannelsLast3d))
        ? "hpu::strided_insert_cl"
        : "hpu::strided_insert";

    node = std::make_shared<ir::StridedInsert>(
        orig_t, insert_t, strides, offset, node_str);
  }
  auto result = empty_hpu_lazy(
      orig_t.sizes(), orig_t.options(), orig_t.suggest_memory_format(), false);
  habana::get_and_set_tensor_const(orig_t, result);
  auto hl_result = GetHbLazyTensor(result);
  hl_result.IrSetNode(node);

  if (is_flush) {
    flush_op(1);
  }
  return result;
}

Tensor HbLazyTensorViews::get_base_tensor(const Tensor& self) {
  // Handle multi level views by traversing up to reach the base tensor (i.e.
  // until there is no entry in view table)
  auto out = self;

  // handle multi level views
  auto hl_t = GetHbLazyTensor(self, true, false);
  while (hl_t.getDataPtr()->stride_params.has_value()) {
    out = hl_t.getDataPtr()->stride_params.value().base;
    hl_t = GetHbLazyTensor(out, true, false);
  }

  return out;
}

const Tensor HbLazyTensorViews::get_recent_base_tensor(const Tensor& self) {
  /* Fetch the most recent version of base*/
  auto& base_t = GetHbLazyTensor(self, true, false).getDataPtr()->recent_base;
  if (base_t.has_value()) {
    return base_t.value();
  }

  return self;
}

bool HbLazyTensorViews::HandleViews(const Tensor& t, const HbLazyTensor& hl_t) {
  PT_LAZY_TRACE;
  bool is_view = false;
  StrideParams params;
  if (hl_t.getDataPtr()->stride_params.has_value()) {
    auto& params = hl_t.getDataPtr()->stride_params.value();
    // pick the most recent version
    // use base if it is as_strided op else use the parent
    auto parent_or_base =
        (params.optype == kStridedOpDefault) ? params.base : params.parent;
    auto recent_orig_t = get_recent_base_tensor(parent_or_base);

    GetHbLazyTensor(recent_orig_t).SetOpAccumulationInProgress();

    Tensor out;
    auto t_opt = c10::make_optional(t);
    bool add_asstrided_node = true;
    add_asstrided_node = false;
    switch (params.optype) {
      case kStridedOpView:
      case kStridedOpViewDtype:
        out = add_view_lazy(recent_orig_t, params.sizes, t_opt);
        break;
      case kStridedOpSlice:
        add_slice_lazy(recent_orig_t, params.params.slice_param, t_opt);
        break;
      case kStridedOpTranspose:
        add_transpose_lazy(recent_orig_t, params.params.transpose_param, t_opt);
        break;
      case kStridedOpT:
        add_t_lazy(recent_orig_t, t_opt);
        break;
      case kStridedOpPermute:
        add_permute_lazy(recent_orig_t, params.sizes, t_opt);
        break;
      case kStridedOpSqueeze:
        add_squeeze_unsqueeze_lazy(
            recent_orig_t,
            params.params.squeeze_param.dim,
            t_opt,
            "aten::squeeze");
        break;
      case kStridedOpUnsqueeze:
        add_squeeze_unsqueeze_lazy(
            recent_orig_t,
            params.params.squeeze_param.dim,
            t_opt,
            "aten::unsqueeze");
        break;
      case kStridedOpSqueezeDims:
        add_squeeze_dims_lazy(recent_orig_t, params.sizes, t_opt);
        break;
      case kStridedOpExpand:
        add_expand_lazy(
            recent_orig_t,
            params.sizes,
            params.params.expand_param.implicit,
            t_opt);
        break;
      case kStridedOpIdentity:
        out = add_identity_lazy(recent_orig_t, t_opt);
        break;
      case kStridedOpDefault:
        add_asstrided_node = true;
        break;
      default:
        HABANA_ASSERT(0);
    }

    // add strided view node only if is not already evaluated
    if (params.viewStatus != kEvaluated) {
      if (add_asstrided_node) {
        out = add_strided_view_node(
            recent_orig_t,
            params.sizes,
            params.strides,
            params.offset,
            false,
            t_opt);
      }

      is_view = true;
      // Ops inside HandleViews function do not call flush_op separately.
      // Hence handling SBS debug counter here
      SBSDebug::getInstance().IncreaseOpsAndTensors(1);
    }
  }
  return is_view;
}

HbLazyTensor HbLazyTensorViews::HandleViewsOrUpdate(
    const at::Tensor& t,
    HbLazyTensor& hl_t) {
  auto hl_out = hl_t;
  auto is_view = HandleViews(t, hl_t);

  if (is_view == false) {
    auto t_updated = get_recent_base_tensor(t);
    hl_out = GetHbLazyTensor(t_updated);
  }
  return hl_out;
}

std::vector<Tensor> HbLazyTensorViews::HandleViewsTensorList(
    const TensorList in_list) {
  std::vector<Tensor> updated_t_list;

  for (auto t : in_list) {
    auto hl_t = GetHbLazyTensor(t);

    auto is_view = HandleViews(t, hl_t);

    if (is_view == false) {
      auto t_updated = get_recent_base_tensor(t);
      updated_t_list.push_back(t_updated);
    } else {
      updated_t_list.push_back(t);
    }
  }

  return updated_t_list;
}

void HbLazyTensorViews::add_strided_view_node_parallel_impl(
    const at::Tensor& self,
    at::IntArrayRef size_in,
    at::IntArrayRef stride_in,
    int64_t storage_offset,
    bool is_update_view,
    at::Tensor& out,
    bool is_out) {
  PT_LAZY_TRACE;
  auto self_ = get_base_tensor(self);
  StrideParams params;
  params.base = self_;
  params.parent = self;
  params.offset = storage_offset;
  params.optype = kStridedOpDefault;

  auto hb_result = GetHbLazyTensor(out);

  if (is_update_view) {
    std::tie(params.sizes, params.strides) =
        AsStridedOperator::compute_output_shape(size_in, stride_in);
    hb_result.getDataPtr()->stride_params = params;

    // book keeping to aid addition of strided view outputs for gradient views
    // of bucket
    if (params.base.dim() == 1) {
      HbContext* devctx = habana_lazy::HbContextArena::Get()->GetHbContext(
          hb_result.GetDevice());

      auto shared_ptr = hb_result.getDataPtr();

      if (shared_ptr) {
        std::lock_guard<std::recursive_mutex> lock(
            habana_lazy::HbContextArena::Get()->GetMutex());
        devctx->insert(hb_result.getTensorUniqueId(), shared_ptr);
      }
    }
  } else {
    hb_result.IrSetNode(create_as_strided_node(
        params.base, size_in, stride_in, storage_offset, is_out));
  }
}

Tensor HbLazyTensorViews::add_strided_view_node(
    const Tensor& self,
    IntArrayRef size_in,
    IntArrayRef stride_in,
    int64_t storage_offset,
    bool is_update_view,
    c10::optional<Tensor> out_t,
    bool is_out) {
  PT_LAZY_TRACE;
  IntArrayRef size = size_in;
  bool is_0d_tensor = false;
  std::vector<int64_t> initvec{1};
  if (size_in.size() == 0) {
    size = initvec;
    is_0d_tensor = true;
  }
  IntArrayRef stride = stride_in;
  if (stride_in.size() == 0) {
    stride = initvec;
  }

  // when we get a call from lowering, we create a storage based backend
  // tensor
  auto exec_mode = get_habana_lazy_executor().getExecutionMode();
  if (exec_mode == kLOWERING) {
    auto result = empty_as_strided_lazy(self, size, stride, storage_offset);
    if (is_0d_tensor) {
      SET_SIZE_STRIDE_0D(result);
    }
    return result;
  }

  Tensor result;
  if (out_t.has_value()) {
    // actual op building phase
    // use the actual out tensor provided by the inplace op
    result = out_t.value();
  } else {
    result = empty_strided_hpu_lazy(
        size, stride, self.options(), false, DATA_TENSOR, storage_offset, self);
  }

  if (habana_lazy::AccThread::Get().CanUseAccThread() &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_VIEW_OPS_MODE) != 0) {
    habana_lazy::AccThread::Get().run([self,
                                       storage_offset,
                                       is_update_view,
                                       result,
                                       size = size.vec(),
                                       stride = stride.vec(),
                                       is_out]() mutable {
      add_strided_view_node_parallel_impl(
          self, size, stride, storage_offset, is_update_view, result, is_out);
      habana_lazy::AccThread::Get().PushCleanupTask(
          [self = std::move(self),
           result = std::move(result),
           size = std::move(size),
           stride = std::move(stride)]() {
            /* Silence unused lambda capture */
            (void)self;
            (void)result;
            (void)size;
            (void)stride;
          });
    });
  } else {
    add_strided_view_node_parallel_impl(
        self, size, stride, storage_offset, is_update_view, result, is_out);
  }

  if (is_0d_tensor) {
    SET_SIZE_STRIDE_0D(result);
  }

  return result;
}

Tensor HbLazyTensorViews::process_strided_view(
    const Tensor& self,
    IntArrayRef size_in,
    IntArrayRef stride_in,
    int64_t storage_offset,
    bool create_storage) {
  PT_LAZY_TRACE;
  IntArrayRef size = size_in;
  bool is_0d_tensor = false;
  std::vector<int64_t> initvec{1};
  if (size_in.size() == 0) {
    size = initvec;
    is_0d_tensor = true;
  }
  IntArrayRef stride = stride_in;
  if (stride_in.size() == 0) {
    stride = initvec;
  }

  Tensor result;
  result = empty_strided_hpu_lazy(
      size,
      stride,
      self.options(),
      create_storage,
      DATA_TENSOR,
      storage_offset,
      self,
      true);

  if (is_0d_tensor) {
    SET_SIZE_STRIDE_0D(result);
  }

  return result;
}

Tensor HbLazyTensorViews::HandleViewsD2H(const Tensor& src) {
  PT_LAZY_TRACE;
  auto is_src_const = habana::is_tensor_const(src);
  auto src_const_id = habana::get_tensor_const_id(src);
  auto out = src;
  habana::set_tensor_const(out, is_src_const, src_const_id);
  auto hl_t = GetHbLazyTensor(src);

  bool is_view = hl_t.getDataPtr()->stride_params.has_value();

  if (is_view) {
    /* need to update tmap here for cases like b = add(a); mark_step(). c =
    slice(b). c.to('cpu')
    Here slice is a view op and Tmap needs to have tensor b*/
    RUNNING_HASH_COMBINE_TENSOR(src)

    bool reuse_base_storage = false;

    Tensor base;
    Tensor base_internal_tensor;
    if (src.is_contiguous() &&
        (GET_ENV_FLAG_NEW(PT_SBS) == SBSModes::SBS_MODE_DISABLED)) {
      base =
          get_recent_base_tensor(hl_t.getDataPtr()->stride_params.value().base);
      TORCH_CHECK(base.storage(), "base tensor should have valid storage");
      base_internal_tensor = GetHbLazyTensor(base).EvaluateTensorData();
      auto hb_impl = habana_lazy::GetHbInternalTensorImpl(base_internal_tensor);
      auto synapse_permute = hb_impl->GetMemoryPermutation();

      // optimization cannot be performed for permuted tensors
      if (synapse_permute.size() == 0) {
        reuse_base_storage = true;
      }
    }

    if (reuse_base_storage) {
      // set backend tensor data for src
      auto storage_impl = base.unsafeGetTensorImpl();

      // internal dtype can be different from src dtype. ex: long
      // will be represented as int

      auto at_internal_tensor = AtenInternalHbTensor(
          c10::Storage(storage_impl->storage()),
          c10::scalarTypeToTypeMeta(habana_helpers::getInternalDtype(
              base_internal_tensor.scalar_type())),
          c10::nullopt,
          src.sizes(),
          src.strides(),
          c10::MemoryFormat::Contiguous);
      at_internal_tensor.unsafeGetTensorImpl()->set_storage_offset(
          src.unsafeGetTensorImpl()->storage_offset());

      habana::set_tensor_const(at_internal_tensor, is_src_const, src_const_id);
      hl_t.SetTensorData(at_internal_tensor);
      hl_t.SetIsConstTensor(is_src_const, src_const_id);
    } else {
      HandleViews(src, hl_t);
      hl_t = GetHbLazyTensor(src);

      hl_t.SetIsConstTensor(is_src_const, src_const_id);
      std::vector<HbLazyTensor> tensors = {hl_t};
      HbLazyTensor::SyncTensorsGraph(&tensors);
    }
  } else {
    // check for updated version
    out = get_recent_base_tensor(src);
    habana::set_tensor_const(out, is_src_const, src_const_id);
  }

  return out;
}

/* treat collectives as inplace operations
    add strided insert node to update the parent tensor
    example:
    b = view(a)
    dist.allreduce(b)
    Here b will be updated by the collective
    c = strided_insert_node(a, b); -> this will ensure that subsequent view
    operations will fetch the updated values
*/
std::vector<at::Tensor> HbLazyTensorViews::UpdateViewDistributed(
    std::vector<at::Tensor>& in_vec) {
  PT_LAZY_TRACE;
  std::vector<at::Tensor> out_vec;
  std::vector<HbLazyTensor> hl_t_vec;

  for (auto t : in_vec) {
    // auto out = src;
    auto hl_t = GetHbLazyTensor(t);

    /* special handling for deep speed where all reduce happens on a view*/
    auto context = get_device_lazy_execution_context();
    auto t_updated = t;

    if (hl_t.getDataPtr()->stride_params.has_value()) {
      auto base =
          get_recent_base_tensor(hl_t.getDataPtr()->stride_params.value().base);

      if (t_updated.is_contiguous()) {
        // optimization for contiguous views
        TORCH_CHECK(
            base.storage().data_ptr(),
            "base tensor is expected to be have storage");
        auto storage = base.storage();

        // PT doesnt allow set_storage to be invoked on detached tensors
        // example: all_reduce(a.view().detach())
        // create a new tensor similar to view tensor t and then set its storage
        t_updated = empty_hpu_lazy(
            t.sizes(), t.options(), t.suggest_memory_format(), false);
        t_updated.unsafeGetTensorImpl()->set_storage_keep_dtype(storage);
        t_updated.unsafeGetTensorImpl()->set_storage_offset(
            t.unsafeGetTensorImpl()->storage_offset());
      } else {
        HandleViews(t, hl_t);

        hl_t = GetHbLazyTensor(t);
        std::vector<HbLazyTensor> tensors = {hl_t};
        // TODO SW-74972 Need to add duplicate removal functionality within
        // syncTensorsGraph before moving it outside the for loop
        HbLazyTensor::SyncTensorsGraph(&tensors);

        // the storage offset of view output is always 0 as per the definition
        // of strided_view kernel set it to 0 before initiating collectives.
        // Refer test case in test_hpu_views_distributed.py
        t_updated.unsafeGetTensorImpl()->set_storage_offset(0);

        // note: this strided insert will be executed lazily after the execution
        // of collectives
        strided_insert_hpu_lazy(t, t, false);
        context->viewContext.isLazyViewPresent = true;
      }
    } else {
      // check for updated version
      t_updated = get_recent_base_tensor(t);
    }

    // Note: storage() api call also sets the front end storage()
    TORCH_CHECK(
        (t_updated.numel() == 0 ||
         (t_updated.storage().data_ptr() && (t_updated.data_ptr() != nullptr))),
        "t_updated tensor is expected to be have storage and valid data_ptr");
    out_vec.emplace_back(t_updated);
  }
  return out_vec;
}

/* API to handle views for all inplace lazy collective kernels. Inplace Lazy
 * collectives on views in general will require a stepmarker prior to execution
 * of the collective. This is to ensure that the strided insert op invoked
 * subsequent to the collective will pick up the updated tensor.
 * TODO this can be optimized when the tensor is a contiguous view on a base.*/
void HbLazyTensorViews::HandleViewsLazyCollective(const at::Tensor& tensor) {
  PT_LAZY_TRACE;

  auto data_ptr = GetHbLazyTensor(tensor).getDataPtr();
  if ((data_ptr->stride_params.has_value()) &&
      (data_ptr->stride_params.value().viewStatus != kEvaluated)) {
    PT_LAZY_DEBUG(
        "stepmarker triggered to handle lazy inplace before collectives");
    PT_IRGRAPH_DEBUG(
        "step marker due to handle lazy inplace before collectives");
    HbLazyTensor::StepMarker({}, nullptr, {}, true);
  }
}

std::vector<StridedOpSliceParams> HbLazyTensorViews::getSliceInsertParams(
    const at::Tensor& recent_orig_t,
    const at::Tensor& recent_src_t,
    const StrideParams& params) {
  // Incase of slice operator on multi axes, it comes as different
  // slice operation on differnt axes, we combine them into single slice
  // operation.
  if (!GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SLICE_INSERT) ||
      (recent_orig_t.sizes().size() != recent_src_t.sizes().size())) {
    return std::vector<StridedOpSliceParams>();
  }
  std::vector<StridedOpSliceParams> back_to_back_slices{};
  c10::optional<StrideParams> params_link_opt = params;
  std::unordered_set<int64_t> dims;
  while (params_link_opt.value().optype == kStridedOpSlice) {
    back_to_back_slices.push_back(params_link_opt.value().params.slice_param);

    // If multiple times same dim exists, use strided insert.
    if (dims.find(params_link_opt.value().params.slice_param.dim) !=
        dims.end()) {
      return std::vector<StridedOpSliceParams>();
    }
    dims.insert(params_link_opt.value().params.slice_param.dim);
    params_link_opt = GetHbLazyTensor(params_link_opt.value().parent)
                          .getDataPtr()
                          ->stride_params;
    if (!params_link_opt.has_value()) {
      break;
    }
    params_link_opt = params_link_opt.value();
  }
  return (params_link_opt.has_value()) ? std::vector<StridedOpSliceParams>()
                                       : back_to_back_slices;
}

bool HbLazyTensorViews::HandleViewsD2D(
    const at::Tensor& src_,
    const at::Tensor& dst) {
  PT_LAZY_TRACE;

  auto hb_src = GetHbLazyTensor(src_);
  HandleViews(src_, hb_src);

  // support for lhs sliced insert ex: a[::] = b. PT lowers this op as
  // as_strided + d2d copy. we replace both these ops by strided insert. This
  // way strided tensor support for D2D copy is avoided
  auto& stride_params_opt = GetHbLazyTensor(dst).getDataPtr()->stride_params;
  if (!stride_params_opt.has_value())
    return false;
  auto& params = stride_params_opt.value();

  // get the base tensor
  // check for most recent version of the original tensor
  auto orig_t = get_base_tensor(params.base);
  auto orig_t_id = GetHbLazyTensorId(orig_t);

  auto src_parent = get_base_tensor(src_);
  auto src_parent_id = GetHbLazyTensorId(src_parent);

  auto src = src_;
  if ((src.numel() < c10::multiply_integers(params.sizes)) ||
      (static_cast<size_t>(src.dim()) < params.sizes.size())) {
    // broadcast needed
    auto t = empty_hpu_lazy(
        dst.sizes(),
        dst.options(),
        c10::nullopt,
        false /*storage*/,
        DATA_TENSOR);
    auto t_opt = c10::make_optional(t);
    src = HbLazyTensorViews::add_expand_lazy(
        src_, dst.sizes().vec(), false /*implicit*/, t_opt);
  }

  // the id check avoids a cycle with strided insert node
  // scenario t1_h[i - 1] += 1. Here the output of the add can be used
  // directly instead of performing one more strided insert
  if (src_parent_id != orig_t_id) {
    auto recent_orig_t = get_recent_base_tensor(orig_t);
    auto recent_src_t = get_recent_base_tensor(src);

    at::Tensor out;
    auto back_to_back_slices =
        getSliceInsertParams(recent_orig_t, recent_src_t, params);
    if (back_to_back_slices.empty()) {
      out = add_strided_insert_node(
          recent_orig_t, recent_src_t, params.strides, params.offset);
    } else {
      out = add_slice_insert_node(
          recent_orig_t, recent_src_t, back_to_back_slices);
    }
    // update orig tensor map
    GetHbLazyTensor(orig_t).getDataPtr()->recent_base = out;
    GetHbLazyTensor(recent_orig_t, true, false).SetOpAccumulationInProgress();
    GetHbLazyTensor(orig_t, false, false).SetOpAccumulationInProgress();
    GetHbLazyTensor(src_).SetOpAccumulationInProgress();
  }
  return true;
}

Tensor HbLazyTensorViews::add_view_lazy(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<Tensor> out_t) {
  PT_LAZY_TRACE;
  int64_t sum_elm = 1;
  for (auto& i : self.sizes()) {
    sum_elm *= i;
  }
  auto inferred_size =
      habana_helpers::infer_size(size, static_cast<int64_t>(sum_elm));

  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hb_result = GetHbLazyTensor(result);
  ir::NodePtr node = std::make_shared<ir::View>(self, inferred_size);
  hb_result.IrSetNode(node);
  return result;
}

Tensor HbLazyTensorViews::add_identity_lazy(
    const Tensor& self,
    c10::optional<Tensor> out_t) {
  PT_LAZY_TRACE;

  HABANA_ASSERT(out_t.has_value());
  HABANA_ASSERT(out_t.has_value());
  auto result = out_t.value();
  auto hb_result = GetHbLazyTensor(result);
  ir::NodePtr node = std::make_shared<ir::Identity>(self, "hpu::identity");
  hb_result.IrSetNode(node);
  return result;
}

Tensor HbLazyTensorViews::add_slice_lazy(
    const Tensor& self,
    const StridedOpSliceParams& params,
    c10::optional<Tensor> out_t) {
  PT_LAZY_TRACE;
  int64_t dim = params.dim;
  int64_t step = params.step;
  int64_t start_val = params.start;
  int64_t end_val = params.end;

  auto node = std::make_shared<ir::Slice>(self, dim, start_val, end_val, step);
  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);

  hl_result.IrSetNode(node);
  return result;
}

Tensor HbLazyTensorViews::add_transpose_lazy(
    const Tensor& self,
    const StridedOpTransposeParams& params,
    c10::optional<Tensor> out_t) {
  PT_LAZY_TRACE;
  int64_t dim0 = params.dim0;
  int64_t dim1 = params.dim1;

  ir::NodePtr node = std::make_shared<ir::Transpose>(self, dim0, dim1);
  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);
  hl_result.IrSetNode(node);
  return result;
}

Tensor HbLazyTensorViews::add_t_lazy(
    const Tensor& self,
    c10::optional<Tensor> out_t) {
  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
  hl_self = HandleViewsOrUpdate(self, hl_self);
  auto node = ir::Node::Create(
      Symbol::fromQualString("aten::t"), {hl_self.GetIrValue()});
  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);
  hl_result.IrSetNode(node);
  std::vector<at::Tensor> input_pt_vec{self};
  node->AddInputPtTensors(input_pt_vec);
  return result;
}

Tensor HbLazyTensorViews::add_permute_lazy(
    const Tensor& self,
    std::vector<int64_t> dims_vec,
    c10::optional<Tensor> out_t) {
  PT_LAZY_TRACE;
  for (unsigned i = 0; i < dims_vec.size(); i++) {
    dims_vec[i] =
        at::maybe_wrap_dim(dims_vec[i], self.dim(), /*wrap_scalar=*/true);
  }
  IntArrayRef dims_(dims_vec);
  ir::NodePtr node = std::make_shared<ir::Permute>(self, dims_);
  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);
  hl_result.IrSetNode(node);
  return result;
}

Tensor HbLazyTensorViews::add_squeeze_unsqueeze_lazy(
    const Tensor& self,
    const int64_t dim,
    c10::optional<Tensor> out_t,
    std::string node_str) {
  PT_LAZY_TRACE;
  auto hl_self = GetHbLazyTensor(self);

  ir::NodePtr node = nullptr;
  if (node_str == "aten::unsqueeze" && !self.dim()) {
    node = std::make_shared<ir::Identity>(self, "hpu::identity");
  } else {
    node = std::make_shared<ir::SqueezeBase>(self, dim, node_str);
  }

  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);
  hl_result.IrSetNode(node);
  return result;
}

Tensor HbLazyTensorViews::add_squeeze_dims_lazy(
    const Tensor& self,
    std::vector<int64_t> dims_vec,
    c10::optional<Tensor> out_t) {
  PT_LAZY_TRACE;
  IntArrayRef dims_(dims_vec);
  ir::NodePtr node = std::make_shared<ir::SqueezeDims>(self, dims_);

  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);
  hl_result.IrSetNode(node);
  return result;
}

// For inplace torch ops acting on views, lazyOp calls will automatically insert
// strided insert node. But we need to do this manually for custom kernels. This
// api is meant to be used in custom kernels for tensors
// that are updated inplace. Strided insert node will be additionally inserted
// if the input tensor is a view.
void HbLazyTensorViews::CustomKernelAddNodeInplace(
    const at::Tensor& weight,
    habana_lazy::ir::NodePtr node,
    int64_t& out_index) {
  auto hl_weight = GetHbLazyTensor(weight);
  auto& params_opt = hl_weight.getDataPtr()->stride_params;
  if ((params_opt.has_value()) &&
      (params_opt.value().viewStatus != kEvaluated)) {
    auto wt_updated = empty_hpu_lazy(
        weight.sizes(),
        weight.options(),
        weight.suggest_memory_format(),
        false);
    auto hl_wt_updated = GetHbLazyTensor(wt_updated);
    hl_wt_updated.IrSetNode(node, out_index++);

    // add strided insert node. Do not flush in lazy eager as it is a fused
    // op. step marker will be used at the end
    strided_insert_hpu_lazy(weight, wt_updated, /*is_flush*/ false);
  } else {
    hl_weight.IrSetNode(node, out_index++);
  }
}

Tensor HbLazyTensorViews::add_expand_lazy(
    const Tensor& self,
    std::vector<int64_t> sizes,
    bool implicit,
    c10::optional<Tensor> out_t) {
  PT_LAZY_TRACE;

  IntArrayRef size_in{sizes};
  auto size = size_in;
  std::vector<int64_t> initvec{1};
  size = (size_in.vec().size() == 0) ? initvec : size_in;

  std::vector<at::Tensor> input_pt_vec;
  std::vector<int64_t> expandedSizes;
  std::vector<int64_t> expandedStrides;
  std::tie(expandedSizes, expandedStrides) =
      at::inferExpandGeometry(self.sizes(), self.strides(), size);

  // expandedStrides will be set to 0 by inferExpandGeometry.
  // Since we give back a contiguous tensor, we will set strides
  // to proper values.
  habana_helpers::recalc_strides(expandedStrides, expandedSizes);

  auto expand_shape = empty_strided_hpu_lazy(
      expandedSizes, expandedStrides, self.options(), false, SHAPE_TENSOR);

  auto hl_self = GetOrCreateHbLazyTensor(self, c10::kHPU);
  hl_self = HandleViewsOrUpdate(self, hl_self);

  ir::NodePtr node;
  if (habana_helpers::GetRefineDynamicShapeStatus() == 1) {
    auto hl_params_shape = GetOrCreateHbLazyTensor(expand_shape, c10::kHPU);
    auto hl_false = GetIrValueForScalar(implicit);
    node = ir::Node::Create(
        Symbol::fromQualString("hpu::expand_ds"),
        {hl_self.GetIrValue(), hl_params_shape.GetIrValue(), hl_false});
  } else {
    node = std::make_shared<ir::Expand>(self, size, implicit);
  }

  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);
  hl_result.IrSetNode(node);
  input_pt_vec.emplace_back(self);
  input_pt_vec.emplace_back(expand_shape);
  node->AddInputPtTensors(input_pt_vec);
  flush_op(1);
  return result;
}

bool is_view_output(
    HbLazyTensor hl_t,
    bool is_allreduce,
    std::set<int64_t> bucket_recent_id,
    size_t& view_out_size) {
  bool is_out = false;

  if (is_allreduce) {
    auto& param_opt = hl_t.getDataPtr()->stride_params;

    if ((param_opt.has_value())) {
      auto& params = param_opt.value();
      if ((params.viewStatus == kViewWrite) && (params.write_cnt == 1)) {
        auto recent_orig_t =
            HbLazyTensorViews::get_recent_base_tensor(params.base);
        auto hl_recent_orig_t = GetHbLazyTensor(recent_orig_t);

        if (hl_recent_orig_t.CurrentIrValue().IsHpuInputNode()) {
          // not a view on strided_insert o/p
          is_out = false;
        } else if (
            (params.optype == kStridedOpDefault) && (params.base.dim() == 1)) {
          // additionaly check for contiguous strides
          auto recalc_stride = params.strides;
          habana_helpers::recalc_strides(recalc_stride, params.sizes);
          auto recent_base_id = hl_recent_orig_t.getTensorUniqueId();
          if ((bucket_recent_id.count(recent_base_id)) &&
              (recalc_stride == params.strides)) {
            is_out = true;
            view_out_size += c10::multiply_integers(params.sizes);
          }
        }
      }
    }
  }
  return is_out;
}

void add_strided_view_output_node(
    HbLazyTensor hl_t,
    int64_t id,
    StrideParams& params,
    HbExecutionContext* context) {
  context->viewContext.view_outputs.emplace(id);
  // pick the most recent version
  auto recent_orig_t = HbLazyTensorViews::get_recent_base_tensor(params.base);
  auto t = AtenFromHbLazyTensor(
      hl_t, c10::nullopt, params.sizes, c10::nullopt, c10::nullopt);
  auto t_opt = c10::make_optional(t);

  HbLazyTensorViews::add_strided_view_node(
      recent_orig_t,
      params.sizes,
      params.strides,
      params.offset,
      false /*is_update_view*/,
      t_opt,
      true
      /* is_out*/);
  params.viewStatus = kEvaluated;
}

void HbLazyTensorViews::HandleViewsLiveTensors(
    HbContext* devctx,
    bool is_allreduce,
    std::set<int64_t>& bucket_recent_id) {
  auto context = get_device_lazy_execution_context();
  size_t view_out_sizes = 0;
  size_t bucket_sizes = 0;

  // view outputs will be added only if the total grad view outputs match the
  // bucket size.
  std::vector<HbLazyTensor> maybe_view_outputs;

  for (auto& uid : devctx->tensors_data_opt_order) {
    std::shared_ptr<Data> data = devctx->getDataPtr(uid);
    if (data != nullptr) {
      auto hl_t = HbLazyTensor(std::move(data));

      if (is_allreduce) {
        if (bucket_recent_id.count(hl_t.getTensorUniqueId())) {
          bucket_sizes += c10::multiply_integers(hl_t.GetSizes());
        }
      }

      // exclude the views
      auto ir_value = hl_t.CurrentIrValue();
      auto& params_opt = hl_t.getDataPtr()->stride_params;
      auto is_view = params_opt.has_value();
      if (is_view) {
        auto is_view_out = is_view_output(
            hl_t, is_allreduce, bucket_recent_id, view_out_sizes);
        if (is_view_out) {
          // collect potential view out candidates
          maybe_view_outputs.emplace_back(hl_t);
        } else {
          context->viewContext.hb_tensors_exclude_out_view.emplace_back(hl_t);
          // clear the writecnt
          params_opt.value().write_cnt = 0;
        }
      }
      if (hl_t.getDataPtr()->recent_base.has_value()) {
        context->viewContext.hb_tensors_exclude_out_view.emplace_back(hl_t);
      }
    } // (data != nullptr)
  }

  bool is_view_out = false;
  if (bucket_sizes && bucket_sizes == view_out_sizes) {
    is_view_out = true;
  }

  for (auto hl_t : maybe_view_outputs) {
    auto id = hl_t.getTensorUniqueId();
    auto& params = hl_t.getDataPtr()->stride_params.value();

    if (is_view_out) {
      add_strided_view_output_node(hl_t, id, params, context);
    } else {
      context->viewContext.hb_tensors_exclude_out_view.emplace_back(hl_t);
    }

    // clear the writecnt
    params.write_cnt = 0;
  }

  if (!context->viewContext.view_outputs.size()) {
    bucket_recent_id.clear();
    PT_LAZY_DEBUG(
        "Strided view outputs not present. Clearing bucket_recent_id.");
  }
}

void HbLazyTensorViews::StepMarkerAllReduce(const std::vector<Tensor>& inputs) {
  habana_lazy::NoAccThread no_acc_thread;

  std::set<int64_t> bucket_recent_id;
  std::vector<habana_lazy::HbLazyTensor> bucket_hl_t;

  for (auto t : inputs) {
    auto base = habana_lazy::HbLazyTensorViews::get_base_tensor(t);
    auto org_id = habana_lazy::GetHbLazyTensorId(base, false, false);

    auto recent_base =
        habana_lazy::HbLazyTensorViews::get_recent_base_tensor(base);
    auto updated_id = habana_lazy::GetHbLazyTensorId(recent_base, true, false);

    if (org_id != updated_id) {
      /* strided inserts have happened */
      bucket_hl_t.emplace_back(
          habana_lazy::GetHbLazyTensor(base, false, false));
      bucket_recent_id.emplace(updated_id);
    }
  }

  /* special processing of view outputs needed only for the bwd case*/
  bool is_allreduce_bwd = (bucket_recent_id.size() > 0);
  PT_IRGRAPH_DEBUG("step marker due to HbLazyTensorViews::StepMarkerAllReduce");
  habana_lazy::HbLazyTensor::StepMarker(
      {},
      nullptr,
      {},
      false /*async*/,
      is_allreduce_bwd,
      bucket_hl_t,
      bucket_recent_id);
}

/* UpdateviewHash needs to be invoked for every input tensor of every op.
Written in unrolled form for optimization*/
size_t HbLazyTensorViews::updateViewHash(
    const habana_lazy::HbLazyTensor& hl_t,
    size_t hash) {
  auto& params_opt = hl_t.getDataPtr()->stride_params;
  if (params_opt.has_value()) {
    auto& params = params_opt.value();
    auto optype = params.optype;
    hash = at::hash_combine(hash, optype);

    switch (optype) {
      case kStridedOpView:
      case kStridedOpViewDtype:
      case kStridedOpPermute:
      case kStridedOpSqueezeDims:
        for (auto& s : params.sizes) {
          hash = at::hash_combine(hash, s);
        }
        break;
      case kStridedOpSlice: {
        auto& slice_params = params.params.slice_param;
        hash = at::hash_combine(hash, slice_params.dim);
        hash = at::hash_combine(hash, slice_params.start);
        hash = at::hash_combine(hash, slice_params.step);
        hash = at::hash_combine(hash, slice_params.end);
      } break;
      case kStridedOpTranspose: {
        auto& transpose_params = params.params.transpose_param;
        hash = at::hash_combine(hash, transpose_params.dim0);
        hash = at::hash_combine(hash, transpose_params.dim1);
      } break;
      case kStridedOpT:
      case kStridedOpIdentity:
        break;
      case kStridedOpSqueeze:
      case kStridedOpUnsqueeze: {
        auto& squeeze_params = params.params.squeeze_param;
        hash = at::hash_combine(hash, squeeze_params.dim);
      } break;
      case kStridedOpExpand: {
        auto& expand_param = params.params.expand_param;
        hash = at::hash_combine(hash, expand_param.implicit);
      } break;
      case kStridedOpDefault:
        for (auto& s : params.sizes) {
          hash = at::hash_combine(hash, s);
        }

        for (auto& s : params.strides) {
          hash = at::hash_combine(hash, s);
        }

        hash = at::hash_combine(hash, params.offset);
        hash = at::hash_combine(hash, params.viewStatus);
        break;
      default:
        TORCH_CHECK("incorrect optype for views ", optype);
    } // switch (optype)
  }

  return hash;
}

} // namespace habana_lazy
