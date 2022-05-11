/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
Tensor add_strided_insert_node(
    const Tensor& orig_t,
    const Tensor& insert_t,
    IntArrayRef strides,
    int64_t offset,
    bool is_flush) {
  auto mf = orig_t.suggest_memory_format();
  ir::NodePtr node;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES)) {
    std::string node_str = ((mf == c10::MemoryFormat::ChannelsLast) ||
                            (mf == c10::MemoryFormat::ChannelsLast3d))
        ? "hpu::strided_insert_cl_ds"
        : "hpu::strided_insert_ds";

    auto out_stride_st = empty_hpu_lazy(
        strides,
        orig_t.options(),
        c10::MemoryFormat::Contiguous,
        false,
        SHAPE_TENSOR);

    std::vector<int64_t> offset_vec = {offset};
    IntArrayRef offset_ref(offset_vec.data(), offset_vec.size());
    auto offset_st = empty_hpu_lazy(
        offset_ref,
        orig_t.options(),
        c10::MemoryFormat::Contiguous,
        false,
        SHAPE_TENSOR);

    node = std::make_shared<ir::StridedInsert>(
        orig_t, insert_t, out_stride_st, offset_st, node_str);
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
  auto hl_result = GetHbLazyTensor(result);

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());

  if (is_flush) {
    flush_op(result);
  }
  return result;
}

Tensor HbLazyTensorViews::get_base_tensor(const Tensor& self) {
  // Handle multi level views by traversing up to reach the base tensor (i.e.
  // until there is no entry in view table)
  auto out = self;
  auto id = GetHbLazyTensor(self).getTensorUniqueId();

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  // handle multi level views
  while (context->view_table.find(id) != context->view_table.end()) {
    out = context->view_table[id].base;
    id = GetHbLazyTensor(out).getTensorUniqueId();
  }

  return out;
}

const Tensor& HbLazyTensorViews::get_recent_base_tensor(const Tensor& self) {
  /* Fetch the most recent version of base from orig tensor map*/
  auto id = GetHbLazyTensor(self).getTensorUniqueId();

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto it = context->orig_tensor_map.find(id);
  if (it != context->orig_tensor_map.end()) {
    return it->second;
  }

  return self;
}

bool HbLazyTensorViews::HandleViews(const Tensor& t, const HbLazyTensor& hl_t) {
  PT_LAZY_TRACE;
  bool is_view = false;
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto id = hl_t.getTensorUniqueId();
  auto it = context->view_table.find(id);
  if (it != context->view_table.end()) {
    StrideParams& params = it->second;

    // pick the most recent version
    // use base if it is as_strided op else use the parent
    auto parent_or_base =
        (params.optype == kStridedOpDefault) ? params.base : params.parent;
    auto recent_orig_t = get_recent_base_tensor(parent_or_base);

    Tensor out;
    auto t_opt = c10::make_optional(t);
    bool add_asstrided_node = true;
    if (GET_ENV_FLAG_NEW(PT_HPU_DONT_USE_STRIDED_VIEW)) {
      add_asstrided_node = false;
      switch (params.optype) {
        case kStridedOpView:
          out = add_view_lazy(recent_orig_t, params.sizes, t_opt);
          break;
        case kStridedOpSlice:
          if (GET_ENV_FLAG_NEW(PT_HPU_USE_STRIDED_VIEW_FOR_SLICE)) {
            add_asstrided_node = true;
          } else {
            add_slice_lazy(recent_orig_t, params.params.slice_param, t_opt);
          }
          break;
        case kStridedOpTranspose:
          add_transpose_lazy(
              recent_orig_t, params.params.transpose_param, t_opt);
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
        case kStridedOpExpand:
          add_expand_lazy(
              recent_orig_t,
              params.sizes,
              params.params.expand_param.implicit,
              t_opt);
          break;
        case kStridedOpDefault:
          add_asstrided_node = true;
          break;
        default:
          HABANA_ASSERT(0);
      }
    }
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
    const TensorList& in_list) {
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

Tensor HbLazyTensorViews::add_strided_view_node(
    const Tensor& self,
    IntArrayRef size_in,
    IntArrayRef stride_in,
    int64_t storage_offset,
    bool is_update_view,
    c10::optional<Tensor> out_t) {
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
  auto exec_mode = habana_lazy_executor.getExecutionMode();
  if (exec_mode == kLOWERING) {
    auto result = empty_as_strided_lazy(self, size, stride, storage_offset);
    if (is_0d_tensor) {
      result.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
    }
    return result;
  }

  auto self_ = get_base_tensor(self);

  Tensor result;
  if (out_t.has_value()) {
    // actual op building phase
    // use the actual out tensor provided by the inplace op
    result = out_t.value();
  } else {
    result = empty_strided_hpu_lazy(
        size,
        stride,
        self_.options(),
        false,
        DATA_TENSOR,
        storage_offset,
        self);
  }

  StrideParams params;
  params.base = self_;
  params.parent = self;
  params.offset = storage_offset;
  params.optype = kStridedOpDefault;

  std::vector<int64_t> out_size_vec, out_stride_vec;
  std::tie(out_size_vec, out_stride_vec) =
      AsStridedOperator::compute_output_shape(self, size, stride);
  IntArrayRef out_size(out_size_vec.data(), out_size_vec.size());
  IntArrayRef out_stride(out_stride_vec.data(), out_stride_vec.size());

  params.sizes = out_size_vec;
  params.strides = out_stride_vec;

  if (is_update_view) {
    updateViewTable(result, params);
  } else {
    auto hb_result = GetHbLazyTensor(result);
    ir::Value& out = hb_result.CurrentIrValue();
    ir::NodePtr node =
        create_as_strided_node(params.base, size, stride, storage_offset);
    out.SetNode(
        node,
        hb_result.GetDevice(),
        hb_result.GetSizes(),
        hb_result.dtype_optional());
  }

  if (is_0d_tensor) {
    result.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});
  }

  return result;
}

Tensor HbLazyTensorViews::HandleViewsD2H(const Tensor& src) {
  PT_LAZY_TRACE;
  auto out = src;
  auto hl_t = GetHbLazyTensor(src);

  auto is_view = HandleViews(src, hl_t);

  if (is_view) {
    hl_t = GetHbLazyTensor(src);
    std::vector<HbLazyTensor> tensors = {hl_t};
    HbLazyTensor::SyncTensorsGraph(&tensors);
  } else {
    // check for updated version
    out = get_recent_base_tensor(src);
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

    auto is_view = HandleViews(t, hl_t);
    auto t_updated = t;
    if (is_view) {
      hl_t = GetHbLazyTensor(t);
      std::vector<HbLazyTensor> tensors = {hl_t};
      // TODO SW-74972 Need to add duplicate removal functionality within
      // syncTensorsGraph before moving it outside the for loop
      HbLazyTensor::SyncTensorsGraph(&tensors);

      // the storage offset of view output is always 0 as per the definition of
      // strided_view kernel set it to 0 before initiating collectives. Refer
      // test case in test_hpu_views_distributed.py
      t_updated.unsafeGetTensorImpl()->set_storage_offset(0);

      // note: this strided insert will be executed lazily after the execution
      // of collectives
      strided_insert_hpu_lazy(t, t, false);
    } else {
      // check for updated version
      t_updated = get_recent_base_tensor(t);
    }

    out_vec.emplace_back(t_updated);
  }
  return out_vec;
}

bool HbLazyTensorViews::HandleViewsD2D(
    const at::Tensor& src,
    const at::Tensor& dst) {
  PT_LAZY_TRACE;
  bool is_view = false;

  auto hb_src = GetHbLazyTensor(src);
  HandleViews(src, hb_src);

  // support for lhs sliced insert ex: a[::] = b. PT lowers this op as
  // as_strided + d2d copy. we replace both these ops by strided insert. This
  // way strided tensor support for D2D copy is avoided
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto hlresult = GetHbLazyTensor(dst);
  auto id = hlresult.getTensorUniqueId();
  auto it = context->view_table.find(id);

  if (it != context->view_table.end()) {
    is_view = true;

    StrideParams* params_ptr = &it->second;

    // get the base tensor
    // check for most recent version of the original tensor
    auto orig_t = get_base_tensor(params_ptr->base);
    auto orig_t_id = GetHbLazyTensor(orig_t).getTensorUniqueId();

    auto src_parent = get_base_tensor(src);
    auto src_parent_id = GetHbLazyTensor(src_parent).getTensorUniqueId();

    // the id check avoids a cycle with strided insert node
    // scenario t1_h[i - 1] += 1. Here the output of the add can be used
    // directly instead of performing one more strided insert
    if (src_parent_id != orig_t_id) {
      auto recent_orig_t = get_recent_base_tensor(orig_t);
      auto out = add_strided_insert_node(
          recent_orig_t, src, params_ptr->strides, params_ptr->offset);

      // update orig tensor map
      context->orig_tensor_map[orig_t_id] = out;
    }
  }

  return is_view;
}

void HbLazyTensorViews::updateViewTable(
    at::Tensor& result,
    StrideParams& params) {
  PT_LAZY_TRACE;
  auto hl_view_t = GetHbLazyTensor(result);
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto id = hl_view_t.getTensorUniqueId();
  PT_VIEWTABLE_DEBUG(
      "updateViewTable tensor id - ",
      id,
      " sizes ",
      params.sizes,
      " strides ",
      params.strides,
      " offset ",
      params.offset);
  auto storage = params.parent.storage();
  result.unsafeGetTensorImpl()->set_storage_keep_dtype(storage);

  context->view_table[id] = params;
}

StrideParams& HbLazyTensorViews::getViewTableParams(HbLazyTensor& hl_view_t) {
  PT_LAZY_TRACE;
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto id = hl_view_t.getTensorUniqueId();
  auto it = context->view_table.find(id);
  TORCH_CHECK(
      it != context->view_table.end(),
      "incorrect tensor id for view table access ",
      id);
  return it->second;
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
  ir::Value& out = hb_result.CurrentIrValue();
  ir::NodePtr node = std::make_shared<ir::View>(self, inferred_size);
  out.SetNode(
      node,
      hb_result.GetDevice(),
      hb_result.GetSizes(),
      hb_result.dtype_optional());
  return result;
}

Tensor HbLazyTensorViews::add_slice_lazy(
    const Tensor& self,
    const StridedOpSliceParams& params,
    c10::optional<Tensor> out_t) {
  PT_LAZY_TRACE;
  int64_t dim = params.dim;
  int64_t step = params.step;
  int64_t start_val = params.start.has_value() ? params.start.value() : 0;
  int64_t end_val = params.end.has_value() ? params.end.value() : INT64_MAX;

  auto node = std::make_shared<ir::Slice>(self, dim, start_val, end_val, step);
  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);

  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  return result;
}

Tensor HbLazyTensorViews::add_transpose_lazy(
    const Tensor& self,
    const StridedOpTransposeParams& params,
    c10::optional<Tensor> out_t) {
  PT_LAZY_TRACE;
  int64_t dim0_ = params.dim0_;
  int64_t dim1_ = params.dim1_;

  ir::NodePtr node = std::make_shared<ir::Transpose>(self, dim0_, dim1_);
  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
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
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
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
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  updateDstDependencies(hl_result, result);
  return result;
}

Tensor HbLazyTensorViews::add_squeeze_unsqueeze_lazy(
    const Tensor& self,
    const int64_t dim,
    c10::optional<Tensor> out_t,
    std::string node_str) {
  PT_LAZY_TRACE;
  auto hl_self = GetHbLazyTensor(self);

  ir::NodePtr node = std::make_shared<ir::SqueezeBase>(self, dim, node_str);
  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
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
  auto context = habana_lazy_executor.getDeviceExecutionContext(0);
  auto hl_weight = GetHbLazyTensor(weight);
  auto id = hl_weight.getTensorUniqueId();
  auto it = context->view_table.find(id);

  if (it == context->view_table.end()) {
    ir::Value& out5 = hl_weight.CurrentIrValue();
    out5.SetNode(
        node,
        hl_weight.GetDevice(),
        hl_weight.GetSizes(),
        hl_weight.dtype_optional(),
        out_index++);
  } else {
    auto wt_updated = empty_hpu_lazy(
        weight.sizes(),
        weight.options(),
        weight.suggest_memory_format(),
        false);
    auto hl_wt_updated = GetHbLazyTensor(wt_updated);
    ir::Value& out5 = hl_wt_updated.CurrentIrValue();
    out5.SetNode(
        node,
        hl_wt_updated.GetDevice(),
        hl_wt_updated.GetSizes(),
        hl_wt_updated.dtype_optional(),
        out_index++);

    // add strided insert node. Do not flush in lazy eager as it is a fused
    // op. step marker will be used at the end
    strided_insert_hpu_lazy(weight, wt_updated, /*is_flush*/ false);
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
  auto hl_params_shape = GetOrCreateHbLazyTensor(expand_shape, c10::kHPU);
  auto hl_false = GetIrValueForScalar(implicit);

  ir::NodePtr node = ir::Node::Create(
      Symbol::fromQualString("hpu::expand"),
      {hl_self.GetIrValue(), hl_params_shape.GetIrValue(), hl_false});

  HABANA_ASSERT(out_t.has_value());
  Tensor result = out_t.value();
  auto hl_result = GetHbLazyTensor(result);
  ir::Value& out = hl_result.CurrentIrValue();
  out.SetNode(
      node,
      hl_result.GetDevice(),
      hl_result.GetSizes(),
      hl_result.dtype_optional());
  input_pt_vec.emplace_back(self);
  input_pt_vec.emplace_back(expand_shape);
  node->AddInputPtTensors(input_pt_vec);
  flush_op(result);
  return result;
}

} // namespace habana_lazy
