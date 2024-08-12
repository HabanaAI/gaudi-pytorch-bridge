/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/eager/wrap_kernels_declarations.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "hpu_ops/cpu_fallback.h"
#include "pytorch_helpers/habana_helpers/frontend_utils.h"
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

using namespace at;
using namespace habana;

// *************************************************
// This file contains list of symbols needed to link new frontend plugin, but
// not relevant for eager execution. They will be removed once backend
// dependencies are cleared out as a part of SW-123330

// It also contains list of manual/override_fn ops included in
// wrap_kernel_register, that are mandatory for PT2.0, but not implemented for
// new frontend. They all will be moved to backend as a part of SW-118176
// *************************************************

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::batch_norm_stats(
    const at::Tensor& input,
    double eps) {
  FALLBACK_UNSUPPORTED_OP2(batch_norm_stats, PARAMS2(input, eps));
}

at::Tensor hpu_wrap::batch_norm_elemt(
    const at::Tensor& input,
#if IS_PYTORCH_AT_LEAST(2, 4)
    const ::std::optional<at::Tensor>& weight,
    const ::std::optional<at::Tensor>& bias,
#else
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
#endif
    const at::Tensor& mean,
    const at::Tensor& invstd,
    double eps) {
  FALLBACK_UNSUPPORTED_OP2(
      batch_norm_elemt, PARAMS2(input, weight, bias, mean, invstd, eps));
}

::std::tuple<at::Tensor, at::Tensor> hpu_wrap::
    batch_norm_gather_stats_with_counts(
        const at::Tensor& input,
        const at::Tensor& mean,
        const at::Tensor& invstd,
#if IS_PYTORCH_AT_LEAST(2, 4)
        const ::std::optional<at::Tensor>& running_mean,
        const ::std::optional<at::Tensor>& running_var,
#else
        const c10::optional<at::Tensor>& running_mean,
        const c10::optional<at::Tensor>& running_var,
#endif
        double momentum,
        double eps,
        const at::Tensor& counts) {
  FALLBACK_UNSUPPORTED_OP2(
      batch_norm_gather_stats_with_counts,
      PARAMS2(
          input,
          mean,
          invstd,
          running_mean,
          running_var,
          momentum,
          eps,
          counts));
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> hpu_wrap::
    batch_norm_backward_reduce(
        const at::Tensor& grad_out,
        const at::Tensor& input,
        const at::Tensor& mean,
        const at::Tensor& invstd,
#if IS_PYTORCH_AT_LEAST(2, 4)
        const ::std::optional<at::Tensor>& weight,
#else
        const c10::optional<at::Tensor>& weight,
#endif
        bool input_g,
        bool weight_g,
        bool bias_g) {
  FALLBACK_UNSUPPORTED_OP2(
      batch_norm_backward_reduce,
      PARAMS2(
          grad_out, input, mean, invstd, weight, input_g, weight_g, bias_g));
}

at::Tensor hpu_wrap::batch_norm_backward_elemt(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& mean,
    const at::Tensor& invstd,
#if IS_PYTORCH_AT_LEAST(2, 4)
    const ::std::optional<at::Tensor>& weight,
#else
    const c10::optional<at::Tensor>& weight,
#endif
    const at::Tensor& mean_dy,
    const at::Tensor& mean_dy_xmu,
    const at::Tensor& count) {
  FALLBACK_UNSUPPORTED_OP2(
      batch_norm_backward_elemt,
      PARAMS2(
          grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count));
}

Tensor& hpu_wrap::index_add_out(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& source,
    const at::Scalar& alpha,
    at::Tensor& out) {
  FALLBACK_UNSUPPORTED_OP2_O(
      index_add, PARAMS2(self, dim, index, source, alpha, out), out);
}

// *************************************************
// BELOW is list of symbols needed to link new frontend plugin but not relevant
// for eager execution. They will be removed once backend dependencies
// are cleared out as a part of SW-123330
// *************************************************
#define EAGER_NOT_SUPPORTED                                                   \
  HABANA_ASSERT(                                                              \
      false, "Frontend Op ", __func__, " not supported with new Eager mode"); \
  std::terminate();

namespace habana_lazy {

at::Tensor squeeze_hpu_lazy(const at::Tensor&, const int64_t) {
  EAGER_NOT_SUPPORTED;
}

std::vector<at::Tensor> split_with_sizes_hpu_lazy(
    const at::Tensor&,
    at::IntArrayRef,
    int64_t) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor nonzero_hpu_lazy(const at::Tensor&) {
  EAGER_NOT_SUPPORTED;
}

::std::tuple<at::Tensor, at::Tensor> _unique_hpu_lazy(
    const at::Tensor&,
    bool,
    bool) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor repeat_inlv_hpu_lazy(const at::Tensor&, c10::optional<int64_t>) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> unique2_hpu_lazy(
    const at::Tensor&,
    bool,
    bool,
    bool) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor append_to_batch_h2d_list(const at::Tensor&) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor get_tensor_for_scalar(double, const at::TensorOptions&) {
  EAGER_NOT_SUPPORTED;
}

void flush_op(
    size_t,
    std::shared_ptr<HbLazyFrontEndInfoToBackend>,
    std::vector<HbLazyTensor>) {
  EAGER_NOT_SUPPORTED;
}

void handle_collective(const at::IValue&) {
  EAGER_NOT_SUPPORTED;
}

Tensor empty_as_strided_lazy(
    const Tensor&,
    IntArrayRef,
    IntArrayRef,
    c10::optional<int64_t>) {
  EAGER_NOT_SUPPORTED;
}

ir::NodePtr create_as_strided_node(
    at::Tensor const&,
    c10::ArrayRef<long>,
    c10::ArrayRef<long>,
    c10::optional<long>,
    bool) {
  EAGER_NOT_SUPPORTED;
}

bool is_inplace(at::Symbol) {
  EAGER_NOT_SUPPORTED;
}

void strided_insert_hpu_lazy(const Tensor&, const Tensor&, bool) {
  EAGER_NOT_SUPPORTED;
}

Tensor empty_strided_hpu_lazy(
    IntArrayRef,
    IntArrayRef,
    const TensorOptions&,
    bool,
    synTensorType,
    int64_t,
    c10::optional<std::reference_wrapper<const at::Tensor>>,
    bool) {
  EAGER_NOT_SUPPORTED;
}

void InitSizesAndStrides(
    at::Tensor&,
    c10::optional<synTensorType>,
    c10::optional<IntArrayRef>,
    c10::optional<IntArrayRef>,
    c10::optional<MemoryFormat>) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& broadcast_hpu_lazy_(
    [[maybe_unused]] at::Tensor& tensor,
    [[maybe_unused]] int64_t root_rank,
    [[maybe_unused]] int64_t comm_id) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& allreduce_hpu_lazy_(
    [[maybe_unused]] at::Tensor& tensor,
    [[maybe_unused]] uint8_t reduce_op,
    [[maybe_unused]] int64_t comm_id) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& reduce_hpu_lazy_(
    [[maybe_unused]] at::Tensor& tensor,
    [[maybe_unused]] int64_t dst_rank,
    [[maybe_unused]] uint8_t reduce_op,
    [[maybe_unused]] int64_t comm_id) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& alltoall_hpu_lazy_out(
    [[maybe_unused]] const at::Tensor& input_tensor,
    [[maybe_unused]] int64_t comm_id,
    [[maybe_unused]] at::Tensor& output_tensor,
    [[maybe_unused]] std::vector<int64_t>& outputSplitSizes,
    [[maybe_unused]] std::vector<int64_t>& inputSplitSizes) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& allgather_hpu_lazy_out(
    [[maybe_unused]] const at::Tensor& inputTensor,
    [[maybe_unused]] int64_t comm_id,
    [[maybe_unused]] at::Tensor& output_tensor) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& reduce_scatter_hpu_lazy_out(
    [[maybe_unused]] const at::Tensor& input_tensor,
    [[maybe_unused]] uint8_t reduce_op,
    [[maybe_unused]] int64_t comm_id,
    [[maybe_unused]] at::Tensor& output_tensor) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& send_hpu_lazy_(
    [[maybe_unused]] at::Tensor& tensor,
    [[maybe_unused]] int64_t dst_rank,
    [[maybe_unused]] int64_t tag,
    [[maybe_unused]] int64_t comm_id) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor& recv_hpu_lazy_(
    [[maybe_unused]] at::Tensor& tensor,
    [[maybe_unused]] int64_t src_rank,
    [[maybe_unused]] int64_t tag,
    [[maybe_unused]] int64_t comm_id) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<at::Tensor&, at::Tensor&> cast_to_fp8_lazy(
    [[maybe_unused]] const at::Tensor& input,
    [[maybe_unused]] const c10::optional<at::Tensor>& scale,
    [[maybe_unused]] bool stochastic_rounding,
    [[maybe_unused]] at::Tensor& out,
    [[maybe_unused]] at::Tensor& amax) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor cast_from_fp8_lazy(
    [[maybe_unused]] const at::Tensor& input,
    [[maybe_unused]] const c10::optional<at::Tensor>& scale,
    [[maybe_unused]] at::ScalarType out_dtype) {
  EAGER_NOT_SUPPORTED;
}

} // namespace habana_lazy

void optimizer_adagrad_hpu_wrap(
    [[maybe_unused]] const TensorList& gradients,
    [[maybe_unused]] TensorList& weights,
    [[maybe_unused]] TensorList& variances,
    [[maybe_unused]] const at::Tensor& epoch_num,
    [[maybe_unused]] at::Tensor& lr,
    [[maybe_unused]] const float wd,
    [[maybe_unused]] const float lrd,
    [[maybe_unused]] const float epsilon) {
  EAGER_NOT_SUPPORTED;
}

void optimizer_lars_hpu_wrap(
    [[maybe_unused]] const at::TensorList params,
    [[maybe_unused]] at::TensorList grads,
    [[maybe_unused]] const std::vector<int64_t> skipMasks,
    [[maybe_unused]] const float eeta,
    [[maybe_unused]] const float weight_decay,
    [[maybe_unused]] const float eps,
    [[maybe_unused]] const float lr) {
  EAGER_NOT_SUPPORTED;
}

void optimizer_sgd_hpu_wrap(
    [[maybe_unused]] const TensorList& gradients,
    [[maybe_unused]] TensorList& weights,
    [[maybe_unused]] at::Tensor& lr,
    [[maybe_unused]] const float wd,
    [[maybe_unused]] const float mom,
    [[maybe_unused]] const float damp,
    [[maybe_unused]] const bool nesterov) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_wrap(
    [[maybe_unused]] const Tensor& gradients,
    [[maybe_unused]] Tensor& weights_in,
    [[maybe_unused]] Tensor& moments_in,
    [[maybe_unused]] const Tensor& indices,
    [[maybe_unused]] const Tensor& learning_rate,
    [[maybe_unused]] const Tensor& valid_count_tensor,
    [[maybe_unused]] float mom,
    [[maybe_unused]] bool nesterov) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_gelu_v2_wrap(
    [[maybe_unused]] const at::Tensor& input,
    [[maybe_unused]] const c10::optional<at::Tensor>& scale,
    [[maybe_unused]] bool stochastic_rounding,
    [[maybe_unused]] bool is_amax) {
  EAGER_NOT_SUPPORTED;
}

void optimizer_ema_hpu_wrap(
    [[maybe_unused]] const TensorList model_inputs,
    [[maybe_unused]] TensorList updated_ema,
    [[maybe_unused]] const at::Tensor& decay) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
    [[maybe_unused]] const Tensor& gradients,
    [[maybe_unused]] Tensor& weights_in,
    [[maybe_unused]] Tensor& moments_in,
    [[maybe_unused]] const Tensor& indices,
    [[maybe_unused]] const Tensor& learning_rate,
    [[maybe_unused]] const Tensor& valid_count_tensor) {
  EAGER_NOT_SUPPORTED;
}

Tensor torchvision_nms_hpu_wrap(
    [[maybe_unused]] const at::Tensor& boxes,
    [[maybe_unused]] const at::Tensor& scores,
    [[maybe_unused]] double iou_threshold) {
  EAGER_NOT_SUPPORTED;
}

Tensor batched_nms_hpu_wrap(
    [[maybe_unused]] const at::Tensor& boxes,
    [[maybe_unused]] const at::Tensor& scores,
    [[maybe_unused]] const at::Tensor& indices,
    [[maybe_unused]] float iou_threshold) {
  EAGER_NOT_SUPPORTED;
}

Tensor embedding_bag_sum_hpu_wrap(
    [[maybe_unused]] const Tensor& input,
    [[maybe_unused]] const Tensor& indices,
    [[maybe_unused]] const Tensor& offsets,
    [[maybe_unused]] const Tensor& valid_count,
    [[maybe_unused]] int64_t kernel_mode) {
  EAGER_NOT_SUPPORTED;
}

Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap(
    [[maybe_unused]] Tensor& out,
    [[maybe_unused]] const Tensor& input,
    [[maybe_unused]] const Tensor& indices,
    [[maybe_unused]] const Tensor& offsets,
    [[maybe_unused]] const Tensor& valid_count,
    [[maybe_unused]] int64_t kernel_mode) {
  EAGER_NOT_SUPPORTED;
}

Tensor habana_cast_to_fp8_wrap(
    [[maybe_unused]] const at::Tensor& input,
    [[maybe_unused]] bool stochastic_rounding,
    [[maybe_unused]] int seed) {
  EAGER_NOT_SUPPORTED;
}

std::vector<at::Tensor> habana_permute_1D_sparse_data_wrap(
    [[maybe_unused]] const at::Tensor& permute,
    [[maybe_unused]] const at::Tensor& lengths,
    [[maybe_unused]] const at::Tensor& indices,
    [[maybe_unused]] const c10::optional<at::Tensor>& weights) {
  EAGER_NOT_SUPPORTED;
}

std::vector<at::Tensor> habana_permute_2D_sparse_data_wrap(
    [[maybe_unused]] const at::Tensor& permute,
    [[maybe_unused]] const at::Tensor& lengths,
    [[maybe_unused]] const at::Tensor& indices,
    [[maybe_unused]] const c10::optional<at::Tensor>& weights) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor habana_split_permute_cat_wrap(
    [[maybe_unused]] const at::Tensor& input,
    [[maybe_unused]] const at::Tensor& indices,
    [[maybe_unused]] int64_t batch_size,
    [[maybe_unused]] int64_t num_features,
    [[maybe_unused]] int64_t dims) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor habana_expand_into_jagged_permute_wrap(
    [[maybe_unused]] const at::Tensor& permute,
    [[maybe_unused]] const at::Tensor& input_offsets,
    [[maybe_unused]] const at::Tensor& output_offsets,
    [[maybe_unused]] int64_t output_size) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
habana_bounds_check_indices_wrap(
    [[maybe_unused]] at::Tensor& indices,
    [[maybe_unused]] at::Tensor& offsets,
    [[maybe_unused]] at::Tensor& warning,
    [[maybe_unused]] const at::Tensor& rows_per_table,
    [[maybe_unused]] int64_t bounds_check_mode,
    [[maybe_unused]] const c10::optional<at::Tensor>& weights) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor matmul_ex_wrap(
    [[maybe_unused]] const at::Tensor& self,
    [[maybe_unused]] const at::Tensor& other,
    [[maybe_unused]] at::ScalarType dtype) {
  EAGER_NOT_SUPPORTED;
}
std::tuple<at::Tensor, at::Tensor> matmul_ex_backward_wrap(
    [[maybe_unused]] const Tensor& grad_output,
    [[maybe_unused]] const Tensor& self,
    [[maybe_unused]] const Tensor& other,
    [[maybe_unused]] at::ScalarType dtype) {
  EAGER_NOT_SUPPORTED;
}

Tensor habana_random_seed_wrap([[maybe_unused]] const at::Tensor& input) {
  EAGER_NOT_SUPPORTED;
}

namespace vision {
namespace ops {
at::Tensor roi_align_fwd_wrap(
    [[maybe_unused]] const at::Tensor& images,
    [[maybe_unused]] const at::Tensor& rois,
    [[maybe_unused]] double spatial_scale,
    [[maybe_unused]] int64_t output_h,
    [[maybe_unused]] int64_t output_w,
    [[maybe_unused]] int64_t sampling_ratio,
    [[maybe_unused]] bool aligned) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor roi_align_bwd_wrap(
    [[maybe_unused]] const at::Tensor& grad_out,
    [[maybe_unused]] const at::Tensor& rois,
    [[maybe_unused]] double spatial_scale,
    [[maybe_unused]] int64_t output_h,
    [[maybe_unused]] int64_t output_w,
    [[maybe_unused]] int64_t bs,
    [[maybe_unused]] int64_t ch,
    [[maybe_unused]] int64_t h,
    [[maybe_unused]] int64_t w,
    [[maybe_unused]] int64_t sampling_ratio,
    [[maybe_unused]] bool aligned) {
  EAGER_NOT_SUPPORTED;
}
} // namespace ops
} // namespace vision
