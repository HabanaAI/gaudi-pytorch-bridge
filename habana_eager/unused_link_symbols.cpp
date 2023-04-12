/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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

#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
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
    const c10::optional<at::Tensor>& weight,
    const c10::optional<at::Tensor>& bias,
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
        const c10::optional<at::Tensor>& running_mean,
        const c10::optional<at::Tensor>& running_var,
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
        const c10::optional<at::Tensor>& weight,
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
    const c10::optional<at::Tensor>& weight,
    const at::Tensor& mean_dy,
    const at::Tensor& mean_dy_xmu,
    const at::Tensor& count) {
  FALLBACK_UNSUPPORTED_OP2(
      batch_norm_backward_elemt,
      PARAMS2(
          grad_out, input, mean, invstd, weight, mean_dy, mean_dy_xmu, count));
}

at::Tensor hpu_wrap::repeat_interleave(
    const at::Tensor& self,
    c10::optional<int64_t> output_size) {
  FALLBACK_UNSUPPORTED_OP2_O(
      repeat_interleave, PARAMS2(self, output_size), Tensor);
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

Tensor& hpu_wrap::index_fill_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Scalar& value) {
  FALLBACK_UNSUPPORTED_OP2_O(
      index_fill_, PARAMS2(self, dim, index, value), int_Scalar);
}

Tensor& hpu_wrap::masked_select_out(
    const at::Tensor& self,
    const at::Tensor& mask,
    at::Tensor& out) {
  FALLBACK_UNSUPPORTED_OP2_O(masked_select, PARAMS2(self, mask, out), out);
}

Tensor hpu_wrap::masked_select(const at::Tensor& self, const at::Tensor& mask) {
  FALLBACK_UNSUPPORTED_OP2(masked_select, PARAMS2(self, mask));
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

} // namespace habana_lazy

void optimizer_adagrad_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  EAGER_NOT_SUPPORTED;
}

void optimizer_sgd_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_wrap(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor,
    float mom,
    bool nesterov) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<at::Tensor&, at::Tensor&> cast_to_fp8_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor cast_from_fp8_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    at::ScalarType out_dtype) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_gelu_v2_wrap(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor) {
  EAGER_NOT_SUPPORTED;
}

Tensor torchvision_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    double iou_threshold) {
  EAGER_NOT_SUPPORTED;
}

Tensor batched_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    const at::Tensor& indices,
    float iou_threshold) {
  EAGER_NOT_SUPPORTED;
}

Tensor embedding_bag_sum_hpu_wrap(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  EAGER_NOT_SUPPORTED;
}

Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  EAGER_NOT_SUPPORTED;
}

Tensor habana_cast_to_fp8_wrap(
    const at::Tensor& input,
    bool stochastic_rounding,
    int seed) {
  EAGER_NOT_SUPPORTED;
}

std::vector<at::Tensor> habana_permute_1D_sparse_data_wrap(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights) {
  EAGER_NOT_SUPPORTED;
}

std::vector<at::Tensor> habana_permute_2D_sparse_data_wrap(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor habana_split_permute_cat_wrap(
    const at::Tensor& input,
    const at::Tensor& indices,
    int64_t batch_size,
    int64_t num_features,
    int64_t dims) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor habana_expand_into_jagged_permute_wrap(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t output_size) {
  EAGER_NOT_SUPPORTED;
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>
habana_bounds_check_indices_wrap(
    at::Tensor& indices,
    at::Tensor& offsets,
    at::Tensor& warning,
    const at::Tensor& rows_per_table,
    int64_t bounds_check_mode,
    const c10::optional<at::Tensor>& weights) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor matmul_ex_wrap(
    const at::Tensor& self,
    const at::Tensor& other,
    at::ScalarType dtype) {
  EAGER_NOT_SUPPORTED;
}
std::tuple<at::Tensor, at::Tensor> matmul_ex_backward_wrap(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& other,
    at::ScalarType dtype) {
  EAGER_NOT_SUPPORTED;
}
at::Tensor linear_ex_wrap(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const at::ScalarType dtype) {
  EAGER_NOT_SUPPORTED;
}
std::vector<at::Tensor> linear_ex_backward_wrap(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& bias_grad_opt,
    const at::ScalarType dtype) {
  EAGER_NOT_SUPPORTED;
}

Tensor habana_random_seed_wrap(const at::Tensor& input) {
  EAGER_NOT_SUPPORTED;
}

namespace vision {
namespace ops {
at::Tensor roi_align_fwd_wrap(
    const at::Tensor& images,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t output_h,
    int64_t output_w,
    int64_t sampling_ratio,
    bool aligned) {
  EAGER_NOT_SUPPORTED;
}

at::Tensor roi_align_bwd_wrap(
    const at::Tensor& grad_out,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t output_h,
    int64_t output_w,
    int64_t bs,
    int64_t ch,
    int64_t h,
    int64_t w,
    int64_t sampling_ratio,
    bool aligned) {
  EAGER_NOT_SUPPORTED;
}
} // namespace ops
} // namespace vision
