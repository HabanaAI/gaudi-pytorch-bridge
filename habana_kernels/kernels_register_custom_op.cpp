/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/lazy_kernels.h"

using namespace torch;
using namespace at;
using namespace habana;
using namespace habana_lazy;

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
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      "optimizer_sparse_sgd_with_valid_count :",
      " gradients=",
      to_string(gradients),
      " weights_in=",
      to_string(weights_in),
      " moments_in=",
      to_string(moments_in),
      " indices=",
      to_string(indices),
      " learning_rate=",
      to_string(learning_rate),
      " valid_count_tensor",
      to_string(valid_count_tensor),
      " mom",
      to_string(mom),
      " nesterov",
      to_string(nesterov));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return optimizer_sparse_sgd_with_valid_count_hpu_lazy(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor,
        mom,
        nesterov);
  } else {
    return optimizer_sparse_sgd_with_valid_count_hpu(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor,
        mom,
        nesterov);
  }
}
void optimizer_sgd_momentum_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      " optimizer_sgd_momentum:",
      " gradients=",
      to_string(gradients),
      " weights=",
      to_string(weights),
      " momentum=",
      to_string(momentum),
      " epoch_num=",
      to_string(epoch_num),
      " lr=",
      to_string(lr),
      " wd=",
      to_string(wd),
      " mom=",
      to_string(mom),
      " damp=",
      to_string(damp),
      " nesterov=",
      to_string(nesterov));
  auto mom_t = get_tensor_for_scalar(mom);
  if (!habana_lazy::isDeviceInLoweringMode() &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_sgd_momentum_hpu_lazy(
        gradients, weights, momentum, epoch_num, lr, mom_t, wd, damp, nesterov);
  } else {
    optimizer_sgd_momentum_hpu(
        gradients, weights, momentum, epoch_num, lr, mom_t, wd, damp, nesterov);
  }
}

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      "optimizer_sparse_adagrad_with_valid_count :",
      " gradients=",
      to_string(gradients),
      " weights_in=",
      to_string(weights_in),
      " moments_in=",
      to_string(moments_in),
      " indices=",
      to_string(indices),
      " learning_rate=",
      to_string(learning_rate),
      " valid_count_tensor",
      to_string(valid_count_tensor));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return optimizer_sparse_adagrad_with_valid_count_hpu_lazy(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor);
  } else {
    return optimizer_sparse_adagrad_with_valid_count_hpu(
        gradients,
        weights_in,
        moments_in,
        indices,
        learning_rate,
        valid_count_tensor);
  }
}
void optimizer_adamw_hpu_wrap(
    const TensorList& gradient_vec,
    TensorList& weight_vec,
    TensorList& exp_avg_vec,
    TensorList& exp_avg_sq_vec,
    const float lr,
    at::Tensor& neg_step_t,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      "optimizer_adamw :",
      " gradient_vec=",
      to_string(gradient_vec),
      " weight_vec=",
      to_string(weight_vec),
      " exp_avg_vec=",
      to_string(exp_avg_vec),
      " exp_avg_sq_vec=",
      to_string(exp_avg_sq_vec),
      " lr=",
      to_string(lr),
      " neg_step_t",
      to_string(neg_step_t),
      " beta1",
      to_string(beta1),
      " beta2",
      to_string(beta2),
      " epsilon",
      to_string(epsilon),
      " weight_decay",
      to_string(weight_decay));
  TORCH_CHECK((weight_vec.size() > 0), "Can not process empty weight vector");
  auto lr_t = get_tensor_for_scalar(lr);
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_adamw_hpu_lazy(
        gradient_vec,
        weight_vec,
        exp_avg_vec,
        exp_avg_sq_vec,
        lr_t,
        neg_step_t,
        beta1,
        beta2,
        epsilon,
        weight_decay);
  } else {
    optimizer_adamw_hpu(
        gradient_vec,
        weight_vec,
        exp_avg_vec,
        exp_avg_sq_vec,
        lr_t,
        neg_step_t,
        beta1,
        beta2,
        epsilon,
        weight_decay);
  }
}
Tensor fused_norm_hpu_wrap(
    std::vector<at::Tensor>& grad,
    const Tensor& max_norm,
    float norm_type) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      "fused_norm :",
      " grad=",
      to_string(grad),
      " max_norm=",
      to_string(max_norm),
      " norm_type=",
      to_string(norm_type));
  TORCH_CHECK((grad.size() > 0), "Can not process empty grad vector");
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return fused_norm_hpu_lazy(grad, max_norm, norm_type);
  } else {
    return fused_norm_hpu(grad, max_norm, norm_type);
  }
}
void optimizer_adagrad_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      " optimizer_adagrad:",
      " gradients=",
      to_string(gradients),
      " weights=",
      to_string(weights),
      " variances=",
      to_string(variances),
      " epoch_num=",
      to_string(epoch_num),
      " lr=",
      to_string(lr),
      " wd=",
      to_string(wd),
      " lrd=",
      to_string(lrd),
      " epsilon=",
      to_string(epsilon));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_adagrad_hpu_lazy(
        gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
  } else {
    optimizer_adagrad_hpu(
        gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
  }
}

void optimizer_ema_hpu_wrap(
    const TensorList& model_inputs,
    TensorList& updated_ema,
    const at::Tensor& decay) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      " optimizer_ema:",
      " model_inputs=",
      to_string(model_inputs),
      " updated_ema=",
      to_string(updated_ema),
      " decay=",
      to_string(decay));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_ema_hpu_lazy(model_inputs, updated_ema, decay);
  }

  return;
}

void optimizer_sgd_hpu_wrap(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      " optimizer_sgd:",
      " gradients=",
      to_string(gradients),
      " weights=",
      to_string(weights),
      " lr=",
      to_string(lr),
      " wd=",
      to_string(wd),
      " mom=",
      to_string(mom),
      " damp=",
      to_string(damp),
      " nesterov=",
      to_string(nesterov));
  if (!habana_lazy::isDeviceInLoweringMode() &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_sgd_hpu_lazy(gradients, weights, lr, wd, mom, damp, nesterov);
  } else {
    optimizer_sgd_hpu(gradients, weights, lr, wd, mom, damp, nesterov);
  }
}

Tensor optimizer_lamb_fused_norm_hpu_wrap(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      " optimizer_lamb_fused_norm:",
      " grad=",
      to_string(grad),
      "max_grad_norm=",
      to_string(max_grad_norm));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return optimizer_lamb_fused_norm_hpu_lazy(grad, max_grad_norm);
  } else {
    return optimizer_lamb_fused_norm_hpu(grad, max_grad_norm);
  }
}

void optimizer_lamb_phase1_hpu_wrap(
    const std::vector<at::Tensor>& gradients,
    std::vector<at::Tensor>& hl_adam_step_vec,
    std::vector<at::Tensor>& hl_adam_norm_vec,
    std::vector<at::Tensor>& hl_weight_norm_vec,
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
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      " optimizer_lamb_phase1:",
      " gradients=",
      to_string(gradients),
      "weights=",
      to_string(weights),
      "exp_avg=",
      to_string(exp_avg),
      "exp_avg_sq=",
      to_string(exp_avg_sq),
      "clip_global_grad_norm=",
      to_string(clip_global_grad_norm),
      "grad_averaging=",
      to_string(grad_averaging),
      "lr=",
      to_string(lr),
      "beta1=",
      to_string(beta1),
      "beta2=",
      to_string(beta2),
      "epsilon=",
      to_string(epsilon),
      "step=",
      to_string(step),
      "bias_correction=",
      to_string(bias_correction),
      "weight_decay=",
      to_string(weight_decay));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_lamb_phase1_hpu_lazy(
        gradients,
        hl_adam_step_vec,
        hl_adam_norm_vec,
        hl_weight_norm_vec,
        weights,
        exp_avg,
        exp_avg_sq,
        clip_global_grad_norm,
        grad_averaging,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        bias_correction,
        weight_decay);
  } else {
    optimizer_lamb_phase1_hpu(
        gradients,
        hl_adam_step_vec,
        hl_adam_norm_vec,
        hl_weight_norm_vec,
        weights,
        exp_avg,
        exp_avg_sq,
        clip_global_grad_norm,
        grad_averaging,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        bias_correction,
        weight_decay);
  }
}

void optimizer_lamb_phase2_hpu_wrap(
    std::vector<at::Tensor>& weight_vec,
    const std::vector<at::Tensor>& adam_norm_vec,
    const std::vector<at::Tensor>& weight_norm_vec,
    const std::vector<at::Tensor>& adam_step_vec,
    const std::vector<at::Tensor>& trust_ratio_vec,
    const float step,
    const float weight_decay,
    const int use_lamb) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      " optimizer_lamb_phase2:",
      " weight_vec=",
      to_string(weight_vec),
      "adam_norm_vec=",
      to_string(adam_norm_vec),
      "weight_norm_vec=",
      to_string(weight_norm_vec),
      "adam_step_vec=",
      to_string(adam_step_vec),
      "trust_ratio_vec=",
      to_string(trust_ratio_vec),
      "step=",
      to_string(step),
      "weight_decay=",
      to_string(weight_decay),
      "use_lamb=",
      to_string(use_lamb));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    optimizer_lamb_phase2_hpu_lazy(
        weight_vec,
        adam_norm_vec,
        weight_norm_vec,
        adam_step_vec,
        trust_ratio_vec,
        step,
        weight_decay,
        use_lamb);
  } else {
    optimizer_lamb_phase2_hpu(
        weight_vec,
        adam_norm_vec,
        weight_norm_vec,
        adam_step_vec,
        trust_ratio_vec,
        step,
        weight_decay,
        use_lamb);
  }
}

Tensor embedding_bag_sum_hpu_wrap(
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      "embedding_bag_sum :",
      " input=",
      to_string(input),
      " indices=",
      to_string(indices),
      " offsets=",
      to_string(offsets),
      " valid_count=",
      to_string(valid_count),
      " kernel_mode=",
      to_string(kernel_mode));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return embedding_bag_sum_hpu_lazy(
        input, indices, offsets, valid_count, kernel_mode);

  } else {
    return embedding_bag_sum_hpu(
        input, indices, offsets, valid_count, kernel_mode);
  }
};
Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap(
    Tensor& out,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& valid_count,
    int64_t kernel_mode) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      "embedding_bag_sum_bwd_out :",
      " out=",
      to_string(out),
      " input=",
      to_string(input),
      " indices=",
      to_string(indices),
      " offsets=",
      to_string(offsets),
      " valid_count=",
      to_string(valid_count),
      " kernel_mode=",
      to_string(kernel_mode));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return embedding_bag_sum_bwd_out_kernel_mode_hpu_lazy(
        out, input, indices, offsets, valid_count, kernel_mode);

  } else {
    return embedding_bag_sum_bwd_out_kernel_mode_hpu(
        out, input, indices, offsets, valid_count, kernel_mode);
  }
};

Tensor gather2d_hpu_wrap(
    const Tensor& input,
    const Tensor& indices,
    int64_t validCount) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      "gather2d :",
      " input=",
      to_string(input),
      " indices=",
      to_string(indices),
      "validCount=",
      to_string(validCount));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return gather2d_hpu_lazy(input, indices, validCount);

  } else {
    return gather2d_hpu(input, indices, validCount);
  }
};

Tensor torchvision_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    double iou_threshold) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      " torchvision_nms:",
      " boxes=",
      to_string(boxes),
      "scores=",
      to_string(scores),
      "iou_threshold=",
      to_string(iou_threshold));

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return habana_nms_hpu_lazy(
        boxes, scores, iou_threshold, -std::numeric_limits<float>::max());
  } else {
    return habana_nms_hpu(
        boxes, scores, iou_threshold, -std::numeric_limits<float>::max());
  }
}

Tensor habana_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    float iou_threshold,
    float score_threshold) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      " habana_nms:",
      " boxes=",
      to_string(boxes),
      "scores=",
      to_string(scores),
      "iou_threshold=",
      to_string(iou_threshold),
      "score_threshold=",
      to_string(score_threshold));
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    return habana_nms_hpu_lazy(boxes, scores, iou_threshold, score_threshold);
  } else {
    return habana_nms_hpu(boxes, scores, iou_threshold, score_threshold);
  }
}

Tensor batched_nms_hpu_wrap(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    const at::Tensor& indices,
    float iou_threshold) {
  PT_OP_TRACE;
  PT_KERNEL_DEBUG(
      " batched_nms:",
      " boxes=",
      to_string(boxes),
      "scores=",
      to_string(scores),
      "indices=",
      to_string(indices),
      "iou_threshold=",
      to_string(iou_threshold));
  return batched_nms_hpu_lazy(boxes, scores, indices, iou_threshold);
}