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
#include <ATen/ATen.h>
#include <torch/library.h>

using namespace at;
using namespace torch;

// Helper code to calculate second derivative of instance_norm.
namespace {
Tensor toNonOptTensor(const c10::optional<Tensor>& t) {
  return t.has_value() ? *t : Tensor();
}

static bool isDefined(const c10::optional<Tensor>& t) {
  return t.has_value() && t->defined();
}

static Tensor sum_exclude_dim1(const Tensor& to_sum, bool keepdim = true) {
  auto r = to_sum.sum(0, keepdim);
  int64_t start_point_exclusive = keepdim ? 1 : 0;
  for (int64_t dim = r.dim() - 1; dim > start_point_exclusive; dim--) {
    r = r.sum(dim, keepdim);
  }
  return r;
}

// Helper for batchnorm_double_backward
// similar to expand_as below, but doesn't do the expand_as; operates as if
// reductions were done with keepdim=True
static Tensor unsqueeze_dim1(const Tensor& src, const Tensor& target) {
  auto src_expanded = src;
  while (src_expanded.sizes().size() < target.sizes().size() - 1) {
    src_expanded = src_expanded.unsqueeze(1);
  }
  if (src_expanded.sizes().size() == target.sizes().size() - 1) {
    src_expanded = src_expanded.unsqueeze(0);
  }
  return src_expanded;
}

// Helper for batchnorm_double_backward
// because gamma/ggG/ggB are 1-dimensional and represent dim==1, we can't
// do a straight expansion because it won't follow the broadcasting rules.
static Tensor expand_as_dim1(const Tensor& src, const Tensor& target) {
  auto src_expanded = src;
  while (src_expanded.sizes().size() < target.sizes().size() - 1) {
    src_expanded = src_expanded.unsqueeze(1);
  }
  return src_expanded.expand_as(target);
}
} // namespace

std::tuple<Tensor, Tensor, Tensor> batchnorm_double_backward(
    const Tensor& input,
    const c10::optional<Tensor>& gamma,
    const Tensor& ggI,
    const Tensor& ggG,
    const Tensor& ggB,
    const Tensor& gO,
    const c10::optional<Tensor>& running_mean,
    const c10::optional<Tensor>& running_var,
    bool training,
    double eps,
    const c10::optional<Tensor>& save_mean,
    const c10::optional<Tensor>& save_invstd,
    std::array<bool, 3> output_mask) {
  bool affine = isDefined(gamma);
  // TODO: Do we have a ScalarOrTensor type?  Would such a thing exist?
  Tensor gamma_expanded;
  Tensor ggG_expanded, ggB_expanded;
  if (affine) {
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    gamma_expanded = expand_as_dim1(*gamma, input);
    if (ggG.defined()) {
      ggG_expanded = expand_as_dim1(ggG, input);
    }
    if (ggB.defined()) {
      ggB_expanded = expand_as_dim1(ggB, input);
    }
  } else {
    gamma_expanded = at::ones({}, input.options());
  }

  // define some terms we will reuse
  auto M = input.size(0);
  for (auto s : input.sizes().slice(2)) {
    M *= s;
  }
  // for half inputs, save_mean, save_invstd are float (ideally, we would cast
  // everything else, but not now)
  auto mu = unsqueeze_dim1(
      training ? toNonOptTensor(save_mean).to(input.scalar_type())
               : toNonOptTensor(running_mean),
      input);
  auto input_sub_mu = input - mu;
  auto sigma2_eps_neg_1_2 = unsqueeze_dim1(
      training ? toNonOptTensor(save_invstd).to(input.scalar_type())
               : toNonOptTensor(running_var).add(Scalar(eps)).pow(-0.5),
      input);
  auto sigma2_eps_neg_1 = sigma2_eps_neg_1_2.pow(2);
  auto sigma2_eps_neg_3_2 = sigma2_eps_neg_1_2.pow(3);

  // calculate gI
  auto input_mu_sigma2_neg_3_2 = input_sub_mu * sigma2_eps_neg_3_2;
  auto gOinmu_sum = sum_exclude_dim1(gO * input_sub_mu);
  auto gO_sum = sum_exclude_dim1(gO);

  Tensor gI;
  if (ggI.defined() && training) {
    auto ggI_sum = sum_exclude_dim1(ggI);
    auto ggIinmu_sum = sum_exclude_dim1(ggI * input_sub_mu);
    auto all_sub = ((ggI_sum * gO_sum).div_(M))
                       .sub_(sum_exclude_dim1(gO * ggI))
                       .add_((sigma2_eps_neg_1 * gOinmu_sum * ggIinmu_sum)
                                 .mul_(3. / static_cast<double>(M)));
    auto gI_0t = (input_mu_sigma2_neg_3_2 * all_sub).div_(M);
    auto gI_1t =
        (ggIinmu_sum * sigma2_eps_neg_3_2).div_(M) * (gO_sum.div(M) - gO);
    auto gI_2t =
        (gOinmu_sum * sigma2_eps_neg_3_2).div_(M) * (ggI_sum.div(M) - ggI);
    gI = gamma_expanded * (gI_0t.add_(gI_1t).add_(gI_2t));
  }

  // add contribution of gamma term to gI
  Tensor gI_G_term;
  if (affine && ggG.defined()) {
    if (training) {
      auto t0 = gO * sigma2_eps_neg_1_2;
      auto t1 = (sigma2_eps_neg_1_2 * gO_sum).div_(-M);
      auto t2 = (input_mu_sigma2_neg_3_2 * sum_exclude_dim1(gO * input_sub_mu))
                    .div_(-M);
      gI_G_term = ggG_expanded * (t0.add_(t1).add_(t2));
      gI = gI.defined() ? gI.add_(gI_G_term) : gI_G_term;
    } else {
      gI_G_term = ggG_expanded * sigma2_eps_neg_1_2 * gO;
      gI = gI.defined() ? gI.add_(gI_G_term) : gI_G_term;
    }
  }

  // this is the first backward's grad_input
  auto first_back_grad_input = [&](const Tensor& gO,
                                   const Tensor& gamma) -> Tensor {
    auto h0 = (gamma * sigma2_eps_neg_1_2).div_(M);
    auto h1 = (M * gO)
                  .sub_(sum_exclude_dim1(gO))
                  .sub_(
                      input_sub_mu.mul(sigma2_eps_neg_1) *
                      sum_exclude_dim1(gO * input_sub_mu));
    return h0 * h1;
  };

  // calculate gG
  Tensor gG;
  if (affine && ggI.defined()) {
    if (training) {
      // gG is just the first backwards with the gamma term removed (then
      // shaped properly)
      gG = ggI *
          first_back_grad_input(gO, at::ones({}, sigma2_eps_neg_1_2.options()));
      gG = sum_exclude_dim1(gG, false);
    } else {
      gG = sum_exclude_dim1(ggI * gO * sigma2_eps_neg_1_2, false);
    }
  }

  // calculate ggO
  Tensor ggO;
  // contribution of input term
  if (ggI.defined()) {
    if (training) {
      ggO = first_back_grad_input(ggI, gamma_expanded);
    } else {
      ggO = ggI * sigma2_eps_neg_1_2 * gamma_expanded;
    }
  }
  if (ggG.defined()) {
    auto ggO_G_term = ggG_expanded * input_sub_mu * sigma2_eps_neg_1_2;
    ggO = ggO.defined() ? ggO.add_(ggO_G_term) : ggO_G_term;
  }
  if (ggB.defined()) {
    auto ggO_B_term = std::move(ggB_expanded);
    ggO = ggO.defined() ? ggO.add_(ggO_B_term) : ggO_B_term;
  }

  if (output_mask[1] && !gG.defined()) {
    AT_ASSERTM(affine, "gamma should always be defined when it requires grad");
  }

  return std::tuple<Tensor, Tensor, Tensor>{gI, gG, ggO};
}