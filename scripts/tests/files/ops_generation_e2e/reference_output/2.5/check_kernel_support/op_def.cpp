#include "__ilshift__.h"
#include "_foreach_add.h"
#include "_fused_dropout.h"
#include "_native_batch_norm_legit.h"
#include "addbmm.h"
#include "as_strided.h"
#include "bitwise_left_shift.h"
#include "bucketize.h"
#include "clone.h"
#include "convolution_backward_overrideable.h"
#include "elu.h"
#include "eq.h"
#include "isfinite.h"
#include "linear_backward.h"
#include "mul.h"
#include "native_dropout.h"
#include "native_group_norm.h"
#include "prod.h"
#include "sort.h"
#include "squeeze.h"
#include "upsample_bicubic2d.h"

#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/tensorexpr/tensorexpr_init.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include "cpu_fallback.h"

using habana_helpers::DTypeHelper;
using namespace torch::jit;
#include<hpu_op0.h>
#include<hpu_op1.h>
#include<hpu_op2.h>
#include<hpu_op3.h>
#include<hpu_op4.h>
#include<hpu_op5.h>
#include<hpu_op6.h>
#include<hpu_op7.h>
#include<hpu_op8.h>
#include<hpu_op9.h>
#include<hpu_op_custom.h>
std::unordered_map<std::string, std::function<bool(c10::FunctionSchema&, bool, bool, const py::list&, py::args& args, const py::kwargs& kwargs)>> fallback_support_check_map = {
{"__ilshift__", &check_support<habana::shared_layer___ilshift__>},
{"_foreach_add_", &check_support<habana::shared_layer__foreach_add_>},
{"_fused_dropout", &check_support<habana::shared_layer__fused_dropout>},
{"_native_batch_norm_legit", &check_support<habana::shared_layer__native_batch_norm_legit>},
{"addbmm", &check_support<habana::shared_layer_addbmm>},
{"as_strided", &check_support<habana::shared_layer_as_strided>},
{"bitwise_left_shift", &check_support<habana::shared_layer_bitwise_left_shift>},
{"bucketize", &check_support<habana::shared_layer_bucketize>},
{"clone", &check_support<habana::shared_layer_clone>},
{"convolution_backward_overrideable", &check_support<habana::shared_layer_convolution_backward_overrideable>},
{"elu", &check_support<habana::shared_layer_elu>},
{"eq_out", &check_support<habana::shared_layer_eq_out>},
{"exp_fast_math", &check_support<habana::shared_layer_exp_fast_math>},
{"isfinite", &check_support<habana::shared_layer_isfinite>},
{"linear_backward", &check_support<habana::shared_layer_linear_backward>},
{"mul_out", &check_support<habana::shared_layer_mul_out>},
{"native_dropout", &check_support<habana::shared_layer_native_dropout>},
{"native_group_norm", &check_support<habana::shared_layer_native_group_norm>},
{"prod_out", &check_support<habana::shared_layer_prod_out>},
{"sort_out", &check_support<habana::shared_layer_sort_out>},
{"squeeze", &check_support<habana::shared_layer_squeeze>},
{"upsample_bicubic2d", &check_support<habana::shared_layer_upsample_bicubic2d>},
};
std::set<std::string> hpu_shared_layer_unsupported_ops = { "dropout",
"matmul" };
