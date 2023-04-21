/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include <torch/script.h>
#include <torch/torch.h>
#include "metrics_test_kernel.h"
static void trigger_test_metrics() {
  metrics_trigger();
}
TORCH_LIBRARY(test_ops, m) {
  m.def("trigger_test_metrics", &trigger_test_metrics);
}
