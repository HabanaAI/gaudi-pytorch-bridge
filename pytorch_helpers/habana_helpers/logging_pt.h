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
#pragma once

#include "habana_helpers/logging.h"

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/SmallVector.h>
#include <c10/util/typeid.h>
#include <torch/csrc/jit/ir/ir.h>

CREATE_OSTREAM_FORMATTER(c10::ArrayRef<long int>);
CREATE_OSTREAM_FORMATTER(c10::Device);
CREATE_OSTREAM_FORMATTER(c10::IValue);
CREATE_OSTREAM_FORMATTER(caffe2::TypeMeta);
CREATE_OSTREAM_FORMATTER(torch::jit::Node);
CREATE_OSTREAM_FORMATTER(torch::jit::Graph);
CREATE_OSTREAM_FORMATTER(at::Tensor);

template <>
struct fmt::formatter<c10::SmallVector<long int, 5>> : ostream_formatter {};
