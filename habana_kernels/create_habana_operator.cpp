/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/conv_kernels.h"
#include "habana_kernels/habana_operator.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_kernels/pool_kernels.h"
#include "habana_kernels/softmax_kernels.h"
#include "habana_kernels/unary_kernels.h"

namespace habana {
using HabanaOperatorPtr = std::shared_ptr<HabanaOperator>;

HabanaOperatorPtr CreateHabanaOperator(
    const int device_id,
    const std::string& node_name,
    c10::ScalarType node_type) {
  HabanaOperatorPtr op = nullptr;

  if ("aten::relu" == node_name) {
    op = std::make_shared<ReluOperator>(device_id, node_type);
  } else if ("aten::sigmoid" == node_name) {
    op = std::make_shared<SigmoidOperator>(device_id, node_type);
  } else if ("aten::abs" == node_name) {
    op = std::make_shared<AbsOperator>(device_id, node_type);
  } else if ("aten::log_softmax" == node_name) {
    op = std::make_shared<LogSoftmaxOperator>(device_id, node_type);
  } else if ("aten::convolution_overrideable" == node_name) {
    op = std::make_shared<ConvOperator>(device_id, node_type);
  } else if ("aten::_log_softmax_backward_data" == node_name) {
    op = std::make_shared<LogSoftmaxBackwardOperator>(device_id, node_type);
  } else if ("aten::conv2d" == node_name) {
    op = std::make_shared<Conv2dOperator>(device_id, node_type);
  } else if ("aten::mm" == node_name) {
    op = std::make_shared<MMOperator>(device_id);
  } else if ("aten::max_pool2d_with_indices" == node_name) {
    op = std::make_shared<MaxPool2dWithIndicesOperator>(device_id, node_type);
  } else if ("aten::max_pool2d" == node_name) {
    op = std::make_shared<MaxPool2dOperator>(device_id, node_type);
  } else if ("aten::permute" == node_name) {
    op = std::make_shared<PermuteOperator>(device_id, node_type);
  } else if ("aten::reshape" == node_name) {
    op = std::make_shared<ReshapeOperator>(device_id, node_type);
  } else if ("aten::t" == node_name) {
    op = std::make_shared<TOperator>(device_id, node_type);
  } else if ("aten::mul" == node_name) {
    op = std::make_shared<MulOperator>(device_id, node_type);
  } else if ("aten::div" == node_name) {
    op = std::make_shared<DivOperator>(device_id, node_type);
  } else if ("aten::add" == node_name) {
    op = std::make_shared<AddOperator>(device_id, node_type);
  }

  // Returning a null pointer for cases not added yet,
  // we can add assert once all kernels are added
  return op;
}
} // namespace habana
