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
#include "habana_kernels/eager_op.h"

namespace habana {
namespace eager {
torch::jit::Stack EagerOpBase::run(const std::vector<OutputSpec>& out_spec) {
  SmallTensorVector input_pt_vec, input_backend_pt_vec;
  habana::eager::MetaDataMap metadata;
  create_inputs(input_pt_vec, metadata);
  convert_inputs_to_backend_tensors(input_pt_vec, input_backend_pt_vec);

  habana::eager::EagerExec hlexec{
      m_symbol, input_backend_pt_vec, out_spec, std::move(metadata)};

  // Launch the execution
  return hlexec.launch();
}

} // namespace eager
} // namespace habana
