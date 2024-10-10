/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
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

#include <string>

class CacheVersion {
 public:
  // returns string containing Hash calculated for content of:
  //    - habana_device, synapse_helpers, Synapse and tpc_kernels (binary
  //    content)
  //    - env variables impacting the compilation of synGraph to synRecipe
  static std::string libs_env_hash();
};