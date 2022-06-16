/*******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************/
#pragma once

#include <string>
class LibSynapseLoader {
 public:
  static void EnsureLoaded();
  static LibSynapseLoader& GetInstance();
  LibSynapseLoader(LibSynapseLoader const&) = delete;
  void operator=(LibSynapseLoader const&) = delete;
  std::string lib_path() {
    return synapse_lib_path_;
  }
  void* DlSym(const char* sym) const;

 private:
  LibSynapseLoader();
  ~LibSynapseLoader();
  void* synapse_lib_handle_;
  std::string synapse_lib_path_;
};
