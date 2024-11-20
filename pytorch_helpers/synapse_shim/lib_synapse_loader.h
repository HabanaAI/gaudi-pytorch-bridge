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
