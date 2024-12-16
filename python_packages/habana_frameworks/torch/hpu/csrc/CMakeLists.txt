###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

add_habana_library(_hpu_C SHARED bindings.cpp Exceptions.cpp Stream.cpp Event.cpp Module.cpp)
target_compile_definitions(_hpu_C PRIVATE TORCH_EXTENSION_NAME=_hpu_C)

set_target_properties(_hpu_C PROPERTIES PREFIX "")
target_link_libraries(_hpu_C PUBLIC bindings torch absl::variant absl::flat_hash_map absl::strings
                                    ${TORCH_PYTHON_LIBRARY})
install(TARGETS _hpu_C LIBRARY DESTINATION habana_torch_plugin/${PYBIND_INSTALL_DIRNAME})
