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

add_habana_library(_debug_C SHARED bindings.cpp)
target_compile_definitions(_debug_C PRIVATE TORCH_EXTENSION_NAME=_debug_C)
set_target_properties(_debug_C PROPERTIES PREFIX "")
target_link_libraries(_debug_C PUBLIC bindings torch absl::flat_hash_map absl::strings ${TORCH_PYTHON_LIBRARY})

add_habana_library(_debug_eager_C SHARED bindings_eager.cpp)
target_compile_definitions(_debug_eager_C PRIVATE TORCH_EXTENSION_NAME=_debug_eager_C)
set_target_properties(_debug_eager_C PROPERTIES PREFIX "")
target_link_libraries(_debug_eager_C PUBLIC bindings torch absl::flat_hash_map absl::strings ${TORCH_PYTHON_LIBRARY})

add_habana_library(_debug_lazy_C SHARED bindings_lazy.cpp)
target_compile_definitions(_debug_lazy_C PRIVATE TORCH_EXTENSION_NAME=_debug_lazy_C)
set_target_properties(_debug_lazy_C PROPERTIES PREFIX "")
target_link_libraries(_debug_lazy_C PUBLIC bindings torch absl::flat_hash_map absl::strings ${TORCH_PYTHON_LIBRARY})

install(TARGETS _debug_C _debug_eager_C _debug_lazy_C LIBRARY DESTINATION habana_torch_plugin/${PYBIND_INSTALL_DIRNAME})
