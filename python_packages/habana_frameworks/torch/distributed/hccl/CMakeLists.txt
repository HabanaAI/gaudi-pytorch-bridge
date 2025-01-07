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

add_habana_library(_hccl_C SHARED process_group_hccl_base.cpp ProcessGroupHCCL.cpp reducer_timer.cpp process_group_hccl_ops.cpp)
target_compile_definitions(_hccl_C PRIVATE TORCH_EXTENSION_NAME=_hccl_C)
target_link_libraries(_hccl_C PUBLIC bindings torch absl::strings ${TORCH_PYTHON_LIBRARY} habana_pytorch_backend)

add_habana_library(_hccl_lazy_C SHARED process_group_lazy_hccl.cpp reducer_timer.cpp  process_group_hccl_ops.cpp)
target_compile_definitions(_hccl_lazy_C PRIVATE TORCH_EXTENSION_NAME=_hccl_lazy_C)
target_link_libraries(_hccl_lazy_C PUBLIC bindings torch absl::strings ${TORCH_PYTHON_LIBRARY} habana_pytorch_backend)

add_habana_library(_hccl_eager_C SHARED process_group_hccl_base.cpp process_group_eager_hccl.cpp reducer_timer.cpp process_group_hccl_ops.cpp)
target_compile_definitions(_hccl_eager_C PRIVATE TORCH_EXTENSION_NAME=_hccl_eager_C)
target_link_libraries(_hccl_eager_C PUBLIC bindings torch absl::strings ${TORCH_PYTHON_LIBRARY} habana_pytorch_backend)

set_target_properties(_hccl_C _hccl_lazy_C _hccl_eager_C PROPERTIES PREFIX "")

install(TARGETS _hccl_C _hccl_lazy_C _hccl_eager_C LIBRARY DESTINATION habana_torch_plugin/${PYBIND_INSTALL_DIRNAME})
