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

add_habana_library(_media_pyt_bridge_C SHARED media_pyt_bridge.cpp pyt_media_proxy.cpp pyt_media_shim.cpp)

target_compile_definitions(_media_pyt_bridge_C PRIVATE TORCH_EXTENSION_NAME=_media_pyt_bridge_C)

set_target_properties(_media_pyt_bridge_C PROPERTIES PREFIX "")
target_link_libraries(_media_pyt_bridge_C PUBLIC bindings torch absl::variant  ${TORCH_PYTHON_LIBRARY} Media)

install(TARGETS _media_pyt_bridge_C LIBRARY DESTINATION habana_torch_plugin/${PYBIND_INSTALL_DIRNAME})
