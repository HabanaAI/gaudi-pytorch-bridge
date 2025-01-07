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

set(CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

add_habana_library(habana_dl_app SHARED main.cpp)
set_target_properties(habana_dl_app PROPERTIES PREFIX "")
target_include_directories(habana_dl_app PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_directories(habana_dl_app PRIVATE $ENV{BUILD_ROOT_LATEST})
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(
  habana_dl_app
  PUBLIC torch nlohmann_json aeon ${TORCH_PYTHON_LIBRARY}
  PRIVATE Python::Python)
if (UPSTREAM_COMPILE)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  install(TARGETS habana_dl_app LIBRARY DESTINATION habana_torch_dataloader/upstream)
else()
  install(TARGETS habana_dl_app LIBRARY DESTINATION habana_torch_dataloader/fork)
endif()
