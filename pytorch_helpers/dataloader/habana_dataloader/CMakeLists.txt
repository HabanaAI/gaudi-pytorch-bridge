# ##############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited. This file contains Habana Labs, Ltd. proprietary and
# confidential information and is subject to the confidentiality and license
# agreements under which it was provided.
#
# ##############################################################################

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
install(TARGETS habana_dl_app LIBRARY DESTINATION habana_torch_dataloader)
