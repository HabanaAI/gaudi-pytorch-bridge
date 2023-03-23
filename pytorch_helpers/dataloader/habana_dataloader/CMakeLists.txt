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
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(TARGET_NAME "habana_dl_app")
add_library(${TARGET_NAME} SHARED main.cpp)
set_target_properties(${TARGET_NAME} PROPERTIES PREFIX "")
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(${TARGET_NAME} PUBLIC "${PYTHON_INCLUDE_DIR}")
target_link_directories(${TARGET_NAME} PRIVATE $ENV{BUILD_ROOT_LATEST})
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(${TARGET_NAME} PUBLIC torch nlohmann_json aeon ${TORCH_PYTHON_LIBRARY})
install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION habana_torch_dataloader)
