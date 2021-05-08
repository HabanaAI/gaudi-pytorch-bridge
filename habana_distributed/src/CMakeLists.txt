###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################

set(HCL_SRCS init.cpp ProcessGroupHCL.cpp hcl_communicator.cpp)
add_library(habana_torch_hcl SHARED ${HCL_SRCS})
set_target_properties(habana_torch_hcl PROPERTIES OUTPUT_NAME ${LIB_NAME})
set_target_properties(habana_torch_hcl PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(habana_torch_hcl PUBLIC -Wall
        -Wno-sign-compare
        -Wno-unused-function)

target_include_directories(habana_torch_hcl PUBLIC ${PYTHON_INCLUDE_DIRS} ${PYTORCH_INCLUDE_DIRS} ./)
target_include_directories(habana_torch_hcl PUBLIC "$ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers" "$ENV{SYNAPSE_ROOT}/include" "$ENV{ABSEIL_CPP_INCLUDE_DIR}")


add_library(Synapse SHARED IMPORTED)
set_target_properties(Synapse PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libSynapse.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{SYNAPSE_ROOT}/include"
  )

add_library(pytorch_synapse_helpers SHARED IMPORTED)
set_target_properties(pytorch_synapse_helpers PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libpytorch_synapse_helpers.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers"
  )

add_library(habana_pytorch_plugin SHARED IMPORTED)
set_target_properties(habana_pytorch_plugin PROPERTIES
  IMPORTED_LOCATION "$ENV{BUILD_ROOT_LATEST}/libhabana_pytorch_plugin.so"
  INTERFACE_INCLUDE_DIRECTORIES "$ENV{PYTORCH_MODULES_ROOT_PATH}/pytorch_helpers"
  )

add_library(torch_python SHARED IMPORTED)
set_target_properties(torch_python PROPERTIES
  IMPORTED_LOCATION "${PYTORCH_LIBRARY_DIRS}/libtorch_python.so"
  INTERFACE_INCLUDE_DIRECTORIES "${PYTORCH_INCLUDE_DIRS}"
  )

target_link_libraries(habana_torch_hcl PUBLIC ${DEPENDS_LIB})
target_link_libraries(habana_torch_hcl PUBLIC torch_python)
target_link_libraries(habana_torch_hcl PUBLIC pytorch_synapse_helpers Synapse habana_pytorch_plugin)

set_target_properties(habana_torch_hcl PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

install(TARGETS habana_torch_hcl LIBRARY DESTINATION lib)
