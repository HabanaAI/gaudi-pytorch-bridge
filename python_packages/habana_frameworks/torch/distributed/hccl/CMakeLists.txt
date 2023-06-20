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

set(TARGET_HCCL_LAZY_WITH_NON_LAZY_COLL_NAME "_hccl_C")
add_library(${TARGET_HCCL_LAZY_WITH_NON_LAZY_COLL_NAME} SHARED
    process_group_hccl_base.cpp
    ProcessGroupHCCL.cpp
    reducer_timer.cpp)
target_compile_definitions(${TARGET_HCCL_LAZY_WITH_NON_LAZY_COLL_NAME} PRIVATE TORCH_EXTENSION_NAME=${TARGET_HCCL_LAZY_WITH_NON_LAZY_COLL_NAME})
set_target_properties(${TARGET_HCCL_LAZY_WITH_NON_LAZY_COLL_NAME} PROPERTIES PREFIX "")
target_compile_options(${TARGET_HCCL_LAZY_WITH_NON_LAZY_COLL_NAME} PRIVATE -Wall -Wextra -Werror)
target_link_libraries(${TARGET_HCCL_LAZY_WITH_NON_LAZY_COLL_NAME} PUBLIC bindings torch absl::strings ${TORCH_PYTHON_LIBRARY} habana_pytorch_backend)
install(TARGETS ${TARGET_HCCL_LAZY_WITH_NON_LAZY_COLL_NAME} LIBRARY DESTINATION habana_torch_plugin/distributed)

set(TARGET_HCCL_LAZY_WITH_LAZY_COLL_NAME "_hccl_lazy_C")
add_library(${TARGET_HCCL_LAZY_WITH_LAZY_COLL_NAME} SHARED
    process_group_lazy_hccl.cpp
    reducer_timer.cpp)
target_compile_definitions(${TARGET_HCCL_LAZY_WITH_LAZY_COLL_NAME} PRIVATE TORCH_EXTENSION_NAME=${TARGET_HCCL_LAZY_WITH_LAZY_COLL_NAME})
set_target_properties(${TARGET_HCCL_LAZY_WITH_LAZY_COLL_NAME} PROPERTIES PREFIX "")
target_compile_options(${TARGET_HCCL_LAZY_WITH_LAZY_COLL_NAME} PRIVATE -Wall -Wextra -Werror)
target_link_libraries(${TARGET_HCCL_LAZY_WITH_LAZY_COLL_NAME} PUBLIC bindings torch absl::strings ${TORCH_PYTHON_LIBRARY} habana_pytorch_backend)
install(TARGETS ${TARGET_HCCL_LAZY_WITH_LAZY_COLL_NAME} LIBRARY DESTINATION habana_torch_plugin/distributed)

set(TARGET_HCCL_EAGER_COLL_NAME "_hccl_eager_C")
add_library(${TARGET_HCCL_EAGER_COLL_NAME} SHARED
    process_group_hccl_base.cpp
    process_group_eager_hccl.cpp
    reducer_timer.cpp)
target_compile_definitions(${TARGET_HCCL_EAGER_COLL_NAME} PRIVATE TORCH_EXTENSION_NAME=${TARGET_HCCL_EAGER_COLL_NAME})
set_target_properties(${TARGET_HCCL_EAGER_COLL_NAME} PROPERTIES PREFIX "")
target_compile_options(${TARGET_HCCL_EAGER_COLL_NAME} PRIVATE -Wall -Wextra -Werror)
target_link_libraries(${TARGET_HCCL_EAGER_COLL_NAME} PUBLIC bindings torch absl::strings ${TORCH_PYTHON_LIBRARY} habana_pytorch_backend)
install(TARGETS ${TARGET_HCCL_EAGER_COLL_NAME} LIBRARY DESTINATION habana_torch_plugin/distributed)
