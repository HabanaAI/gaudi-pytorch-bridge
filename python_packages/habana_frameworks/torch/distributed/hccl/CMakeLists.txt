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

install(TARGETS _hccl_C _hccl_lazy_C _hccl_eager_C LIBRARY DESTINATION habana_torch_plugin/distributed)
