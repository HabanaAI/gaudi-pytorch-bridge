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

install(TARGETS _debug_C _debug_eager_C _debug_lazy_C LIBRARY DESTINATION habana_torch_plugin/utils)
