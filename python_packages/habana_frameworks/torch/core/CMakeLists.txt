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

add_habana_library(_core_C SHARED bindings.cpp)

target_compile_definitions(_core_C PRIVATE TORCH_EXTENSION_NAME=_core_C)

target_link_libraries(_core_C PUBLIC bindings torch ${TORCH_PYTHON_LIBRARY})

set_target_properties(_core_C PROPERTIES PREFIX "")

install(TARGETS _core_C LIBRARY DESTINATION habana_torch_plugin)
