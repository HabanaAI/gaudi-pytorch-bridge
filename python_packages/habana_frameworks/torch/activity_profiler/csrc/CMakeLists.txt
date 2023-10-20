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
add_habana_library(_activity_profiler_C SHARED
    bindings.cpp)
target_compile_definitions(_activity_profiler_C PRIVATE TORCH_EXTENSION_NAME=_activity_profiler_C)

set_target_properties(_activity_profiler_C PROPERTIES PREFIX "")
target_link_libraries(_activity_profiler_C PRIVATE bindings torch ${TORCH_PYTHON_LIBRARY})
install(TARGETS _activity_profiler_C LIBRARY DESTINATION habana_torch_plugin/utils)
