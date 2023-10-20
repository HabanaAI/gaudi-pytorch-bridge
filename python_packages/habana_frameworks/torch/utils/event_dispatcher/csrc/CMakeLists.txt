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

add_habana_library(_event_dispatcher_C SHARED bindings.cpp)
target_compile_definitions(_event_dispatcher_C PRIVATE TORCH_EXTENSION_NAME=_event_dispatcher_C)
set_target_properties(_event_dispatcher_C PROPERTIES PREFIX "")
target_link_libraries(_event_dispatcher_C PUBLIC bindings torch absl::strings ${TORCH_PYTHON_LIBRARY})
install(TARGETS _event_dispatcher_C LIBRARY DESTINATION habana_torch_plugin/utils)
