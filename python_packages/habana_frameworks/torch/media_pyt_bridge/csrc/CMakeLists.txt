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

add_habana_library(_media_pyt_bridge_C SHARED media_pyt_bridge.cpp pyt_media_proxy.cpp pyt_media_shim.cpp)

target_compile_definitions(_media_pyt_bridge_C PRIVATE TORCH_EXTENSION_NAME=_media_pyt_bridge_C)

set_target_properties(_media_pyt_bridge_C PROPERTIES PREFIX "")
target_link_libraries(_media_pyt_bridge_C PUBLIC bindings torch absl::variant  ${TORCH_PYTHON_LIBRARY} Media)

install(TARGETS _media_pyt_bridge_C LIBRARY DESTINATION habana_torch_plugin)
