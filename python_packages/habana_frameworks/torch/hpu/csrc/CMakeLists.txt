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

add_habana_library(_hpu_C SHARED bindings.cpp object_ptr.cpp Exceptions.cpp Stream.cpp Event.cpp Module.cpp)
target_compile_definitions(_hpu_C PRIVATE TORCH_EXTENSION_NAME=_hpu_C)

set_target_properties(_hpu_C PROPERTIES PREFIX "")
target_link_libraries(_hpu_C PUBLIC bindings torch absl::variant absl::flat_hash_map absl::strings
                                    ${TORCH_PYTHON_LIBRARY})
install(TARGETS _hpu_C LIBRARY DESTINATION habana_torch_plugin)
