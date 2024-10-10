###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################

# Hacky workaround because we are stuck with an ancient CMake version.
# TODO: Once we have CMake >= 3.11, revert the whole commit that introduced the hack and use FetchContent instead.
function(set_includes_as_system)
  if (${ARGC} EQUAL 0)
    message(FATAL_ERROR "set_includes_as_system called with no arguments")
  endif ()

  set(empty "")
  foreach (target ${ARGV})
    get_target_property(include_dirs ${target} INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(${target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${empty}")
    target_include_directories(${target} SYSTEM INTERFACE ${include_dirs})
  endforeach ()
endfunction()
