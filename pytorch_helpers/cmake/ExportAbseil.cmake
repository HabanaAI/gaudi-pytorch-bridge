function(find_absl_targets DIRECTORY)
  get_property(
    ABSL_TARGETS_IN_DIRECTORY
    DIRECTORY "${DIRECTORY}"
    PROPERTY BUILDSYSTEM_TARGETS)
  list(APPEND ABSL_TARGETS ${ABSL_TARGETS_IN_DIRECTORY})

  get_property(
    SUBDIRECTORIES
    DIRECTORY "${DIRECTORY}"
    PROPERTY SUBDIRECTORIES)
  foreach(SUBDIRECTORY IN LISTS SUBDIRECTORIES)
    find_absl_targets("${SUBDIRECTORY}")
  endforeach()

  return(PROPAGATE ABSL_TARGETS)
endfunction()

find_absl_targets($ENV{THIRD_PARTIES_ROOT}/abseil-cpp)

export(
  TARGETS ${ABSL_TARGETS}
  NAMESPACE absl::
  FILE abseilConfig.cmake)

export(PACKAGE abseil)
