cmake_minimum_required(VERSION 3.10)

set(ABSL_PREFIX "absl_abi98")
set(ABSL_PACKAGE_NAME "abseil_abi98")
project(${ABSL_PREFIX} CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

set(CMAKE_CXX_CLANG_TIDY "")
set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "")

set(CMAKE_CXX_FLAGS -w)
set(CMAKE_CXX_STANDARD 17)

#include_directories($ENV{THIRD_PARTIES_ROOT}/abseil-cpp)
list(APPEND CMAKE_MODULE_PATH $ENV{THIRD_PARTIES_ROOT}/abseil-cpp/absl/copts)
list(APPEND CMAKE_MODULE_PATH $ENV{THIRD_PARTIES_ROOT}/abseil-cpp/CMake)

set(BUILD_TESTING OFF)
include(AbseilInstallDirs)
include(CMakePackageConfigHelpers)
include(AbseilHelpers)
list(APPEND ABSL_COMMON_INCLUDE_DIRS $ENV{THIRD_PARTIES_ROOT}/abseil-cpp)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(ABSL_USING_CLANG ON)
else()
  set(ABSL_USING_CLANG OFF)
endif()

# find dependencies
## pthread
find_package(Threads REQUIRED)

function(absl_cc_library)
  cmake_parse_arguments(ABSL_CC_LIB
    "DISABLE_INSTALL;PUBLIC;TESTONLY"
    "NAME"
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
    ${ARGN}
    )

  if(NOT ABSL_CC_LIB_TESTONLY OR ABSL_RUN_TESTS)
    get_property(ABSL_LIB_LIST GLOBAL PROPERTY ABSL_LIB_LIST)
    list(APPEND ABSL_LIB_LIST ${ABSL_PREFIX}_${ABSL_CC_LIB_NAME})
    set_property(GLOBAL PROPERTY ABSL_LIB_LIST ${ABSL_LIB_LIST})
    set(ABSL_CC_LIB_DEPS_TRANSFORMED)
    foreach(DEP ${ABSL_CC_LIB_DEPS})
      string(REPLACE "absl::" "${ABSL_PREFIX}::" DEP_TRANSFORMED ${DEP})
      list(APPEND ABSL_CC_LIB_DEPS_TRANSFORMED ${DEP_TRANSFORMED})
    endforeach()
    set(ABSL_CC_LIB_DEPS ${ABSL_CC_LIB_DEPS_TRANSFORMED})
    if(ABSL_ENABLE_INSTALL)
      message(FATAL_ERROR "ABSL_ENABLE_INSTALL is not supported by absl proxy")
    else()
      set(_NAME "${ABSL_PREFIX}_${ABSL_CC_LIB_NAME}")
    endif()

    # Check if this is a header-only library
    # Note that as of February 2019, many popular OS's (for example, Ubuntu
    # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
    # use list(FILTER...)
    set(ABSL_CC_SRCS "${ABSL_CC_LIB_SRCS}")
    foreach(src_file IN LISTS ABSL_CC_SRCS)
      if(${src_file} MATCHES ".*\\.(h|inc)")
        list(REMOVE_ITEM ABSL_CC_SRCS "${src_file}")
      endif()
    endforeach()
    if("${ABSL_CC_SRCS}" STREQUAL "")
      set(ABSL_CC_LIB_IS_INTERFACE 1)
    else()
      set(ABSL_CC_LIB_IS_INTERFACE 0)
    endif()

    if(NOT ABSL_CC_LIB_IS_INTERFACE)
      add_library(${_NAME} STATIC "")
      target_sources(${_NAME} PRIVATE ${ABSL_CC_LIB_SRCS} ${ABSL_CC_LIB_HDRS})
      target_include_directories(${_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${ABSL_COMMON_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:${ABSL_INSTALL_INCLUDEDIR}>
        )
      target_compile_options(${_NAME}
        PRIVATE ${ABSL_CC_LIB_COPTS})
      target_link_libraries(${_NAME}
        PUBLIC ${ABSL_CC_LIB_DEPS}
        PRIVATE
        ${ABSL_CC_LIB_LINKOPTS}
        ${ABSL_DEFAULT_LINKOPTS}
        )
      target_compile_definitions(${_NAME} PUBLIC ${ABSL_CC_LIB_DEFINES})

      # Add all Abseil targets to a a folder in the IDE for organization.
      if(ABSL_CC_LIB_PUBLIC)
        set_property(TARGET ${_NAME} PROPERTY FOLDER ${ABSL_IDE_FOLDER})
      elseif(ABSL_CC_LIB_TESTONLY)
        set_property(TARGET ${_NAME} PROPERTY FOLDER ${ABSL_IDE_FOLDER}/test)
      else()
        set_property(TARGET ${_NAME} PROPERTY FOLDER ${ABSL_IDE_FOLDER}/internal)
      endif()

      # INTERFACE libraries can't have the CXX_STANDARD property set
      set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${ABSL_CXX_STANDARD})
      set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

      # When being installed, we lose the absl_ prefix.  We want to put it back
      # to have properly named lib files.  This is a no-op when we are not being
      # installed.
      set_target_properties(${_NAME} PROPERTIES
        OUTPUT_NAME "${ABSL_PREFIX}_${_NAME}"
        EXCLUDE_FROM_ALL TRUE
        )
    else()
      # Generating header-only library
      add_library(${_NAME} INTERFACE)
      target_include_directories(${_NAME}
        INTERFACE
        $<BUILD_INTERFACE:${ABSL_COMMON_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:${ABSL_INSTALL_INCLUDEDIR}>
        )
      target_link_libraries(${_NAME}
        INTERFACE
        ${ABSL_CC_LIB_DEPS}
        ${ABSL_CC_LIB_LINKOPTS}
        ${ABSL_DEFAULT_LINKOPTS}
        )
      target_compile_definitions(${_NAME} INTERFACE ${ABSL_CC_LIB_DEFINES})
    endif()

    if (NOT ABSL_CC_LIB_TESTONLY AND ABSL_ENABLE_INSTALL)
      message(FATAL_ERROR "absl wrapper does not support installation")
    endif ()

    add_library(${ABSL_PREFIX}::${ABSL_CC_LIB_NAME} ALIAS ${_NAME})
    set_target_properties(${ABSL_PREFIX}_${ABSL_CC_LIB_NAME} PROPERTIES
      EXPORT_NAME ${ABSL_CC_LIB_NAME})
  endif ()
endfunction()

set_property(GLOBAL PROPERTY ABSL_LIB_LIST)
add_subdirectory($ENV{THIRD_PARTIES_ROOT}/abseil-cpp/absl ${ABSL_PREFIX})
get_property(ABSL_LIB_LIST GLOBAL PROPERTY ABSL_LIB_LIST)

# make the lib list available in other parts of the build tree
set(ABSL_LIB_LIST "${ABSL_LIB_LIST}" PARENT_SCOPE)

include(SetIncludesAsSystem)
set_includes_as_system(${ABSL_LIB_LIST})

export(TARGETS ${ABSL_LIB_LIST}
  NAMESPACE ${ABSL_PREFIX}::
  FILE "${ABSL_PACKAGE_NAME}Config.cmake")

export(PACKAGE ${ABSL_PACKAGE_NAME})
