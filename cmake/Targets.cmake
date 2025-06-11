###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################

include(CheckCXXCompilerFlag)

function(find_keyword keyword result_var)
  set(${result_var}
      FALSE
      PARENT_SCOPE)

  foreach(arg IN LISTS ARGN)
    if(arg STREQUAL ${keyword})
      set(${result_var}
          TRUE
          PARENT_SCOPE)
      break()
    endif()
  endforeach()
endfunction()

function(set_up_warnings target_name)
  # TODO: Add -Wconversion
  target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wno-error=deprecated-declarations -Wimplicit-fallthrough
                                                -Wformat -Wformat-security)

  check_cxx_compiler_flag("-Werror=template-id-cdtor" HAS_WERROR_TEMPLATE_ID_CTOR)

  if(HAS_WERROR_TEMPLATE_ID_CTOR)
    # GCC 14.2 emits C++20 related error even in C++17 mode when -Wall is set.
    # As a W/A don't emit error in this case
    target_compile_options(${target_name} PRIVATE -Wno-error=template-id-cdtor)
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0.0")
    # According to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=80635 GCC older than 11 may trigger
    # bugous maybe-uninitialized warning for std::optional destructor. And this was observed on d10
    # build with gcc8.3.0.
    # As a W/A don't emit error in this case.
    target_compile_options(${target_name} PRIVATE -Wno-error=maybe-uninitialized)
  endif()

  if(PROJECT_IS_TOP_LEVEL)
    target_compile_options(${target_name} PRIVATE -Werror -Werror=format-security)
  endif()
endfunction()

function(set_up_hardening target_name)
  target_compile_options(${target_name} PRIVATE -fcf-protection=full) # SDL requirement
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsanitize=cfi" HAS_FSANITIZE_CFI) # clang only
  if(HAS_FSANITIZE_CFI)
    # SDL requirement
    target_compile_options(${target_name} PRIVATE -fsanitize=cfi)
    target_link_options(${target_name} PRIVATE -fsanitize=cfi)
  endif()
  target_compile_options(${target_name} PRIVATE -fPIE -fPIC) # SDL requirement
  target_compile_options(${target_name} PRIVATE -fstack-protector-strong -fstack-clash-protection) # SDL requirement
  target_link_options(${target_name} PRIVATE -fstack-protector-strong -fstack-clash-protection)
endfunction()

function(attach_sanitizers_if_requested target_name)
  if(SANITIZER)
    target_compile_options(${target_name} PRIVATE -fsanitize=address -fsanitize=undefined -fno-sanitize=vptr
                                                  -fsanitize-address-use-after-scope -Og)
    target_link_options(${target_name} PRIVATE -fsanitize=address -fsanitize=leak -fsanitize=undefined)
  endif()

  if(THREAD_SANITIZER)
    target_compile_options(${target_name} PRIVATE -O0 -g3 -fsanitize=thread)
  endif()
endfunction()

function(allow_code_coverage_if_requested target_name)
  if(CODE_COVERAGE)
    target_compile_options(${target_name} PRIVATE --coverage -O0)
    target_link_libraries(${target_name} PRIVATE --coverage)
  endif()
endfunction()

function(set_up_link_options target_name)
  # Enable Immediate Binding mode as required by SDL
  target_link_options(${target_name} PRIVATE -Wl,-z,now)
  # Enable Inexecutable Stack as required by SDL
  target_link_options(${target_name} PRIVATE -Wl,-z,noexecstack)
  # Enable Read-Only Relocation as required by SDL
  target_link_options(${target_name} PRIVATE -Wl,-z,relro)
  # Enable Position Independent Execution as required by SDL
  target_link_options(${target_name} PRIVATE -pie)
endfunction()

function(add_habana_library target_name)
  add_library(${target_name} ${ARGN})
  add_library(npu::${target_name} ALIAS ${target_name})

  find_keyword(INTERFACE IS_INTERFACE ${ARGN})

  if(NOT IS_INTERFACE)
    set_up_warnings(${target_name})
    set_up_hardening(${target_name})
    allow_code_coverage_if_requested(${target_name})
    attach_sanitizers_if_requested(${target_name})
    set_up_link_options(${target_name})
  endif()

endfunction()

function(add_habana_executable target_name)
  add_executable(${target_name} ${ARGN})
  add_executable(npu::${target_name} ALIAS ${target_name})

  find_keyword(INTERFACE IS_INTERFACE ${ARGN})

  if(NOT IS_INTERFACE)
    set_up_warnings(${target_name})
    set_up_hardening(${target_name})
    attach_sanitizers_if_requested(${target_name})
    allow_code_coverage_if_requested(${target_name})
    set_up_link_options(${target_name})
  endif()
endfunction()

if(SANITIZER)
  message("Building sanitizers configuration")
endif()

if(THREAD_SANITIZER)
  message("Building thread sanitizer configuration")
endif()
