cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(tensor_comparator)

set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0 -fPIC" CACHE STRING "debug flags" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -fPIC" CACHE STRING "release flags" FORCE)
set(CMAKE_CXX_CLANG_TIDY "")

if(NOT DEFINED ENV{THIRD_PARTIES_ROOT})
    message(FATAL_ERROR "THIRD_PARTIES_ROOT env var doesn't exist")
endif()

add_subdirectory(src)
