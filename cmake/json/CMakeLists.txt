cmake_minimum_required(VERSION 3.10)
project(json CXX)
add_library(json INTERFACE)
target_include_directories(json INTERFACE $ENV{THIRD_PARTIES_ROOT}/json/single_include)