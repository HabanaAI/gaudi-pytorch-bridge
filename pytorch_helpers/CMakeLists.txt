###############################################################################
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
################################################################################
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

add_subdirectory(synapse_helpers)

find_package(PythonLibs REQUIRED)
find_package(NumPy)
find_program(SWIG swig)
add_subdirectory(synapse_logger)
