if (NOT DEFINED ENV{THIRD_PARTIES_ROOT})
    message(FATAL_ERROR "THIRD_PARTIES_ROOT env var doesn't exist")
endif()

# This is an external library that should be removed from the bridge to another repository, see here: SW-73476
add_library(tensor_comparator_core STATIC
                    tensor_comparator.cpp
                    serialization.cpp
                    validator.cpp
                    comparison_methods.cpp
                    ../../synapse/src/infra/float16.cpp
                    ../../synapse/src/infra/sim_fp16.cpp
                    ../../include/comparator.hpp
                    ../../include/comparison_methods.hpp
                    ../../include/results_warehouse.hpp
                    ../../include/serialization.hpp
                    ../../include/tensor_comparator.hpp
                    ../../include/threadpool.hpp
                    ../../include/types.hpp
                    ../../include/validator.hpp
)

target_include_directories(tensor_comparator_core
    PRIVATE ../../include/
    PUBLIC  ../../include/
)

include_directories(../../synapse/src/infra
                    $ENV{THIRD_PARTIES_ROOT}/json/single_include
                    ../../include
                    ../src/tensor_comparator
                    $ENV{THIRD_PARTIES_ROOT}/eigen
)

set_target_properties(tensor_comparator_core
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

set(CMAKE_CXX_FLAGS                " ${CMAKE_CXX_FLAGS_INIT} -fPIC")
set(CMAKE_CXX_STANDARD 17)