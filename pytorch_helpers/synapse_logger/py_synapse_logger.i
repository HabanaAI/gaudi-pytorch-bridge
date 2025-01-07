%module py_synapse_logger

%{
#define SWIG_FILE_WITH_INIT
#include "py_synapse_logger.h"
using namespace synapse_logger;
%}

%include "std_string.i"
%include "numpy.i"

%init %{
    import_array();
%}

%apply (float* IN_ARRAY1, int DIM1) {(float* vec, int n)}

%include "py_synapse_logger.h"
