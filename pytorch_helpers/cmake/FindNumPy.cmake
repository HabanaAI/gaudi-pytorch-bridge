
if(NumPy_FOUND)
  # reuse cached variables
  message(STATUS "Reuse cached information from NumPy ${NumPy_VERSION} ")
else()
  message(STATUS "Detecting NumPy using ${PYTHON_EXECUTABLE}"
    " (use -DPYTHON_EXECUTABLE=... otherwise)")
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from numpy import get_include;print(get_include());"
    OUTPUT_VARIABLE NUMPY_INFORMATION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode)

  if(NOT "${retcode}" STREQUAL "0")
    message(FATAL_ERROR "Detecting NumPy info - failed  \n Did you installed NumPy?")
  else()
    message(STATUS "Detecting NumPy info - done")
  endif()

  string(REPLACE "\n" ";" NUMPY_INFORMATION_LIST ${NUMPY_INFORMATION_STRING})
  list(GET NUMPY_INFORMATION_LIST 0 NUMPY_DETECTED_INCLUDE_DIR)

  set(NumPy_INCLUDE_DIR ${NUMPY_DETECTED_INCLUDE_DIR})
endif()

SET(NumPy_INCLUDE_DIR ${NumPy_INCLUDE_DIR} CACHE PATH "path to numpy header files")
SET(NumPy_FOUND ${NumPy_FOUND} CACHE BOOL "numpy detected flag")
mark_as_advanced(NumPy_FOUND NumPy_INCLUDE_DIR)
