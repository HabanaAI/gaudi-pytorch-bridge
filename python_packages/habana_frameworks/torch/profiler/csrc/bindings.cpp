#include <torch/extension.h>

#include "pytorch_helpers/synapse_logger/synapse_logger.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  ///////////////// Host Profiler API //////////////////
  m.def("setup_profiler", []() {
    std::string cmd = "stop_data_capture";
    synapse_logger::command(cmd);
    cmd = "no_eager_flush";
    synapse_logger::command(cmd);
    cmd = "use_pid_suffix";
    synapse_logger::command(cmd);
    cmd = "optimize_trace";
    synapse_logger::command(cmd);
  });

  m.def("start_profiler", []() {
    std::string cmd = "restart";
    synapse_logger::command(cmd);
  });

  m.def("stop_profiler", []() {
    std::string cmd = "disable_mask";
    synapse_logger::command(cmd);
  });

  m.doc() = "This module registers hpu host profiler API";
}
