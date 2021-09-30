/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <atomic>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

#include <torch/extension.h>

#include "blocking_queue.h"
#include "data_loader.h"
#include "nlohmann/json.hpp"
#include "pybind11_json.hpp"

using nlohmann::json;

namespace py = pybind11;
namespace aeondataloader = scaleoutdemoloader;

class HabanaAcceleratedPytorchDL {
 public:
  HabanaAcceleratedPytorchDL(
      py::dict dict_config,
      bool pin_memory,
      bool use_prefetch,
      bool channels_last,
      bool drop_last)
      : m_prefetchQueue(s_buffer_level) {
    std::string config_path_name = saveDictToFile(dict_config);
    m_record_count = initializeAeon(config_path_name);

    m_batch_size = m_json_config["batch_size"];
    json image_etl = getImageEtl();
    m_img_height = image_etl["height"];
    m_img_width = image_etl["width"];
    m_pin_memory = pin_memory;
    m_use_prefetch = use_prefetch;
    m_channels_last = channels_last;
    m_user_idx = 0;
    m_aeon_idx = 0;
    m_shouldStopPrefetch = false;

    if (drop_last) {
      m_total_batch_count = m_record_count / m_batch_size;
      m_last_batch_remainder = 0;
    } else {
      // Round up
      m_total_batch_count = (m_record_count + m_batch_size - 1) / m_batch_size;
      m_last_batch_remainder = m_record_count % m_batch_size;
    }
  }

  ~HabanaAcceleratedPytorchDL() {
    if (m_use_prefetch) {
      stopRunningThread();
    }

    aeondataloader::destroy_data_loader(m_loader);
  }

  std::pair<torch::Tensor, torch::Tensor> getNextTensorTuple() {
    // Stop when done
    if (++m_user_idx > m_total_batch_count)
      throw pybind11::stop_iteration();

    if (m_use_prefetch)
      return m_prefetchQueue.pop();
    else {
      return getTensorTuple(m_user_idx == m_total_batch_count);
    }
  }

  HabanaAcceleratedPytorchDL* getIter() {
    aeondataloader::data_loader_reset(m_loader);
    m_user_idx = 0;
    m_aeon_idx = 0;
    if (m_use_prefetch) {
      runPrefetchThread();
    }
    return this;
  }

  int getLength() {
    return m_total_batch_count;
  }

  uint64_t getRecordCount() {
    return m_record_count;
  }

 protected:
  void addPytorchPairToQueueThread() {
    while (++m_aeon_idx <= m_total_batch_count && !m_shouldStopPrefetch) {
      // This is a blocking API
      m_prefetchQueue.push(getTensorTuple(m_aeon_idx == m_total_batch_count));
    }
  }

  std::pair<torch::Tensor, torch::Tensor> getTensorTuple(bool is_last_batch) {
    auto image_options = torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .pinned_memory(m_pin_memory);
    auto target_options =
        torch::TensorOptions().dtype(torch::kInt32).pinned_memory(m_pin_memory);

    int step_batch_size;
    if (is_last_batch) {
      if (m_last_batch_remainder == 0) {
        step_batch_size = m_batch_size;
      } else {
        step_batch_size = m_last_batch_remainder;
      }
    } else {
      step_batch_size = m_batch_size;
    }

    auto image = torch::empty(
        {step_batch_size, m_img_height, m_img_width, 3}, image_options);
    auto target = torch::empty({step_batch_size}, target_options);

    const int image_size =
        m_img_height * m_img_width * 3 * step_batch_size * sizeof(float);
    const int target_size = step_batch_size * sizeof(uint32_t);

    char* image_data_ptr = (char*)image.data_ptr();
    char* label_data_ptr = (char*)target.data_ptr();

    // Copy data to the ptr
    aeondataloader::data_loader_get_data(
        m_loader, aeondataloader::IMAGE, image_size, image_data_ptr);
    aeondataloader::data_loader_get_data(
        m_loader, aeondataloader::LABEL, target_size, label_data_ptr);
    // get_data API does not advance iterator
    aeondataloader::data_loader_inc(m_loader);

    /* This is the format in which pytorch expects to accept the data */
    target = target.to(torch::kInt64);

    if (!m_channels_last) {
      /* Converting Image from NHWC -> NCHW */
      image = image.permute({0, 3, 1, 2});
    }

    return std::make_pair(image, target);
  }

  json getImageEtl() {
    const json& etl_json = m_json_config["etl"];
    for (auto it = etl_json.begin(); it != etl_json.end(); ++it) {
      if ((*it)["type"] == "image") {
        return *it;
      }
    }
  }

  std::string saveDictToFile(py::dict dict_config) {
    std::string config_path_name = std::tmpnam(nullptr);
    // Convert py::dict to nlohmann::json with pybind11 binding
    m_json_config = dict_config;

    std::ofstream config_out_stream(config_path_name);
    config_out_stream << m_json_config;
    return config_path_name;
  }

  uint64_t initializeAeon(const std::string& config_path_name) {
    m_loader = aeondataloader::create_data_loader();
    aeondataloader::data_loader_init(m_loader, config_path_name.c_str());
    uint64_t record_count = aeondataloader::get_database_size(m_loader);
    return record_count;
  }

  void runPrefetchThread() {
    // Always try to stop before running
    stopRunningThread();
    m_prefetchThread = std::thread(
        &HabanaAcceleratedPytorchDL::addPytorchPairToQueueThread, this);
  }

  void stopRunningThread() {
    // Indicate thread to stop
    m_shouldStopPrefetch = true;

    // Notify thread to finish last push if didn't quit yet
    m_prefetchQueue.clear();

    // Wait for thread to exit
    if (m_prefetchThread.joinable()) {
      m_prefetchThread.join();
    }

    // Remove the last push
    m_prefetchQueue.clear();

    // Thread has stopped, ready for another thread to run
    m_shouldStopPrefetch = false;
  }

 private:
  // Configuration
  json m_json_config;
  int m_batch_size;
  int m_img_height;
  int m_img_width;
  int m_total_batch_count;
  bool m_pin_memory;
  bool m_use_prefetch;
  uint64_t m_record_count;
  int m_last_batch_remainder;
  bool m_channels_last;

  // For prefetching:
  static const int s_buffer_level = 3;
  std::thread m_prefetchThread;
  BlockingQueue<std::pair<torch::Tensor, torch::Tensor>> m_prefetchQueue;
  // internal index for current index in aeon prefetching, always >= m_user_idx
  int m_aeon_idx;
  std::atomic<bool> m_shouldStopPrefetch;

  // For user-API
  int m_user_idx;

  // AEON DL
  void* m_loader;
};

PYBIND11_MODULE(habana_dl_app, m) {
  m.doc() = "pybind11 wrapper for aeon-pytorch generation";

  py::class_<HabanaAcceleratedPytorchDL>(m, "HabanaAcceleratedPytorchDL")
      .def(py::init<py::dict, bool, bool, bool, bool>())
      .def("__iter__", &HabanaAcceleratedPytorchDL::getIter)
      .def("__next__", &HabanaAcceleratedPytorchDL::getNextTensorTuple)
      .def("__len__", &HabanaAcceleratedPytorchDL::getLength)
      .def("getRecordCount", &HabanaAcceleratedPytorchDL::getRecordCount);
}
