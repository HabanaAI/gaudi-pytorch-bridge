/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include <atomic>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <torch/extension.h>

#include "blocking_queue.h"
#include "data_loader.h"
#include "nlohmann/json.hpp"
#include "pybind11_json.hpp"

using nlohmannV340::json;

namespace py = pybind11;
namespace aeondataloader = scaleoutdemoloader;

json getJsonConfig(const json& cfg, std::string key) {
  const json& etl_json = cfg["etl"];
  for (auto it = etl_json.begin(); it != etl_json.end(); ++it) {
    if ((*it)["type"] == key) {
      return *it;
    }
  }
  std::stringstream ss;
  ss << "missing json config: " << key;
  throw std::runtime_error(ss.str());
}

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
    json image_etl = getJsonConfig(m_json_config, "image");
    m_img_height = image_etl["height"];
    m_img_width = image_etl["width"];
    m_pin_memory = pin_memory;
    m_use_prefetch = use_prefetch;
    m_channels_last = channels_last;
    m_user_idx = 0;
    m_shouldStopPrefetch = false;
    m_aeon_permute = image_etl.value("aeon_permute", true);

    if (drop_last) {
      m_total_batch_count = m_record_count / m_batch_size;
      m_last_batch_remainder = 0;
    } else {
      // Round up
      m_total_batch_count = (m_record_count + m_batch_size - 1) / m_batch_size;
      m_last_batch_remainder = m_record_count % m_batch_size;
    }
  }

  virtual ~HabanaAcceleratedPytorchDL() {
    if (m_use_prefetch) {
      stopRunningThread();
    }

    aeondataloader::destroy_data_loader(m_loader);
  }

  std::vector<torch::Tensor> getNextTensorTuple() {
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
    int aeon_idx = 0;
    while (++aeon_idx <= m_total_batch_count && !m_shouldStopPrefetch) {
      // This is a blocking API
      m_prefetchQueue.push(getTensorTuple(aeon_idx == m_total_batch_count));
    }
  }

  int get_step_batch_size(bool is_last_batch) {
    if (is_last_batch) {
      if (m_last_batch_remainder == 0) {
        return m_batch_size;
      } else {
        return m_last_batch_remainder;
      }
    }
    return m_batch_size;
  }

  at::Tensor make_image_tensor(at::TensorOptions image_options) {
    if (!m_channels_last && m_aeon_permute) {
      return torch::empty(
          {m_batch_size, 3, m_img_height, m_img_width},
          image_options,
          {torch::MemoryFormat::Contiguous});
    } else {
      return torch::empty(
          {m_batch_size, m_img_height, m_img_width, 3},
          image_options,
          {torch::MemoryFormat::Contiguous});
    }
  }

  void maybe_permute(at::Tensor& t) {
    if (!m_channels_last && !m_aeon_permute) {
      /* Converting Image from NHWC -> NCHW */
      t = t.permute({0, 3, 1, 2}).contiguous();
    }
  }

  virtual std::vector<torch::Tensor> getTensorTuple(bool is_last_batch) {
    auto image_options = torch::TensorOptions().dtype(torch::kFloat32);
    auto target_options = torch::TensorOptions().dtype(torch::kInt32);

    int step_batch_size = get_step_batch_size(is_last_batch);

    auto image = make_image_tensor(image_options);
    if (m_pin_memory) {
      image = at::native::pin_memory(image, torch::kHPU);
    }
    auto target = torch::empty({m_batch_size}, target_options);
    if (m_pin_memory) {
      target = at::native::pin_memory(target, torch::kHPU);
    }

    const int image_size =
        m_img_height * m_img_width * 3 * m_batch_size * sizeof(float);
    const int target_size = m_batch_size * sizeof(uint32_t);

    char* image_data_ptr = (char*)image.data_ptr();
    char* label_data_ptr = (char*)target.data_ptr();

    // Copy data to the ptr
    aeondataloader::data_loader_get_data(
        m_loader, aeondataloader::IMAGE, image_size, image_data_ptr);
    aeondataloader::data_loader_get_data(
        m_loader, aeondataloader::LABEL, target_size, label_data_ptr);
    // get_data API does not advance iterator
    aeondataloader::data_loader_inc(m_loader);

    // Workaround for bad batch size
    if (step_batch_size < m_batch_size) {
      image = image.narrow(0, 0, step_batch_size);
      target = target.narrow(0, 0, step_batch_size);
    }

    /* This is the format in which pytorch expects to accept the data */
    target = target.to(torch::kInt64);

    maybe_permute(image);

    return make_vec(image, target);
  }

  std::string saveDictToFile(py::dict dict_config) {
    char tmp_fname[] = "/tmp/dl_dict_XXXXXX";
    int fd = mkstemp(tmp_fname);
    if (fd == -1) {
      throw std::system_error(errno, std::system_category());
    }
    close(fd);
    std::string config_path_name = std::string(tmp_fname);

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

  template <typename... T>
  std::vector<torch::Tensor> make_vec(T&... t) {
    return {std::move(t)...};
  }

 protected:
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
  bool m_aeon_permute;

  // For prefetching:
  static const int s_buffer_level = 3;
  std::thread m_prefetchThread;
  BlockingQueue<std::vector<torch::Tensor>> m_prefetchQueue;
  // internal index for current index in aeon prefetching, always >= m_user_idx
  std::atomic<bool> m_shouldStopPrefetch;

  // For user-API
  int m_user_idx;

  // AEON DL
  void* m_loader;
};

class SsdHDL : public HabanaAcceleratedPytorchDL {
 public:
  SsdHDL(
      py::dict dict_config,
      bool pin_memory,
      bool use_prefetch,
      bool channels_last,
      bool drop_last)
      : HabanaAcceleratedPytorchDL(
            dict_config,
            pin_memory,
            use_prefetch,
            channels_last,
            drop_last) {
    m_max_gt_boxes =
        getJsonConfig(m_json_config, "localization_ssd")["max_gt_boxes"];
  }

  std::vector<torch::Tensor> getTensorTuple(bool is_last_batch) override {
    auto image_options = torch::TensorOptions().dtype(torch::kFloat32);
    auto img_id_options = torch::TensorOptions().dtype(torch::kInt32);
    auto img_size_options = torch::TensorOptions().dtype(torch::kInt32);
    auto bbox_options = torch::TensorOptions().dtype(torch::kFloat32);
    auto label_options = torch::TensorOptions().dtype(torch::kInt32);

    int step_batch_size = get_step_batch_size(is_last_batch);

    auto image = make_image_tensor(image_options);
    auto bbox = torch::empty({m_batch_size, m_max_gt_boxes, 4}, bbox_options);
    auto label = torch::empty({m_batch_size, m_max_gt_boxes}, label_options);
    auto img_id = torch::empty({m_batch_size}, img_id_options);
    auto img_shape = torch::empty({m_batch_size, 2}, img_size_options);
    if (m_pin_memory) {
      label = at::native::pin_memory(label, torch::kHPU);
      image = at::native::pin_memory(image, torch::kHPU);
      bbox = at::native::pin_memory(bbox, torch::kHPU);
      img_id = at::native::pin_memory(img_id, torch::kHPU);
      img_shape = at::native::pin_memory(img_shape, torch::kHPU);
    }

    const int image_size =
        m_img_height * m_img_width * 3 * m_batch_size * sizeof(float);
    char* image_data_ptr = (char*)image.data_ptr();

    const int bbox_size = m_batch_size * m_max_gt_boxes * 4 * sizeof(float);
    char* bbox_ptr = (char*)bbox.data_ptr();
    const int label_size = m_batch_size * m_max_gt_boxes * sizeof(uint32_t);
    const int img_id_size = m_batch_size * sizeof(uint32_t);
    char* label_ptr = (char*)label.data_ptr();
    char* img_id_ptr = (char*)img_id.data_ptr();
    const int img_shape_size = 2 * m_batch_size * sizeof(uint32_t);
    char* img_shape_ptr = (char*)img_shape.data_ptr();

    // // Copy data to the ptr
    aeondataloader::data_loader_get_data(
        m_loader, aeondataloader::IMAGE, image_size, image_data_ptr);
    aeondataloader::data_loader_get_data(
        m_loader, aeondataloader::BBOX_LABEL, label_size, label_ptr);
    aeondataloader::data_loader_get_data(
        m_loader, aeondataloader::BBOX, bbox_size, bbox_ptr);
    aeondataloader::data_loader_get_data(
        m_loader, aeondataloader::IMAGE_SHAPE, img_shape_size, img_shape_ptr);
    aeondataloader::data_loader_get_data(
        m_loader, aeondataloader::IMAGE_ID, img_id_size, img_id_ptr);
    // get_data API does not advance iterator
    aeondataloader::data_loader_inc(m_loader);

    // Workaround for bad batch size
    if (step_batch_size < m_batch_size) {
      image = image.narrow(0, 0, step_batch_size);
      bbox = bbox.narrow(0, 0, step_batch_size);
      label = label.narrow(0, 0, step_batch_size);
      img_id = img_id.narrow(0, 0, step_batch_size);
      img_shape = img_shape.narrow(0, 0, step_batch_size);
    }

    maybe_permute(image);

    /* This is the format in which pytorch expects to accept the data */
    label = label.to(torch::kInt64);

    return make_vec(image, img_id, img_shape, bbox, label);
  }

 private:
  int64_t m_max_gt_boxes;
};

class Factory {
 public:
  static std::unique_ptr<HabanaAcceleratedPytorchDL> create(
      py::dict dict_config,
      bool pin_memory,
      bool use_prefetch,
      bool channels_last,
      bool drop_last) {
    json config = dict_config;
    if (is_ssd_config(config)) {
      return std::make_unique<SsdHDL>(
          dict_config, pin_memory, use_prefetch, channels_last, drop_last);
    } else {
      return std::make_unique<HabanaAcceleratedPytorchDL>(
          dict_config, pin_memory, use_prefetch, channels_last, drop_last);
    }
  }
  static bool is_ssd_config(json config) {
    try {
      if (getJsonConfig(config, "localization_ssd").empty()) {
        return false;
      }
    } catch (...) {
      return false;
    }
    return true;
  }
};

PYBIND11_MODULE(habana_dl_app, m) {
  m.doc() = "pybind11 wrapper for aeon-pytorch generation";

  py::class_<HabanaAcceleratedPytorchDL>(m, "HabanaAcceleratedPytorchDL")
      .def_static(
          "create",
          [](py::dict dict_config,
             bool pin_memory,
             bool use_prefetch,
             bool channels_last,
             bool drop_last) {
            return Factory::create(
                dict_config,
                pin_memory,
                use_prefetch,
                channels_last,
                drop_last);
          },
          py::return_value_policy::move)
      .def(py::init<py::dict, bool, bool, bool, bool>())
      .def("__iter__", &HabanaAcceleratedPytorchDL::getIter)
      .def("__next__", &HabanaAcceleratedPytorchDL::getNextTensorTuple)
      .def("__len__", &HabanaAcceleratedPytorchDL::getLength)
      .def("getRecordCount", &HabanaAcceleratedPytorchDL::getRecordCount);
}
