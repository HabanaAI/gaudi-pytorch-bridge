/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/habana_tensor.h"

#include <absl/strings/str_format.h>
#include <synapse.h>
#include <algorithm>
#include <iterator>
#include "habana_helpers/logging.h"

namespace synapse_helpers {

std::ostream& operator<<(
    std::ostream& out,
    const synTensorDescriptor& syn_tensor) {
  out << "synapse_tensor"
      << (syn_tensor.m_name ? std::string(" ") + syn_tensor.m_name
                            : "<unnamed>")
      << " at " << std::hex << syn_tensor.m_ptr << std::dec << ", dims=(";
  unsigned dim;
  for (dim = 0; dim + 1 < syn_tensor.m_dims; ++dim) {
    out << syn_tensor.m_sizes[dim] << ", ";
  }
  out << syn_tensor.m_sizes[dim] << ")";
  out << ", dtype=" << syn_tensor.m_dataType
      << ", weights=" << (syn_tensor.m_isWeights ? "T" : "F")
      << ", quantized=" << (syn_tensor.m_isQuantized ? "T" : "F");
  if (syn_tensor.m_batchPos == INVALID_BATCH_POS) {
    out << ", batchPos=INVALID";
  } else {
    out << ", batchPos=0x" << std::hex << syn_tensor.m_batchPos;
  }
  return out;
}

void tensor::shape_t::set_rank(dimension_count_t rank) noexcept {
  HABANA_ASSERT(rank.value <= SYN_MAX_TENSOR_DIM);
  rank_ = rank;
}

tensor::tensor(
    synDeviceId device_id,
    synDataType data_type,
    uint64_t total_size_bytes,
    shape_t shape,
    std::string tensor_name,
    synGraphHandle graph,
    bool is_persistent,
    shared_memory_section section,
    bool is_const,
    void* host_ptr,
    const uint64_t offset)
    : tensor_name_{std::move(tensor_name)},
      device_id_{device_id},
      data_type_{data_type},
      total_size_bytes_{total_size_bytes},
      shape_{shape},
      tensor_{},
      is_persistent_{is_persistent},
      memory_section_{std::move(section)},
      graph_{graph},
      is_const_{is_const},
      host_ptr_{host_ptr},
      offset_(offset) {}

tensor::tensor(tensor&& other) noexcept
    : tensor_name_(std::move(other.tensor_name_)),
      device_id_{other.device_id_},
      data_type_{other.data_type_},
      total_size_bytes_{other.total_size_bytes_},
      shape_{other.shape_},
      tensor_{other.tensor_},
      placeholder_{other.placeholder_},
      is_persistent_{other.is_persistent_},
      memory_section_{std::move(other.memory_section_)},
      graph_{other.graph_},
      is_const_{other.is_const_},
      host_ptr_{other.host_ptr_},
      offset_{other.offset_} {
  other.tensor_ = nullptr;
  other.memory_section_ = nullptr;
  other.graph_ = nullptr;
}

tensor& tensor::operator=(tensor&& other) noexcept {
  if (this == &other)
    return *this;
  cleanup();
  tensor_name_ = std::move(other.tensor_name_);
  device_id_ = other.device_id_;
  data_type_ = other.data_type_;
  total_size_bytes_ = other.total_size_bytes_;
  shape_ = other.shape_;
  tensor_ = other.tensor_;
  placeholder_ = other.placeholder_;
  is_persistent_ = other.is_persistent_;
  memory_section_ = std::move(other.memory_section_);
  graph_ = other.graph_;
  is_const_ = other.is_const_;
  host_ptr_ = other.host_ptr_;

  other.tensor_ = nullptr;
  other.memory_section_ = nullptr;
  other.graph_ = nullptr;

  return *this;
}

synapse_error_o tensor::create() {
  synStatus status;
  synTensorDescriptor trdescriptor{};

  PT_SYNHELPER_DEBUG("Allocate host memory handle.");
  // descriptor_.m_ptr =
  // reinterpret_cast<void*>(device_id_.get().get_next_index());
  // TODO: define create inputs function

  trdescriptor.m_name = tensor_name_.c_str();
  trdescriptor.m_dataType = data_type_;
  trdescriptor.m_dims = shape_.rank().value;
  if (is_const_) {
    HABANA_ASSERT(host_ptr_);
    trdescriptor.m_isQuantized = true;
    trdescriptor.m_ptr = host_ptr_;
  }
  std::copy_n(
      shape_.data(), shape_.rank().value, std::begin(trdescriptor.m_sizes));

  if (is_const_) {
    HABANA_ASSERT(!is_persistent_);
    status = synConstTensorCreate(&tensor_, &trdescriptor);
  } else {
    HABANA_ASSERT(!memory_section_ || (memory_section_ && is_persistent_));
    if (!memory_section_ && is_persistent_) {
      auto memory_attributes{
          synMemoryAttribute::MEMORY_ATTRIBUTE_DEVICE |
          (is_persistent_ ? synMemoryAttribute::MEMORY_ATTRIBUTE_PERSISTENT
                          : 0)};
      synSectionHandle section;
      HABANA_ASSERT(graph_ != nullptr);
      status = synSectionCreate(&section, memory_attributes, graph_);
      SYNAPSE_SUCCESS_CHECK_WITH_OP(
          "Memory section create failed.", status, cleanup());
      memory_section_ = std::make_shared<memory_section>(section);
      PT_SYNHELPER_DEBUG(
          "synTensorCreate ", *this, " created with offset ", offset_);
      status =
          synTensorCreate(&tensor_, &trdescriptor, *memory_section_, offset_);
    } else if (memory_section_ && is_persistent_) {
      // the only valid use case for today with user-defined memory section is
      // to do in-place update, therefore offset parameter is 0
      PT_SYNHELPER_DEBUG(
          "synTensorCreate ", *this, " created with offset ", offset_);
      status =
          synTensorCreate(&tensor_, &trdescriptor, *memory_section_, offset_);
    } else {
      status = synTensorCreate(&tensor_, &trdescriptor, nullptr, 0);
    }
  }

  SYNAPSE_SUCCESS_CHECK_WITH_OP("Tensor create failed.", status, cleanup());

  PT_SYNHELPER_DEBUG("created ", *this);
  return {};
}

tensor::~tensor() {
  cleanup();
}

void tensor::cleanup() {
  if (tensor_) {
    PT_SYNHELPER_DEBUG("cleaning ", *this);
    memory_section_ = nullptr;
    synDestroyTensor(tensor_);
    tensor_ = nullptr;
  }
}

tensor tensor::create_placeholder(synDeviceId syn_device) {
  static uint64_t id = -1;
  std::string name = absl::StrFormat("placeholder_tensor_%d", ++id);

  tensor tensor{
      syn_device, synDataType::syn_type_na, 0, shape_t{0_D}, name, nullptr};
  tensor.set_placeholder();

  return tensor;
}

uint64_t tensor::num_elements() const {
  uint64_t ret = 1;
  for (const auto& dim : shape_) {
    if (dim != 0) {
      ret *= dim;
    }
    if (dim == static_cast<decltype(dim)>(-1))
      return -1;
  }
  return ret;
}

memory_section::memory_section(uint64_t memory_attributes, synGraphHandle graph)
    : memory_section_{} {
  auto status = synSectionCreate(&memory_section_, memory_attributes, graph);
  if (synSuccess != status)
    PT_SYNHELPER_FATAL("Unable to create a memory section with err: ", status);
}

tensor::shape_t::dimension_count_t operator"" _D(unsigned long long arg) {
  return tensor::shape_t::dimension_count_t{static_cast<unsigned>(arg)};
}

} // namespace synapse_helpers
