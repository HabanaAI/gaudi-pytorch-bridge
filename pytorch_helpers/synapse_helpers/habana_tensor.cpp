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
#include "synapse_helpers/env_flags.h"
#include "synapse_helpers/tensor_builder_base.h"

namespace synapse_helpers {

bool tensor::generate_placeholder_{false};

void tensor::shape_t::set_rank(dimension_count_t rank) noexcept {
  HABANA_ASSERT(rank.value <= HABANA_DIM_MAX);
  rank_ = rank;
}

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
  out << ", tensorType=";
  switch (syn_tensor.m_tensorType) {
    case DATA_TENSOR:
      out << "DATA_TENSOR";
      break;
    case DATA_TENSOR_DYNAMIC:
      out << "DATA_TENSOR_DYNAMIC";
      break;
    case SHAPE_TENSOR:
      out << "SHAPE_TENSOR";
      break;
    case INPUT_DESCRIBING_SHAPE_TENSOR:
      out << "INPUT_DESCRIBING_SHAPE_TENSOR";
      break;
    case DEVICE_SHAPE_TENSOR:
      out << "DEVICE_SHAPE_TENSOR";
      break;
    case TENSOR_TYPE_MAX:
    default:
      HABANA_ASSERT(false);
  }
  return out;
}

std::string tensor::shape_t::debug_string() const {
  std::string s = "[";
  for (unsigned i = 0; i < rank_.value; i++) {
    if (i > 0)
      s.append(",");
    s.append(std::to_string(dims_.at(i)));
  }
  s.append("]");
  return s;
}

tensor::dynamic_shape_t::dynamic_shape_t(shape_t min, shape_t max)
    : min_{min}, max_{max} {
  HABANA_ASSERT(min_.rank() == max_.rank());
}

tensor::tensor(
    synDeviceId device_id,
    synDataType data_type,
    uint64_t total_size_bytes,
    const shape_t shape,
    const shape_t stride,
    std::string tensor_name,
    synGraphHandle graph,
    bool is_persistent,
    shared_memory_section section,
    bool is_const,
    void* host_ptr,
    const uint64_t host_ptr_size,
    const uint64_t offset,
    synTensorType tensor_type)
    : tensor_name_{tensor_name},
      device_id_{device_id},
      data_type_{data_type},
      total_size_bytes_{total_size_bytes},
      shape_{shape},
      stride_{stride},
      tensor_{},
      is_persistent_{is_persistent},
      memory_section_{std::move(section)},
      graph_{graph},
      is_const_{is_const},
      host_ptr_{host_ptr},
      host_ptr_size_{host_ptr_size},
      offset_(offset),
      tensor_type_(tensor_type) {}

tensor::tensor(
    synDeviceId device_id,
    synDataType data_type,
    uint64_t total_size_bytes,
    const dynamic_shape_t& shape,
    const dynamic_shape_t& stride,
    std::string tensor_name,
    synGraphHandle graph,
    bool is_persistent,
    shared_memory_section section,
    bool is_const,
    void* host_ptr,
    const uint64_t host_ptr_size,
    const uint64_t offset,
    synTensorType tensor_type)
    : tensor_name_{tensor_name},
      device_id_{device_id},
      data_type_{data_type},
      total_size_bytes_{total_size_bytes},
      shape_{shape},
      stride_{stride},
      tensor_{},
      is_persistent_{is_persistent},
      memory_section_{std::move(section)},
      graph_{graph},
      is_const_{is_const},
      host_ptr_{host_ptr},
      host_ptr_size_{host_ptr_size},
      offset_(offset),
      tensor_type_(tensor_type) {}

tensor::tensor(tensor&& other) noexcept
    : tensor_name_{other.name()},
      device_id_{other.device_id_},
      data_type_{other.data_type_},
      total_size_bytes_{other.total_size_bytes_},
      shape_{other.shape_},
      stride_{other.stride_},
      tensor_{other.tensor_},
      placeholder_{other.placeholder_},
      is_persistent_{other.is_persistent_},
      memory_section_{std::move(other.memory_section_)},
      graph_{other.graph_},
      is_const_{other.is_const_},
      host_ptr_{other.host_ptr_},
      host_ptr_size_{other.host_ptr_size_},
      offset_{other.offset_},
      tensor_type_{other.tensor_type_},
      pt_shape_{other.pt_shape_} {
  other.tensor_ = nullptr;
  other.memory_section_ = nullptr;
  other.graph_ = nullptr;
}

tensor& tensor::operator=(tensor&& other) noexcept {
  if (this == &other)
    return *this;
  cleanup();
  tensor_name_ = other.name();
  device_id_ = other.device_id_;
  data_type_ = other.data_type_;
  total_size_bytes_ = other.total_size_bytes_;
  shape_ = other.shape_;
  stride_ = other.stride_;
  tensor_ = other.tensor_;
  placeholder_ = other.placeholder_;
  is_persistent_ = other.is_persistent_;
  memory_section_ = std::move(other.memory_section_);
  graph_ = other.graph_;
  is_const_ = other.is_const_;
  host_ptr_ = other.host_ptr_;
  host_ptr_size_ = other.host_ptr_size_;
  tensor_type_ = other.tensor_type_;
  pt_shape_ = other.pt_shape_;

  other.tensor_ = nullptr;
  other.memory_section_ = nullptr;
  other.graph_ = nullptr;

  return *this;
}

// [[deprecated("Use new Synapse APIs")]]
synapse_error_o tensor::create_old_synapi() {
  synStatus status;
  synTensorDescriptor trdescriptor{};

  trdescriptor.m_name = name().c_str();
  trdescriptor.m_dataType = data_type_;
  trdescriptor.m_dims = shape_.max().rank().value;
  trdescriptor.m_tensorType = tensor_type_;
  if (is_const_) {
    HABANA_ASSERT(host_ptr_);
    trdescriptor.m_isQuantized = true;
    trdescriptor.m_ptr = host_ptr_;
  }
  std::copy_n(
      shape_.max_.data(),
      shape_.max_.rank().value,
      std::begin(trdescriptor.m_sizes));

  if (is_const_) {
    HABANA_ASSERT(!is_persistent_);
    HABANA_ASSERT(tensor_type_ == DATA_TENSOR);
    status = synConstTensorCreate(&tensor_, &trdescriptor);
  } else {
    HABANA_ASSERT(!memory_section_ || (memory_section_ && is_persistent_));
    std::copy_n(
        shape_.min_.data(),
        shape_.min_.rank().value,
        std::begin(trdescriptor.m_minSizes));
    if (tensor_type_ == SHAPE_TENSOR ||
        tensor_type_ == INPUT_DESCRIBING_SHAPE_TENSOR) {
      HABANA_ASSERT(data_type_ == syn_type_uint32);
      status = synTensorCreate(&tensor_, &trdescriptor, nullptr, 0);
    } else if (!memory_section_ && is_persistent_) {
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
          "synTensorCreate ",
          *this,
          " new section created with offset ",
          offset_);
      status =
          synTensorCreate(&tensor_, &trdescriptor, *memory_section_, offset_);
    } else if (memory_section_ && is_persistent_) {
      // the only valid use case for today with user-defined memory section is
      // to do in-place update, therefore offset parameter is 0
      PT_SYNHELPER_DEBUG(
          "synTensorCreate ",
          *this,
          " existing section created with offset ",
          offset_);
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

synapse_error_o tensor::create() {
  if (GET_ENV_FLAG_NEW(PT_HPU_INTERNAL_OLD_SYNAPI)) {
    return create_old_synapi();
  }
  synStatus status;
  // Create the synTensor handle, with the given tensor type and name
  status = synTensorHandleCreate(
      &tensor_, graph_, tensor_type_, tensor_name_.c_str());
  SYNAPSE_SUCCESS_CHECK_WITH_OP(
      "synTensorHandleCreate failed.", status, cleanup());

  synTensorGeometry maxGeometry;
  // Add tensor dimension via synTensorGeometry
  // Max geometry is also used as the actual geometry. In synapse side,
  // synGeometryMaxSizes is aliased to synGeometrySizes
  uint32_t maxSizes[sizeof(maxGeometry.sizes) / sizeof(uint32_t)] = {0};

  // TBD: Once GC min-max shape inferencing is available, the non_persistent
  // synapse tensors shapes need to be zero-filled.
  std::copy_n(
      shape_.max_.data(), shape_.max_.rank().value, std::begin(maxSizes));

  maxGeometry.dims = shape_.max().rank().value;
  memcpy(maxGeometry.sizes, maxSizes, sizeof(maxGeometry.sizes));
  status = synTensorSetGeometry(tensor_, &maxGeometry, synGeometrySizes);
  SYNAPSE_SUCCESS_CHECK_WITH_OP(
      "synTensorSetGeometry failed.", status, cleanup());

  // Add strides and datatype.
  // As of now synapse supports only default strides -
  // Set the desired data type of the tensor in the device.
  // In the future, this API can also be used to set the strides of a tensors,
  // but currently only default strides are allowed.
  // If the given strides are empty (zeros) then they will be calculated
  // inside the tensor according to its geometry.
  synTensorDeviceLayout deviceLayout;
  uint32_t strides[sizeof(deviceLayout.strides) / sizeof(uint32_t)] = {0};

  if (GET_ENV_FLAG_NEW(PT_HPU_ZERO_STRIDE_SYNTENSOR)) {
    PT_SYNHELPER_DEBUG("Not passing strides to synapse, all strides will be 0");
  } else {
    std::copy_n(
        stride_.max_.data(), stride_.max_.rank().value, std::begin(strides));
  }

  memcpy(deviceLayout.strides, strides, sizeof(deviceLayout.strides));
  deviceLayout.deviceDataType = data_type_;
  if (tensor_type_ == SHAPE_TENSOR ||
      tensor_type_ == INPUT_DESCRIBING_SHAPE_TENSOR ||
      tensor_type_ == DEVICE_SHAPE_TENSOR) {
    HABANA_ASSERT(data_type_ == syn_type_uint32);
  }
  status = synTensorSetDeviceLayout(tensor_, &deviceLayout);
  SYNAPSE_SUCCESS_CHECK_WITH_OP(
      "synTensorSetDeviceLayout failed.", status, cleanup());

  if (has_dynamic_shape()) {
    synTensorGeometry minGeometry;
    uint32_t minSizes[sizeof(minGeometry.sizes) / sizeof(uint32_t)] = {0};

    // TBD: Once GC min-max shape inferencing is available, the non_persistent
    // synapse tensors shapes need to be zero-filled.
    std::copy_n(
        shape_.min_.data(), shape_.min_.rank().value, std::begin(minSizes));

    minGeometry.dims = shape_.min().rank().value;
    memcpy(minGeometry.sizes, minSizes, sizeof(minGeometry.sizes));
    status = synTensorSetGeometry(tensor_, &minGeometry, synGeometryMinSizes);
    SYNAPSE_SUCCESS_CHECK_WITH_OP(
        "synTensorSetGeometry min sizes failed.", status, cleanup());
  }

  if (is_const_) {
    HABANA_ASSERT(!is_persistent_);
    HABANA_ASSERT(tensor_type_ == DATA_TENSOR);
    status = synTensorSetHostPtr(
        tensor_, host_ptr_, host_ptr_size_, data_type_, true);
    SYNAPSE_SUCCESS_CHECK_WITH_OP(
        "synTensorSetHostPtr failed.", status, cleanup());
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
          "synTensorCreate ",
          *this,
          " new mem section created with offset ",
          offset_);
      status = synTensorAssignToSection(tensor_, *memory_section_, offset_);
      SYNAPSE_SUCCESS_CHECK_WITH_OP(
          "synTensorAssignToSection failed.", status, cleanup());
    } else if (memory_section_ && is_persistent_) {
      // the only valid use case for today with user-defined memory section is
      // to do in-place update, therefore offset parameter is 0
      PT_SYNHELPER_DEBUG(
          "synTensorCreate ",
          *this,
          " old mem section created with offset ",
          offset_);
      status = synTensorAssignToSection(tensor_, *memory_section_, offset_);
      SYNAPSE_SUCCESS_CHECK_WITH_OP(
          "synTensorAssignToSection failed.", status, cleanup());
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

tensor tensor::create_placeholder(
    synDeviceId syn_device,
    const std::vector<int64_t>& pt_shape,
    const std::vector<int64_t>& pt_stride,
    synTensorType tensor_type) {
  auto name = detail::tensor_name_generator::generate();
  tensor tensor{
      syn_device,
      synDataType::syn_type_na,
      0,
      shape_t{0_D},
      shape_t{0_D},
      name,
      nullptr};
  tensor.set_placeholder();
  tensor.pt_shape_ = pt_shape;
  tensor.pt_strides_ = pt_stride;
  tensor.tensor_type_ = tensor_type;
  return tensor;
}

uint64_t tensor::num_elements() const {
  uint64_t ret = 1;
  for (const auto& dim : shape_.max()) {
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
