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
    case HOST_TO_DEVICE_TENSOR:
      out << "HOST_TO_DEVICE_TENSOR";
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
    uint64_t tensor_id,
    synGraphHandle graph,
    bool is_persistent,
    bool is_external,
    shared_memory_section section,
    bool is_const,
    bool is_const_section,
    void* host_ptr,
    const uint64_t host_ptr_size,
    const uint64_t offset,
    synTensorType tensor_type,
    synapse_helpers::layouts::MemoryPermutation memory_permutation)
    : tensor_name_{tensor_name},
      tensor_id_{tensor_id},
      device_id_{device_id},
      data_type_{data_type},
      total_size_bytes_{total_size_bytes},
      shape_{shape},
      stride_{stride},
      tensor_{},
      is_persistent_{is_persistent},
      is_external_{is_external},
      memory_section_{std::move(section)},
      graph_{graph},
      is_const_{is_const},
      is_const_section_{is_const_section},
      host_ptr_{host_ptr},
      host_ptr_size_{host_ptr_size},
      offset_(offset),
      tensor_type_(tensor_type),
      permutation_(memory_permutation) {}

tensor::tensor(
    synDeviceId device_id,
    synDataType data_type,
    uint64_t total_size_bytes,
    const dynamic_shape_t& shape,
    const dynamic_shape_t& stride,
    std::string tensor_name,
    uint64_t tensor_id,
    synGraphHandle graph,
    bool is_persistent,
    bool is_external,
    shared_memory_section section,
    bool is_const,
    bool is_const_section,
    void* host_ptr,
    const uint64_t host_ptr_size,
    const uint64_t offset,
    synTensorType tensor_type,
    synapse_helpers::layouts::MemoryPermutation memory_permutation)
    : tensor_name_{tensor_name},
      tensor_id_{tensor_id},
      device_id_{device_id},
      data_type_{data_type},
      total_size_bytes_{total_size_bytes},
      shape_{shape},
      stride_{stride},
      tensor_{},
      is_persistent_{is_persistent},
      is_external_{is_external},
      memory_section_{std::move(section)},
      graph_{graph},
      is_const_{is_const},
      is_const_section_{is_const_section},
      host_ptr_{host_ptr},
      host_ptr_size_{host_ptr_size},
      offset_(offset),
      tensor_type_(tensor_type),
      permutation_(memory_permutation) {}

tensor::tensor(tensor&& other) noexcept
    : tensor_name_{other.name()},
      tensor_id_{other.id()},
      device_id_{other.device_id_},
      data_type_{other.data_type_},
      total_size_bytes_{other.total_size_bytes_},
      shape_{other.shape_},
      stride_{other.stride_},
      tensor_{other.tensor_},
      placeholder_{other.placeholder_},
      is_persistent_{other.is_persistent_},
      is_external_{other.is_external_},
      is_intermediate_shape_(other.is_intermediate_shape_),
      memory_section_{std::move(other.memory_section_)},
      graph_{other.graph_},
      is_const_{other.is_const_},
      is_const_section_{other.is_const_section_},
      host_ptr_{other.host_ptr_},
      host_ptr_size_{other.host_ptr_size_},
      offset_{other.offset_},
      tensor_type_{other.tensor_type_},
      pt_shape_{other.pt_shape_},
      permutation_(other.permutation_),
      dont_allow_permute_(other.dont_allow_permute_) {
  other.tensor_ = nullptr;
  other.memory_section_ = nullptr;
  other.graph_ = nullptr;
}

tensor& tensor::operator=(tensor&& other) noexcept {
  if (this == &other)
    return *this;
  cleanup();
  tensor_name_ = other.name();
  tensor_id_ = other.id();
  device_id_ = other.device_id_;
  data_type_ = other.data_type_;
  total_size_bytes_ = other.total_size_bytes_;
  shape_ = other.shape_;
  stride_ = other.stride_;
  tensor_ = other.tensor_;
  placeholder_ = other.placeholder_;
  is_persistent_ = other.is_persistent_;
  is_external_ = other.is_external_;
  is_intermediate_shape_ = other.is_intermediate_shape_;
  memory_section_ = std::move(other.memory_section_);
  graph_ = other.graph_;
  is_const_ = other.is_const_;
  is_const_section_ = other.is_const_section_;
  host_ptr_ = other.host_ptr_;
  host_ptr_size_ = other.host_ptr_size_;
  tensor_type_ = other.tensor_type_;
  pt_shape_ = other.pt_shape_;
  permutation_ = other.permutation_;
  dont_allow_permute_ = other.dont_allow_permute_;

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

  if (is_external_) {
    HABANA_ASSERT(is_persistent_);
    status = synTensorSetExternal(tensor_, is_external_);
    SYNAPSE_SUCCESS_CHECK_WITH_OP(
        "Failed to set tensor external.", status, cleanup());
  }

  PT_SYNHELPER_DEBUG("created ", *this);
  return {};
}

synapse_error_o tensor::set_permutation() {
  if (permutation_.size() == 0) {
    return {};
  }
  synStatus status;
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
    PT_SYNHELPER_DEBUG(
        "Calling synTensorSetPermutation on: ",
        tensor_id_,
        "  permutation size: ",
        permutation_.size(),
        " permutation: ",
        VecToString(permutation_));
    synTensorPermutation synPermutation;
    std::copy(
        permutation_.begin(), permutation_.end(), synPermutation.permutation);
    synPermutation.dims = permutation_.size();
    status = synTensorSetPermutation(tensor_, &synPermutation);
    SYNAPSE_SUCCESS_CHECK_WITH_OP(
        "synTensorSetPermutation failed.", status, cleanup());
  }
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

  if (GET_ENV_FLAG_NEW(PT_HPU_INFERENCE_MODE) && have_quantization_data_ &&
      tensor_type_ == DATA_TENSOR) {
    status = synTensorSetQuantizationData(
        tensor_,
        SYN_QUANT_DYNAMIC_RANGE,
        &dynamic_range_,
        sizeof(synQuantDynamicRange));
    SYNAPSE_SUCCESS_CHECK_WITH_OP(
        "synTensorSetQuantizationData failed.", status, cleanup());
    PT_SYNHELPER_DEBUG(
        "syn Tensor Quantization set ",
        tensor_name_,
        " Min=",
        dynamic_range_.min,
        " Max=",
        dynamic_range_.max);
  }

  synTensorGeometryExt maxGeometry;
  // Add tensor dimension via synTensorGeometryExt
  // Max geometry is also used as the actual geometry. In synapse side,
  // synGeometryMaxSizes is aliased to synGeometrySizes
  tensor_size_t maxSizes[sizeof(maxGeometry.sizes) / sizeof(tensor_size_t)] = {
      0};

  // TBD: Once GC min-max shape inferencing is available, the non_persistent
  // synapse tensors shapes need to be zero-filled.
  std::copy_n(
      shape_.max_.data(), shape_.max_.rank().value, std::begin(maxSizes));

  maxGeometry.dims = shape_.max().rank().value;
  memcpy(maxGeometry.sizes, maxSizes, sizeof(maxGeometry.sizes));
  status = synTensorSetGeometryExt(tensor_, &maxGeometry, synGeometrySizes);
  SYNAPSE_SUCCESS_CHECK_WITH_OP(
      "synTensorSetGeometryExt failed.", status, cleanup());

  if (is_host_to_device_tensor()) {
    status = synTensorSetHostPtr(
        tensor_, host_ptr_, total_size_bytes_ * 2, data_type_, false);
    SYNAPSE_SUCCESS_CHECK_WITH_OP("Set host ptr failed.", status, cleanup());
  }

  status = synTensorSetDeviceDataType(tensor_, data_type_);
  SYNAPSE_SUCCESS_CHECK_WITH_OP(
      "Set device data type failed", status, cleanup());

  if (permutation_.size()) {
    HABANA_ASSERT(
        permutation_.size() == maxGeometry.dims,
        " create tensor invalid permutation ",
        permutation_.size(),
        maxGeometry.dims);
  }
  set_permutation();

  if (is_const_) {
    HABANA_ASSERT(!is_persistent_);
    HABANA_ASSERT(tensor_type_ == DATA_TENSOR);
    status = synTensorSetHostPtr(
        tensor_, host_ptr_, host_ptr_size_, data_type_, true);
    SYNAPSE_SUCCESS_CHECK_WITH_OP(
        "synTensorSetHostPtr failed.", status, cleanup());
  } else {
    HABANA_ASSERT(!memory_section_ || (memory_section_ && is_persistent_));

    synTensorGeometryExt minGeometry;
    tensor_size_t minSizes[sizeof(minGeometry.sizes) / sizeof(tensor_size_t)] =
        {0};

    // TBD: Once GC min-max shape inferencing is available, the non_persistent
    // synapse tensors shapes need to be zero-filled.
    std::copy_n(
        shape_.min_.data(), shape_.min_.rank().value, std::begin(minSizes));

    if (is_const_section_) {
      auto err = synHostMalloc(device_id_, total_size_bytes_, 0, &host_ptr_);
      HABANA_ASSERT(err != synOutOfHostMemory);
      host_ptr_size_ = total_size_bytes_;
      status = synTensorSetHostPtr(
          tensor_, host_ptr_, total_size_bytes_, data_type_, false);
      SYNAPSE_SUCCESS_CHECK_WITH_OP(
          "synTensorSetHostPtr min sizes failed.", status, cleanup());
    }

    minGeometry.dims = shape_.min().rank().value;
    memcpy(minGeometry.sizes, minSizes, sizeof(minGeometry.sizes));
    status =
        synTensorSetGeometryExt(tensor_, &minGeometry, synGeometryMinSizes);
    SYNAPSE_SUCCESS_CHECK_WITH_OP(
        "synTensorSetGeometryExt min sizes failed.", status, cleanup());
    if (tensor_type_ == SHAPE_TENSOR ||
        tensor_type_ == INPUT_DESCRIBING_SHAPE_TENSOR) {
      HABANA_ASSERT(data_type_ == syn_type_uint32);
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
          " new mem section created with offset ",
          offset_);
      if (is_const_section_) {
        status = synSectionSetConst(*memory_section_, true);
        SYNAPSE_SUCCESS_CHECK_WITH_OP(
            "synSectionSetConst failed.", status, cleanup());
        // to do:: currently fixing section group to 1, later need to be cleaned
        status = synSectionSetGroup(*memory_section_, 1);
        SYNAPSE_SUCCESS_CHECK_WITH_OP(
            "synSectionSetGroup failed.", status, cleanup());
      }

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
  if (is_external_) {
    HABANA_ASSERT(is_persistent_);
    status = synTensorSetExternal(tensor_, is_external_);
    SYNAPSE_SUCCESS_CHECK_WITH_OP(
        "Failed to set tensor external.", status, cleanup());
  }

  PT_SYNHELPER_DEBUG("created ", *this);
  return {};
}

tensor::~tensor() {
  cleanup();
}

void tensor::cleanup() {
  if (tensor_) {
    if (is_const_section_ && host_ptr_) {
      synHostFree(device_id_, host_ptr_, 0);
      host_ptr_ = nullptr;
    }
    PT_SYNHELPER_DEBUG("cleaning ", *this);
    memory_section_ = nullptr;
    // No need to destroy tensors explicitly as those would be destroyed
    // once the graph is destroyed.
    if (!((GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) &&
          GET_ENV_FLAG_NEW(PT_HPU_LAZY_EAGER_SHAPE_AGNOSTIC_GRAPH))) {
      synDestroyTensor(tensor_);
    }
    tensor_ = nullptr;
  }
}

// Method for creating dummy tensor with sizes and strides info only
tensor tensor::create_placeholder(
    const std::vector<int64_t>& pt_shape,
    const std::vector<int64_t>& pt_stride) {
  tensor tensor{
      0,
      synDataType::syn_type_na,
      0,
      shape_t{0_D},
      shape_t{0_D},
      std::string(),
      0,
      nullptr};
  tensor.pt_shape_ = pt_shape;
  tensor.pt_strides_ = pt_stride;
  tensor.placeholder_ = true;
  return tensor;
}

tensor tensor::create_placeholder(
    synDeviceId syn_device,
    const std::vector<int64_t>& pt_shape,
    const std::vector<int64_t>& pt_stride,
    bool persistent,
    const std::string& suffix,
    synTensorType tensor_type,
    bool tensor_id_inc_flag) {
  auto tensor_id = detail::tensor_name_generator::get_tensor_id();
  auto name =
      detail::tensor_name_generator::generate(suffix, tensor_id_inc_flag);
  tensor tensor{
      syn_device,
      synDataType::syn_type_na,
      0,
      shape_t{0_D},
      shape_t{0_D},
      name,
      tensor_id,
      nullptr,
      persistent};
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
