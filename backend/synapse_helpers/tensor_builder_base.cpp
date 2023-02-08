/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "backend/synapse_helpers/tensor_builder_base.h"
#include "backend/synapse_helpers/type_conversions.h"
#include "habana_helpers/logging.h"

#include <string>

namespace synapse_helpers {

tensor::shape_t to_shape_t(const std::vector<int64_t>& shape, bool reverse) {
  auto shape_size = shape.size() > 0 ? shape.size() : 1;
  tensor::shape_t dimensions{tensor::shape_t::dimension_count_t{
      static_cast<unsigned>(shape_size)}}; // TODO make it more readable
  if (shape.size() == 0) {
    PT_SYNHELPER_DEBUG("to_shape_t: Converting 0D to 1D with {1} shape");
    dimensions[0] = 1;
  } else {
    // write dimension backwards, e.g. NHWC as CWHN
    for (size_t i = 0; i < shape.size(); ++i) {
      dimensions[i] = reverse ? shape[shape.size() - i - 1] : shape[i];
    }
  }
  return dimensions;
}

// Stride calculation routine specific for shape tensors
tensor::shape_t to_shape_tensor_stride_t(const int64_t& stride_rank) {
  auto stride_size = stride_rank > 0 ? stride_rank : 1;
  tensor::shape_t dimensions{
      tensor::shape_t::dimension_count_t{static_cast<unsigned>(stride_size)}};
  // For shape tensors GC mandates strides to be 0
  for (uint i = 0; i < stride_size; i++) {
    dimensions[i] = 0;
  }
  return dimensions;
}

tensor::shape_t to_stride_t(
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& shape,
    synDataType data_type,
    bool reverse) {
  HABANA_ASSERT(reverse);
  auto stride_size = stride.size() > 0 ? stride.size() : 1;
  tensor::shape_t dimensions{tensor::shape_t::dimension_count_t{
      static_cast<unsigned>(stride_size)}}; // TODO make it more readable
  auto size = size_of_syn_data_type(data_type);

  PT_SYNHELPER_DEBUG("to_stride_t : tensor element size = ", size);
  if (IS_SYNHELPER_DEBUG_ENABLED) {
    std::string str = "to_stride_t : tensor shape {";
    for (const auto& s : shape) {
      str += std::to_string(s) + ", ";
    }

    str += "}";
    PT_SYNHELPER_DEBUG(str);

    str = "to_stride_t : tensor stride {";
    for (const auto& s : stride) {
      str += std::to_string(s) + ", ";
    }
    str += "}";
    PT_SYNHELPER_DEBUG(str);
  }

  auto tensor_size = size;
  for (const auto& s : shape) {
    tensor_size *= s;
  }

  // write strides backwards
  // Synapse supports strides on FCD to be element size only
  if (stride.size() == 0) {
    dimensions[0] = size;
  } else if (stride[stride.size() - 1] != 1) {
    PT_SYNHELPER_FATAL(
        "FCD stride for tensor is ",
        stride[stride.size() - 1],
        ". FCD stride > 1 is not supported in the PT HPU bridge.");
    // asserting explicitly incase the user turned off fatal logging
    HABANA_ASSERT(false);
  } else {
    // The way to add strides to synapse tensor is described below -
    //
    // GC description
    // ==============
    // FCD is not included. It is assumed Sizeof(element)
    // Each stride, is the amount of bytes to jump from element to
    //  element on the next dimension.
    // FCD is the channels in synapse semantics.
    // The last stride is the total tensor size
    // Example for a trivial stride on a 2X3 float tensor:
    // Sizes : [2,3] Strides: [8,24]
    // If we want to skip an element each time we move on the second dimesion
    //  (since strides on FCD are not supported), the strides would be: [16,
    //  48]. This time we will jump 16 bytes between elements [0,1] , [0,2],
    //  [0,3]
    // The sizes of the tensor will remain [2,3] However its section will now
    //  require 48 bytes in stead of 24.

    const auto num_dims = stride.size();
    // For an 1D tensor, synapse wants the stride to be size * shape(dim(0)).
    // For example, a float32 tensor of shape[3] will have stride {3*4} = {12}
    // The shape(dim(0)) is retrived from the PT tensor shape
    // Use stride in case it is higher than the shape (indicates a non-deafult
    // stride)
    auto first_dim_value = shape[num_dims - 1];
    if (num_dims > 1) {
      if (stride[num_dims - 2] > first_dim_value) {
        first_dim_value = stride[num_dims - 2];
      }
    }
    dimensions[0] = size * first_dim_value;
    if (num_dims > 1) {
      // First dim stride for synapse tensor is already set above. The last
      // dim stride will be the entire tensor size. Fill up the synapse
      // strides from second to last but one. The PT strides are looked up in
      // reverse.
      for (size_t dim_to_fill = 1; dim_to_fill < num_dims - 1; ++dim_to_fill) {
        const auto pt_stride_reverse_dim = num_dims - 1 - dim_to_fill;
        // Pick up the PT stride for the previous dim in rever direction.
        // Example, a float32 PT tensor of shape[3, 4, 5] and stride (20, 5,
        // 1) Synapse tensor shape is {5, 4, 3} (reversed) and strides should
        // be -
        //        PT stride         Reverse PT stride         Synapse stride
        //           20                      1                ______ 4*5
        //            5                      5 ______________| _____ 4*20
        //            1                     20 _______________|      4*60
        //  That is - {4*5, 4*20, 4*60}
        dimensions[dim_to_fill] = stride[pt_stride_reverse_dim - 1] * size;
      }
      // Fill up last dim stride
      dimensions[num_dims - 1] = shape[0] * dimensions[num_dims - 2];

      // Ensure that the last dim stride doesn't exceed the tensor size
      if (dimensions[num_dims - 1] > tensor_size) {
        dimensions[num_dims - 1] = tensor_size;
      }
    }
  }

  PT_SYNHELPER_DEBUG("to_stride_t : calculated dimensions {", dimensions, "}");
  return dimensions;
}

namespace detail {

thread_local uint64_t tensor_name_generator::syn_tensor_id = 0;

std::string tensor_name_generator::get_next_tensor_name(
    const std::string& suffix) {
  std::string tensor_name = std::to_string(syn_tensor_id);
  if (IS_SYNHELPER_DEBUG_ENABLED || GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2) {
    tensor_name = "tensor_" + tensor_name;
    if (!suffix.empty()) {
      tensor_name.append("_" + suffix);
    }
  }
  return tensor_name;
}

std::string tensor_name_generator::generate(
    const std::string& suffix,
    bool tensor_id_inc_flag) {
  std::string tensor_name = get_next_tensor_name(suffix);
  if (tensor_id_inc_flag) {
    syn_tensor_id++;
  }
  if (IS_SYNHELPER_DEBUG_ENABLED || GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 2) {
    to_netron_syntax(tensor_name);
  }
  return tensor_name;
}

// netron app doesn't handle '::' in nodes and tensors
// changing them to '/' for display purposes
void tensor_name_generator::to_netron_syntax(std::string& name) {
  std::vector<std::string> symbols_to_replace = {"::", "."};
  for (auto& s : symbols_to_replace) {
    size_t pos = 0;
    while ((pos = name.find(s, pos)) != std::string::npos) {
      name.replace(pos, s.size(), "/");
      pos += s.length();
    }
  }
}

void tensor_name_generator::set_tensor_id(uint64_t id) {
  syn_tensor_id = id;
}

uint64_t tensor_name_generator::get_tensor_id() {
  return syn_tensor_id;
}

void tensor_name_generator::reset() {
  syn_tensor_id = 0;
}

uint64_t size_bytes_from_shape(
    const tensor::shape_t& shape,
    synDataType dataType) {
  HABANA_ASSERT(shape.rank().value <= HABANA_DIM_MAX);
  HABANA_ASSERT(shape.rank().value > 0);
  uint64_t size = size_of_syn_data_type(dataType);
  for (auto i{0U}; i < shape.rank().value; ++i) {
    size *= shape[i];
  }
  return size;
}

} // namespace detail
} // namespace synapse_helpers
