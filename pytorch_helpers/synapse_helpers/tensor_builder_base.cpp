/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "synapse_helpers/tensor_builder_base.h"
#include "habana_helpers/logging.h"
#include "synapse_helpers/type_conversions.h"

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
  // write strides backwards
  // Synapse supports strides on FCD to be element size only
  if (stride.size() == 0) {
    dimensions[0] = size;
  } else if (stride[stride.size() - 1] != 1) {
    PT_SYNHELPER_WARN(
        "FCD stride for tensor is ",
        stride[stride.size() - 1],
        " Non 1 FCD is unsupported in Synapse, hence setting all strides to 0.");
    for (size_t dim_to_fill = 0; dim_to_fill < SYN_GAUDI_MAX_TENSOR_DIM - 1;
         ++dim_to_fill) {
      dimensions[dim_to_fill] = 0;
    }
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
    dimensions[0] = size * shape[num_dims - 1];
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
    }
  }

  PT_SYNHELPER_DEBUG("to_stride_t : calculated dimensions {", dimensions, "}");
  return dimensions;
}

namespace detail {

thread_local uint64_t tensor_name_generator::syn_tensor_id = 0;

std::string tensor_name_generator::get_next_tensor_name() {
  return "tensor_" + std::to_string(syn_tensor_id);
}

std::string tensor_name_generator::generate() {
  return "tensor_" + std::to_string(syn_tensor_id++);
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
