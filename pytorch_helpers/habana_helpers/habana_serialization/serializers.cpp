/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "serializers.h"

namespace serialization {

void serialize(std::ostream& os, const char* input) {
  // assumes that input is not nullptr
  size_t size = strlen(input) + 1;
  os.write(reinterpret_cast<char const*>(&size), sizeof(size));
  os.write(input, size);
}

void serialize(std::ostream& os, std::string const& input) {
  serialize(os, static_cast<int>(input.size()));
  os.write(input.data(), input.size());
}

// PT part
void serialize(std::ostream& os, c10::Device const& input) {
  c10::DeviceType dType = input.type();
  os.write(reinterpret_cast<char const*>(&dType), sizeof(c10::DeviceType));
  c10::DeviceIndex dIndex = input.index();
  os.write(reinterpret_cast<char const*>(&dIndex), sizeof(c10::DeviceIndex));
}

void serialize(std::ostream& os, caffe2::TypeMeta input) {
  c10::ScalarType scalarType = input.toScalarType();
  os.write(reinterpret_cast<char const*>(&scalarType), sizeof(c10::ScalarType));
}

#define SERIALIZE_OPT(TYPE)                    \
  serialize(os, input.has_##TYPE());           \
  if (input.has_##TYPE()) {                    \
    serialize(os, input.TYPE##_opt().value()); \
  }

void serialize(std::ostream& os, c10::TensorOptions const& input) {
  SERIALIZE_OPT(device)
  SERIALIZE_OPT(dtype)
  SERIALIZE_OPT(layout)
  SERIALIZE_OPT(requires_grad)
  SERIALIZE_OPT(pinned_memory)
  SERIALIZE_OPT(memory_format)
}
} // namespace serialization
