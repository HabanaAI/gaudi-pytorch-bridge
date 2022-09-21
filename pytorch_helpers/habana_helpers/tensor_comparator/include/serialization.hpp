#pragma once
#include <types.hpp>
#include <map>
#include <string>

namespace TensorComparison_pt {
namespace jsn {
ThresholdFileMap parse(const std::string& fileName);
template <typename T>
bool dump(T& t, const std::string& fileName);
} // namespace jsn

namespace csv {
bool dump(ResultMap& m, const std::string& fileName);
bool dump(ResultVec& m, const std::string& fileName);
} // namespace csv
} // namespace TensorComparison_pt