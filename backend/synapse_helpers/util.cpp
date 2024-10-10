#include <iomanip>
#include <sstream>

#include "util.h"

namespace synapse_helpers {
std::string uint64_to_hex_string(uint64_t number) {
  std::ostringstream ss;
  ss << "0x" << std::setfill('0') << std::setw(12) << std::hex << number;
  return ss.str();
}
} // namespace synapse_helpers