#pragma once

#include <iostream>

#define LOG_FUNC_BEGIN \
  std::cout << "HABANA_LOG: begin of " << __PRETTY_FUNCTION__ << "\n"
#define LOG_FUNC_END \
  std::cout << "HABANA_LOG: end of " << __PRETTY_FUNCTION__ << "\n"
