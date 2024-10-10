/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "check_device_type.h"
#include <sys/stat.h>
#include <iostream>

bool is_simulator() {
  struct stat st = {};
  if (stat("/sys/class/accel/accel0/device/device_type", &st) == 0) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(
        "cat /sys/class/accel/accel0/device/device_type"
        " | grep -i 'sim' | wc -w",
        "r");
    if (!pipe) {
      return false;
    }
    while (!feof(pipe)) {
      if (fgets(buffer, 128, pipe) != NULL)
        result += buffer;
    }
    pclose(pipe);
    int sim_cnt;
    sscanf(result.c_str(), "%d", &sim_cnt);
    return (sim_cnt > 0);
  }
  return false;
}