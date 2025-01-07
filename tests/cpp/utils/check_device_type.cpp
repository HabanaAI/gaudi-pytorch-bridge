/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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