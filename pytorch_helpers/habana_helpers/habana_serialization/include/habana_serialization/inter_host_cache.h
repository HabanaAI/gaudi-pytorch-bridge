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
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include "cache_file_handler.h"

namespace serialization {

// This class is for inter-host Recipe/Metadata transfer
// Rank-0 is the server
// All other Ranks will "send_file" to Rank-0 and "recv_file" from Rank-0
// This feature is disabled if there are any errors during setup
class InterHostCache {
  int sockfd;
  int master_port;
  int rank, w_size, l_w_size;
  bool is_cache_valid_;
  std::string cache_path_;
  std::string master_addr;
  std::shared_ptr<CacheFileHandler> cfHandler_;
  std::mutex mutex_;

  // For simplicity, keep Commands length same
  const std::string cmdSet{"SET\0"};
  const std::string cmdGet{"GET\0"};
  const std::string cmdEnd{"END\0"};
  const std::string cmdAck{"ACK\0"};
  const std::string cmdRej{"REJ\0"};

  static const int MAX_SIZE = 2048;
  char data[MAX_SIZE];

  // Rank-0 will launch this function for each clients
  void thread_function(int clientfd);
  bool _send_file(std::string filename, int sock, char* buff);
  bool _recv_file(std::string filename, int sock, char* buff);
  void invalidate_cache() {
    is_cache_valid_ = false;
  }

 public:
  InterHostCache(
      std::string& cache_path,
      std::shared_ptr<CacheFileHandler> cfHandler);
  ~InterHostCache();

  // Setup Socket-Client connection
  void init();
  // Send file to Rank-0
  bool send_file(std::string filename);
  // Receive file from Rank-0
  bool recv_file(std::string filename);
};

} // namespace serialization
