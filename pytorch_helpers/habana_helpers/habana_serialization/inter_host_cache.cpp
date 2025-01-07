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
#include "habana_serialization/inter_host_cache.h"
#include <arpa/inet.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include "backend/synapse_helpers/env_flags.h"
#include "habana_helpers/logging.h"

#define NUM_ATTEMPTS 100
#define PER_ATTEMPT_SLEEP_MS 50
#define SOCKET_TX_PORT_OFFSET 100
#define INTERHOST_LOG "[INTERHOST] "

#define CHECK(val, line)    \
  {                         \
    if (val <= 0) {         \
      PT_HABHELPER_FATAL(   \
          INTERHOST_LOG,    \
          "Error Val: ",    \
          val,              \
          ", Line: ",       \
          line,             \
          ", Error: ",      \
          errno,            \
          ", ",             \
          strerror(errno)); \
    }                       \
  }

namespace serialization {

InterHostCache::InterHostCache(
    std::string& cache_path,
    std::shared_ptr<CacheFileHandler> cfHandler)
    : is_cache_valid_{true}, cache_path_{cache_path}, cfHandler_{cfHandler} {
  const char* s_rank =
      getenv("RANK") ? getenv("RANK") : getenv("OMPI_COMM_WORLD_RANK");
  const char* s_wsize = getenv("WORLD_SIZE") ? getenv("WORLD_SIZE")
                                             : getenv("OMPI_COMM_WORLD_SIZE");
  const char* s_master_port = getenv("MASTER_PORT");
  const char* s_master_addr = getenv("MASTER_ADDR");
  const char* s_l_w_size = getenv("LOCAL_WORLD_SIZE")
      ? getenv("LOCAL_WORLD_SIZE")
      : getenv("OMPI_COMM_WORLD_LOCAL_SIZE");

  {
    // Basic Sanity check
    std::stringstream ss;
    if (s_rank)
      ss << "Rank: " << s_rank << "\t";
    if (s_wsize)
      ss << "WorldSize: " << s_wsize << "\t";
    if (s_master_addr)
      ss << "MasterAddr: " << s_master_addr << "\t";
    if (s_master_port)
      ss << "MasterPort: " << s_master_port << "\t";
    if (s_l_w_size)
      ss << "Local WorldSize: " << s_l_w_size << "\t";

    PT_HABHELPER_DEBUG(INTERHOST_LOG, ss.str());
    if (!s_rank || !s_wsize || !s_master_port || !s_master_addr ||
        !s_l_w_size) {
      PT_HABHELPER_WARN(INTERHOST_LOG, "InterHostCache() Failed: ", ss.str());
      invalidate_cache();
      return;
    }
  }

  rank = std::atoi(s_rank);
  l_w_size = std::atoi(s_l_w_size);
  w_size = std::atoi(s_wsize);
  master_port = std::atoi(s_master_port) + SOCKET_TX_PORT_OFFSET;
  master_addr = s_master_addr;
  // No need to initiate inter host communication if single node run
  if ((rank == 0 && w_size > l_w_size) ||
      (rank >= l_w_size && w_size > l_w_size)) {
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
      PT_HABHELPER_WARN(INTERHOST_LOG, "Socket() failed");
      invalidate_cache();
      return;
    }
  }
}

void InterHostCache::init() {
  // PT_ENABLE_INTER_HOST_CACHING could be set by mistake for single node runs
  // avoid doing additional initialization
  if (!is_cache_valid_ || w_size == l_w_size) {
    invalidate_cache();
    return;
  }

  struct sockaddr_in server_addr;
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = master_port;
  server_addr.sin_addr.s_addr = inet_addr(master_addr.c_str());

  int e{-1};
  if (rank == 0) {
    e = bind(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));
  } else if (rank >= l_w_size) {
    int attempt = 0;
    e = connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));
    while (e < 0 && attempt < NUM_ATTEMPTS) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(PER_ATTEMPT_SLEEP_MS));
      e = connect(sockfd, (struct sockaddr*)&server_addr, sizeof(server_addr));
      attempt++;
    }
  }

  if (e < 0) {
    PT_HABHELPER_WARN(INTERHOST_LOG, "Bind/Connect failed");
    invalidate_cache();
    close(sockfd);
    return;
  }

  if (rank == 0) {
    e = listen(sockfd, w_size);
    if (e != 0) {
      PT_HABHELPER_WARN(INTERHOST_LOG, "Listen() failed");
      invalidate_cache();
      close(sockfd);
      return;
    }

    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);

    int client_count = l_w_size;
    // Rank-0 will create a separate thread per client for listening
    std::vector<std::thread> tp;
    while (client_count < w_size) {
      int clientfd =
          accept(sockfd, (struct sockaddr*)&client_addr, &client_len);
      tp.emplace_back(&InterHostCache::thread_function, this, clientfd);
      client_count++;
      PT_HABHELPER_DEBUG(INTERHOST_LOG, "Connected clients: ", client_count);
    }

    for (auto& th : tp) {
      th.detach();
    }
  }

  PT_HABHELPER_DEBUG(INTERHOST_LOG, "InterHostCache() Completed: Rank=", rank);
}

void InterHostCache::thread_function(int clientfd) {
  char tdata[MAX_SIZE];

  while (true) {
    int bytes_read;

    // Get command from Client
    bytes_read = recv(clientfd, tdata, cmdSet.size() + 1, 0);
    if (bytes_read == 0)
      break;
    CHECK(bytes_read, __LINE__);

    std::string cmd(tdata);
    if (!cmdEnd.compare(cmd)) {
      // Terminate this thread if "END" command is received
      PT_HABHELPER_DEBUG(INTERHOST_LOG, "Thread completed");
      send(clientfd, cmdAck.c_str(), cmdSet.size() + 1, 0);
      break;
    }

    size_t num;
    // Get the number of bytes in filename that will be sent next
    bytes_read = recv(clientfd, &num, sizeof(num), 0);
    CHECK(bytes_read, __LINE__);

    // Get the filename
    bytes_read = recv(clientfd, tdata, num, 0);
    CHECK(bytes_read, __LINE__);

    std::string filename(tdata);
    std::string recpfile = recipe_file_path(cache_path_, filename);
    std::string metafile = metadata_file_path(cache_path_, filename);
    size_t size = 0;
    int fd = cfHandler_->openAndLockFile(
        metafile.c_str(), O_RDWR | O_CREAT, false, size);

    if (!cmdSet.compare(cmd)) {
      // Get lock status and decide if you can continue
      if (fd < 0 || size > 0) {
        if (fd >= 0)
          cfHandler_->fileClose(fd);

        send(clientfd, cmdRej.c_str(), cmdSet.size() + 1, 0);
        continue;
      }

      send(clientfd, cmdAck.c_str(), cmdSet.size() + 1, 0);
      _recv_file(metafile, clientfd, tdata);
      _recv_file(recpfile, clientfd, tdata);

      cfHandler_->addFileInfo(filename);

    } else if (!cmdGet.compare(cmd)) {
      // Get lock status and decide if you can continue
      if (fd < 0 || size <= 0) {
        if (fd >= 0)
          cfHandler_->fileClose(fd);

        send(clientfd, cmdRej.c_str(), cmdSet.size() + 1, 0);
        continue;
      }

      send(clientfd, cmdAck.c_str(), cmdSet.size() + 1, 0);
      _send_file(metafile, clientfd, tdata);
      _send_file(recpfile, clientfd, tdata);

    } else {
      PT_HABHELPER_FATAL(INTERHOST_LOG, "Unknown CMD: ", tdata);
      break;
    }

    cfHandler_->fileClose(fd);
  }

  close(clientfd);
  return;
}

InterHostCache::~InterHostCache() {
  if (rank != 0 && is_cache_valid_ && rank >= l_w_size) {
    send(sockfd, cmdEnd.c_str(), cmdSet.size() + 1, 0);
    auto bytes_recv = recv(sockfd, data, cmdSet.size() + 1, 0);
    CHECK(bytes_recv, __LINE__);
    close(sockfd);
  }
}

bool InterHostCache::_send_file(std::string filename, int sock, char* buff) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    PT_HABHELPER_FATAL(INTERHOST_LOG, "Unable to open for Send(): ", filename);
    return false;
  }
  fseek(fp, 0L, SEEK_END);
  size_t num = ftell(fp);
  send(sock, &num, sizeof(num), 0);
  fseek(fp, 0L, SEEK_SET);

  int bytes_recv{0};
  while ((bytes_recv = fread(buff, 1, MAX_SIZE, fp)) > 0) {
    send(sock, buff, bytes_recv, 0);
  }

  bytes_recv = recv(sock, buff, cmdSet.size() + 1, 0);
  CHECK(bytes_recv, __LINE__);
  fclose(fp);
  return true;
}

bool InterHostCache::_recv_file(std::string filename, int sock, char* buff) {
  FILE* fp = fopen(filename.c_str(), "wb");
  if (fp == NULL) {
    PT_HABHELPER_FATAL(INTERHOST_LOG, "Unable to open for Recv(): ", filename);
    return false;
  }

  size_t num;
  int bytes_recv = recv(sock, &num, sizeof(num), 0);
  CHECK(bytes_recv, __LINE__);

  while (num > 0) {
    bytes_recv = recv(sock, buff, num > MAX_SIZE ? MAX_SIZE : num, 0);
    CHECK(bytes_recv, __LINE__);
    fwrite(buff, 1, bytes_recv, fp);
    num -= bytes_recv;
  }

  send(sock, cmdAck.c_str(), cmdSet.size() + 1, 0);
  fclose(fp);
  return true;
}

bool InterHostCache::send_file(std::string cache_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Avoid sending files in node0
  if (rank == 0 || !is_cache_valid_ || rank < l_w_size)
    return false;

  std::string recpfile = recipe_file_path(cache_path_, cache_id);
  std::string metafile = metadata_file_path(cache_path_, cache_id);
  size_t size = 0;
  int fd = cfHandler_->openAndLockFile(
      metafile.c_str(), O_RDWR | O_CREAT, false, size);

  if (fd < 0 || size <= 0) {
    if (fd >= 0)
      cfHandler_->fileClose(fd);

    return false;
  }

  // Limited namespace scope.
  {
    using namespace Logger;
    PT_HABHELPER_INFO("InterHostCache Send", cache_id);
  }

  size_t num;
  int bytes_recv;

  cache_id += "\0";
  num = cache_id.size() + 1;
  send(sockfd, cmdSet.c_str(), cmdSet.size() + 1, 0);
  send(sockfd, &num, sizeof(num), 0);
  send(sockfd, cache_id.c_str(), num, 0);
  bytes_recv = recv(sockfd, data, cmdSet.size() + 1, 0);
  CHECK(bytes_recv, __LINE__);
  if (!cmdRej.compare(data)) {
    cfHandler_->fileClose(fd);
    return false;
  }

  _send_file(metafile, sockfd, data);
  _send_file(recpfile, sockfd, data);

  PT_HABHELPER_DEBUG(INTERHOST_LOG, "Sent by: ", rank, ", File: ", cache_id);
  cfHandler_->fileClose(fd);
  return true;
}

bool InterHostCache::recv_file(std::string cache_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (rank == 0 || !is_cache_valid_ || rank < l_w_size)
    return false;

  std::string recpfile = recipe_file_path(cache_path_, cache_id);
  std::string metafile = metadata_file_path(cache_path_, cache_id);
  size_t size = 0;
  int fd = cfHandler_->openAndLockFile(
      metafile.c_str(), O_RDWR | O_CREAT, false, size);

  if (fd < 0 || size > 0) {
    if (fd >= 0)
      cfHandler_->fileClose(fd);

    return false;
  }

  // Limited namespace scope
  {
    using namespace Logger;
    PT_HABHELPER_INFO("InterHostCache Recv", cache_id);
  }

  size_t num;
  int bytes_recv;

  cache_id += "\0";
  num = cache_id.size() + 1;
  send(sockfd, cmdGet.c_str(), cmdSet.size() + 1, 0);
  send(sockfd, &num, sizeof(num), 0);
  send(sockfd, cache_id.c_str(), num, 0);
  bytes_recv = recv(sockfd, data, cmdSet.size() + 1, 0);
  CHECK(bytes_recv, __LINE__);
  if (!cmdRej.compare(data)) {
    cfHandler_->fileClose(fd);
    return false;
  }

  _recv_file(metafile, sockfd, data);
  _recv_file(recpfile, sockfd, data);

  cfHandler_->addFileInfo(cache_id);

  PT_HABHELPER_DEBUG(
      INTERHOST_LOG, "Received by: ", rank, ", File: ", cache_id);

  cfHandler_->fileClose(fd);
  return true;
}

} // namespace serialization
