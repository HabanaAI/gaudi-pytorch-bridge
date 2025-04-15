/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include <ATen/ATen.h>

class CopyOperation {
 public:
    CopyOperation(
        const at::Tensor& src,
        const at::Tensor& dst,
        bool non_blocking,
        c10::hpu::HPUStream stream,
        void* host_ptr)
        : src_(src),
          dst_(dst),
          non_blocking_(non_blocking),
          stream_(stream),
          host_ptr_(host_ptr) {}

    const at::Tensor& src() const {
        return src_;
    }

    const at::Tensor& dst() const {
        return dst_;
    }

    bool non_blocking() const {
        return non_blocking_;
    }

    c10::hpu::HPUStream stream() const {
        return c10::hpu::HPUStream(stream_);
    }

    void* host_ptr() const {
        return host_ptr_;
    }


 private:
    at::Tensor src_;
    at::Tensor dst_;
    bool non_blocking_;
    c10::Stream stream_;
    void* host_ptr_;
};
