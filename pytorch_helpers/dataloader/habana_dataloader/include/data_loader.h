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

namespace scaleoutdemoloader {
enum LoaderDataType { IMAGE, LABEL, BBOX, BBOX_LABEL, IMAGE_SHAPE, IMAGE_ID };

void* create_data_loader();
void destroy_data_loader(void* data_loader);

void data_loader_init(void* loader, const char* path);
void data_loader_get_data(
    void* loader,
    const LoaderDataType type,
    const unsigned size,
    char* data);
void data_loader_inc(void* loader);
void data_loader_reset(void* loader);

uint64_t get_database_size(void* data_loader);
} // namespace scaleoutdemoloader
