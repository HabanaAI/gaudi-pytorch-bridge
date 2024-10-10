/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
