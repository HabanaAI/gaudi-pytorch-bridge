/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once

#include <pybind11/pybind11.h>

#define TORCH_HCL_CPP_API __attribute__((visibility("default")))

void torch_hcl_python_init();
