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
#include <string>
namespace synapse_logger {

void dump_reference(
    const std::string& ref,
    const std::string& ref_type,
    float* vec,
    int n);

void command(const std::string& x);
void put_log(const std::string& what);

void start_hw_profile();

void stop_hw_profile();

} // namespace synapse_logger
