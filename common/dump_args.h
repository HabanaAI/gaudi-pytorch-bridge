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
#pragma once

#define DUMP_ARG(a) " " #a "=", to_string(a)
#define DUMP_2ARGS(a1, a2) DUMP_ARG(a1), DUMP_ARG(a2)
#define DUMP_3ARGS(a1, ...) DUMP_ARG(a1), DUMP_2ARGS(__VA_ARGS__)
#define DUMP_4ARGS(a1, ...) DUMP_ARG(a1), DUMP_3ARGS(__VA_ARGS__)
#define DUMP_5ARGS(a1, ...) DUMP_ARG(a1), DUMP_4ARGS(__VA_ARGS__)
#define DUMP_6ARGS(a1, ...) DUMP_ARG(a1), DUMP_5ARGS(__VA_ARGS__)
#define DUMP_7ARGS(a1, ...) DUMP_ARG(a1), DUMP_6ARGS(__VA_ARGS__)
#define DUMP_8ARGS(a1, ...) DUMP_ARG(a1), DUMP_7ARGS(__VA_ARGS__)
#define DUMP_9ARGS(a1, ...) DUMP_ARG(a1), DUMP_8ARGS(__VA_ARGS__)
#define DUMP_10ARGS(a1, ...) DUMP_ARG(a1), DUMP_9ARGS(__VA_ARGS__)
#define DUMP_11ARGS(a1, ...) DUMP_ARG(a1), DUMP_10ARGS(__VA_ARGS__)
#define DUMP_12ARGS(a1, ...) DUMP_ARG(a1), DUMP_11ARGS(__VA_ARGS__)