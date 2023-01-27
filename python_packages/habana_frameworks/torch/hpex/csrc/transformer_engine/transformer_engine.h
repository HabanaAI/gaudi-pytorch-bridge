/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 * Changes:
 * - Removed unused declarations
 ******************************************************************************/

/*! \file transformer_engine.h
 *  \brief Base classes and functions of Transformer Engine API.
 */

#ifndef TRANSFORMER_ENGINE_TRANSFORMER_ENGINE_H_
#define TRANSFORMER_ENGINE_TRANSFORMER_ENGINE_H_

#include <stddef.h>

#ifdef __cplusplus

/*! \namespace transformer_engine
 *  \brief Namespace containing C++ API of Transformer Engine.
 */
namespace transformer_engine {

/*! \enum DType
 *  \brief TE datatype.
 */
enum class DType {
  kByte = 0,
  kInt32 = 1,
  kFloat32 = 2,
  kFloat16 = 3,
  kBFloat16 = 4,
  kFloat8E4M3 = 5,
  kFloat8E5M2 = 6,
  kNumTypes
};

} // namespace transformer_engine

#endif

#endif // TRANSFORMER_ENGINE_TRANSFORMER_ENGINE_H_
