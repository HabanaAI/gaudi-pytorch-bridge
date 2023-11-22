/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/mv.h"

namespace habana {

OutputMetaDataVector MvOpsMeta(const at::Stack& stack) {
  auto mat = stack.at(0).toTensor();

  OutputMetaData meta;
  meta.shape = {mat.sizes()[0]};
  meta.dtype = mat.scalar_type();
  return {meta};
}
} // namespace habana
