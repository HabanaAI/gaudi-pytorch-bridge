/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, foreach) {
  GenerateInputs(2);

  at::_foreach_abs_({GetCpuInput(0), GetCpuInput(1)});
  at::_foreach_abs_({GetHpuInput(0), GetHpuInput(1)});

  Compare(GetCpuInput(0), GetHpuInput(0));
  Compare(GetCpuInput(1), GetHpuInput(1));
}
