# ******************************************************************************
# Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************


# Generic test for scatter add based on the rules specified in the PT doc

# From : https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html#torch.Tensor.scatter_add_
# self, index and src should have same number of dimensions.
# It is also required that index.size(d) <= src.size(d) for all dimensions d,
# and that index.size(d) <= self.size(d) for all dimensions d != dim.
# Note that index and src do not broadcast.

# interpretation: index.size(dim) can be greter than self.size(dim)
# i.e, src.size(d> <= self.size(d) for all d except d = dim; where it can be more

import random

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch


@pytest.mark.xfail(reason="Results mismatch")
def test_scatter_add():
    DIM_MAX = 5  # 0-4
    MAX_SIZE = 10
    TEST_COUNT = 50
    # the src size on axis dim can be anything
    # so use a separate def for greater freedom to control this
    MAX_SRC_SIZE_ON_AXIS_DIM = 20
    src_dt = torch.float32
    slf_dt = torch.float32

    def get_d():
        return random.randint(1, DIM_MAX)

    def get_axis(d):
        axis = random.randint(0, d - 1)
        # print("axis  = ", axis)
        return axis

    def create_self_tensor():
        d = get_d()
        s = [random.randint(1, MAX_SIZE) for _ in range(d)]
        shape = tuple(s)
        self = torch.zeros(shape, dtype=slf_dt)
        # print("self size  = ", self.size())

        return self

    def create_src_tensor(self, axis):
        s = []
        for i in range(self.dim()):
            if i != axis:
                s.append(random.randint(1, self.size(i)))
            else:
                s.append(random.randint(1, MAX_SRC_SIZE_ON_AXIS_DIM))
        shape = tuple(s)

        src = torch.ones(shape, dtype=src_dt)
        # print("src size  = ", src.size())
        return src

    def create_index_tensor(self, src, axis):
        s = []
        for i in range(src.dim()):
            s.append(random.randint(1, src.size(i)))
        shape = tuple(s)
        # index contents should be 0 .. self.size(axis)-1
        index = torch.randint(0, self.size(axis), shape)
        # print("index size  = ", index.size())
        # print("index  = ", index)
        return index

    output_rtol = 1e-3
    output_atol = 1e-3

    for t in range(TEST_COUNT):
        self = create_self_tensor()
        axis = get_axis(self.dim())
        src = create_src_tensor(self, axis)
        index = create_index_tensor(self, src, axis)
        self_h = self.to("hpu")
        src_h = src.to("hpu")
        index_h = index.to("hpu")
        self.scatter_add_(axis, index, src)
        self_h.scatter_add_(axis, index_h, src_h)
        htcore.mark_step()

        print(
            "TC = ",
            t,
            "\taxis = ",
            axis,
            "SIZES : self = ",
            self.size(),
            "\t\t\t index = ",
            index.size(),
            "\t\t\tsrc size = ",
            index.size(),
        )
        np.testing.assert_allclose(
            self.detach().numpy(),
            self_h.to("cpu").detach().numpy(),
            output_rtol,
            output_atol,
        )
