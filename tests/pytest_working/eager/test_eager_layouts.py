###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import torch

torch.manual_seed(0)


# Test case to simulate the writes to gradient bucket views similar to what happens in DDP all reduce
def test_hpu_simulate_allreduce_bucket_views():
    import habana_frameworks.torch.core as htcore

    def fn(dev, a, g1, g2):

        v1 = a.as_strided([N, C, H, W], [256, 4, 2, 1], 16)
        v2 = a.as_strided([4, 4], [4, 1], N * C * H * W)

        conv_op = torch.nn.Conv2d(C, C, 1, stride=1).to(dev)
        d = conv_op(g1)
        torch.mul(d, 0.5, out=v1)

        return v1

    with torch.no_grad():
        dev = torch.device("cpu")
        torch.manual_seed(0)
        N = 64
        C = 64
        H = 2
        W = 2
        buff_size = N * C * H * W + 16
        bucket = torch.randn(buff_size).to(dev)
        g1 = torch.randn([N, C, H, W]).to(dev)
        g2 = torch.randn([4, 4]).to(dev)

        o1 = fn(dev, bucket, g1, g2)

        dev = torch.device("hpu")
        torch.manual_seed(0)
        hbucket = torch.randn([buff_size]).to(dev)
        hg1 = torch.randn([N, C, H, W]).to(dev)
        hg2 = torch.randn([4, 4]).to(dev)

        ho1 = fn(dev, hbucket, hg1, hg2)

        assert torch.allclose(ho1.cpu(), o1, atol=0.01, rtol=0.01)

        # TODO SW-154859
        # assert torch.allclose(ha.cpu(), a, atol = 0.01, rtol = 0.01) # problematic
