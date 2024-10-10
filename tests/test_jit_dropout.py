import os

import habana_frameworks.torch.core as htcore
import numpy as np
import pytest
import torch
import torch.nn as nn
from test_utils import cpu, env_var_in_scope, hpu

pytestmark = pytest.mark.skip(reason="Tests in this file are chaning env variables")

test_case_list = [(384, 768)]


@pytest.mark.parametrize("D1, D2", test_case_list)
@pytest.mark.xfail(reason="AttributeError: module 'habana_frameworks.torch.core' has no attribute 'enable'")
def test_dropout(D1, D2):
    with env_var_in_scope(
        {
            "PT_HPU_GRAPH_FUSION_OPS_FILE": os.path.join(
                os.environ["MODEL_GARDEN_PYTORCH_PATH"], "nlp/bert/BERT_Fusion_Ops.txt"
            )
        }
    ):
        shape = (D1, D2)
        in_t = torch.ones(shape, requires_grad=True)
        hpu_t = in_t.detach().to(hpu)
        hpu_t.requires_grad = True

        with torch.jit.optimized_execution(True):
            dp = 0.3
            dropoutmod = nn.Dropout(p=dp)
            htcore.enable()
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_profiling_executor(False)
            hpu_t = in_t.to(hpu)
            model_trace_hpu = torch.jit.trace(dropoutmod, hpu_t, check_trace=False)
            model_trace_hpu.graph_for(hpu_t)
            # print("Fused graph on HPU: ")
            # print(model_trace_hpu_graph)
            hpu_out = model_trace_hpu(hpu_t)
            output1 = hpu_out.to(cpu).detach().numpy()
            grad_out = torch.randn((D1, D2), requires_grad=False)
            hpu_out.backward(grad_out.detach().to(hpu))
            # Determine the sample dropout probability
            dropout_prob_sample = 1.0 - float(np.nonzero(output1)[0].size) / np.cumprod(shape)[-1]
            np.testing.assert_almost_equal(dp, dropout_prob_sample, decimal=2)


if __name__ == "__main__":
    test_dropout(*test_case_list[0])
