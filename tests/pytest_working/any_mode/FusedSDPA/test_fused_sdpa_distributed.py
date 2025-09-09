###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import copy
import os
from copy import deepcopy
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
from habana_frameworks.torch.hpex.kernels import FusedSDPA
from test_utils import (
    compare_tensors,
)
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import (
    DTensor,
    Replicate,
    Shard,
    distribute_tensor,
)
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.distributed._tensor.common_dtensor import (
    NUM_DEVICES,
    DTensorTestBase,
    ModelArgs,
    Transformer,
    TransformerBlock,
    skip_unless_torch_gpu,
    with_comms,
)

os.environ["HABANA_VISIBLE_MODULES"] = "0,1,2,3,4,5,6,7"


def is_pytest_mode_lazy():
    # Read PT_HPU_LAZY_MODE and return its state
    return os.environ.get("PT_HPU_LAZY_MODE") == "1"


# Communication count expectations for different operations
c10d_functional = torch.ops.c10d_functional
reduce_scatter, all_gather, all_reduce = (
    c10d_functional.reduce_scatter_tensor,
    c10d_functional.all_gather_into_tensor,
    c10d_functional.all_reduce,
)


class ExpCommCounts(NamedTuple):
    fwd: dict | None = None
    bwd: dict | None = None
    optim: dict | None = None


class FusedAttentionHPU(nn.Module):
    """Custom Attention module that uses FusedSDPA instead of F.scaled_dot_product_attention"""

    def __init__(self, args: ModelArgs, attn_fn=F.scaled_dot_product_attention):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dropout_p = args.dropout_p
        self.resid_dropout = nn.Dropout(args.dropout_p)
        self.use_attn_mask = args.use_attn_mask
        self.attn_fn = attn_fn

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)

        # If attn_fn is F.scaled_dot_product_attention and used with SDPBackend.OVERRIDEABLE
        # AutogradHPU backend is called for at::_scaled_dot_product_fused_attention_overrideable
        if self.attn_fn == F.scaled_dot_product_attention:
            with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
                output = self.attn_fn(
                    queries,
                    keys,
                    values,
                    None,
                    self.dropout_p if self.training else 0,
                    self.use_attn_mask,
                )
        else:  # FusedSDPA.apply: direct Autograd Override
            output = self.attn_fn(
                queries,
                keys,
                values,
                None,
                self.dropout_p if self.training else 0,
                self.use_attn_mask,
            )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output))


class FusedTransformerHPU(Transformer):
    """Custom Transformer that uses FusedAttentionHPU for attention layers"""

    def __init__(self, args: ModelArgs, attn_fn=F.scaled_dot_product_attention):
        super().__init__(args)
        # Replace attention modules in each layer with FusedAttentionHPU
        for layer in self.layers:
            layer.attention = FusedAttentionHPU(args, attn_fn)


class DTensorFusedSDPATest(DTensorTestBase):
    def _check_module(self, m1, m2, check_grad=False):
        """Compare modules parameter by parameter"""
        named_parameters = dict(m1.named_parameters())
        for name, param_m2 in m2.named_parameters():
            self.assertTrue(name in named_parameters)
            param_m1 = named_parameters[name]
            if check_grad:
                param_m2 = param_m2.grad
                param_m1 = param_m1.grad
            if isinstance(param_m2, DTensor):
                replicate = [Replicate()]
                param_m2 = param_m2.redistribute(device_mesh=param_m2.device_mesh, placements=replicate).to_local()
            self.assertEqual(param_m2, param_m1)

    def _setup_single_gpu_fused_model(self, model_args, dtype, attn_fn):
        """Setup single GPU model with FusedSDPA"""
        return FusedTransformerHPU(model_args, attn_fn).to(device=self.device_type, dtype=dtype)

    def _setup_tp_fused_model(self, model, is_seq_parallel, dtype, attn_fn):
        """Setup tensor parallel model with FusedSDPA"""
        model_tp = deepcopy(model)
        self._check_module(model, model_tp)
        device_mesh = DeviceMesh(self.device_type, torch.arange(0, NUM_DEVICES))
        local_output_for_attn = dtype is torch.float64
        return self._parallelize_fused_transformer(
            model_tp,
            device_mesh,
            is_seq_parallel,
            local_output_for_attn=local_output_for_attn,
        )

    def _parallelize_fused_transformer(
        self,
        module: FusedTransformerHPU,
        device_mesh: DeviceMesh,
        use_seq_parallel: bool,
        local_output_for_attn: bool = False,
    ) -> nn.Module:
        """Parallelize FusedTransformerHPU similar to Transformer.parallelize"""
        # Parallelize the root submodules.
        if use_seq_parallel:
            root_plan = {
                "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                "pos_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(0)),
                "norm": SequenceParallel(),
            }
        else:
            root_plan = {
                "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
                "pos_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
            }

        module_tp = parallelize_module(module, device_mesh, root_plan)

        # Parallelize the attention and feed forward submodules.
        for layer in module_tp.layers:
            layer_parallelize_plan = {}
            if use_seq_parallel:
                layer_parallelize_plan["attention"] = PrepareModuleInput(
                    input_layouts=Shard(1),
                    desired_input_layouts=Replicate(),
                )
                # shard the LayerNorms
                layer_parallelize_plan["attention_norm"] = SequenceParallel()
                layer_parallelize_plan["ffn_norm"] = SequenceParallel()

            # Parallelize attention layers (FusedAttentionHPU)
            layer_parallelize_plan["attention.wq"] = ColwiseParallel(use_local_output=local_output_for_attn)
            layer_parallelize_plan["attention.wk"] = ColwiseParallel(use_local_output=local_output_for_attn)
            layer_parallelize_plan["attention.wv"] = ColwiseParallel(use_local_output=local_output_for_attn)
            layer_parallelize_plan["attention.wo"] = (
                RowwiseParallel(output_layouts=Shard(1)) if use_seq_parallel else RowwiseParallel()
            )

            # Parallelize feed forward layers
            layer_parallelize_plan["feed_forward.w1"] = (
                ColwiseParallel(input_layouts=Shard(1)) if use_seq_parallel else ColwiseParallel()
            )
            layer_parallelize_plan["feed_forward.w2"] = (
                RowwiseParallel(output_layouts=Shard(1)) if use_seq_parallel else RowwiseParallel()
            )

            parallelize_module(layer, device_mesh, layer_parallelize_plan)

        # Parallelize the output submodule
        output_parallelize_plan = (
            ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
            )
            if use_seq_parallel
            else ColwiseParallel(output_layouts=Replicate())
        )
        parallelize_module(module_tp.output, device_mesh, output_parallelize_plan)

        # Handle weight tying if enabled
        if module_tp.model_args.weight_tying:
            module_tp.output.weight = module_tp.tok_embeddings.weight

        return module_tp

    def _setup_optimizer(self, model, model_tp):
        """Setup optimizers for both models"""
        LR = 0.25
        optim = torch.optim.Adam(model.parameters(), lr=LR)
        optim_tp = torch.optim.Adam(model_tp.parameters(), lr=LR)
        return optim, optim_tp

    def _validate_fwd(self, model, model_tp, inp, expected_comms_dict=None, check_comms=True):
        """Validate forward pass"""
        output = model(inp)
        with CommDebugMode() as comm_mode:
            output_tp = model_tp(inp)
        self.assertEqual(output, output_tp)
        if check_comms:
            self.assertDictEqual(comm_mode.get_comm_counts(), expected_comms_dict or {})
        return output, output_tp

    def _validate_bwd(
        self,
        model,
        model_tp,
        output,
        output_tp,
        expected_comms_dict=None,
        check_comms=True,
    ):
        """Validate backward pass"""
        output.sum().backward()
        with CommDebugMode() as comm_mode:
            output_tp.sum().backward()
        self._check_module(model, model_tp, check_grad=True)
        if check_comms:
            self.assertDictEqual(comm_mode.get_comm_counts(), expected_comms_dict or {})

    def _validate_optim_step(
        self,
        model,
        model_tp,
        optim,
        optim_tp,
        expected_comms_dict=None,
        check_comms=True,
    ):
        """Validate optimizer step"""
        optim.step()
        from torch.distributed.tensor.experimental import implicit_replication

        with implicit_replication(), CommDebugMode() as comm_mode:
            optim_tp.step()
        self._check_module(model, model_tp)
        if check_comms:
            self.assertDictEqual(comm_mode.get_comm_counts(), expected_comms_dict or {})

    @staticmethod
    def _thaw_params(thaw_params, model, model_tp):
        """Set requires_grad patterns for specific parameters"""
        if not thaw_params:
            return
        for target_model in [model, model_tp]:
            for n, p in target_model.named_parameters():
                if n not in thaw_params:
                    p.requires_grad_(False)

    @pytest.mark.skipif(is_pytest_mode_lazy(), reason="DTensor test is not supported in lazy mode")
    @with_comms
    @skip_unless_torch_gpu
    @parametrize("attn_fn", [FusedSDPA.apply, F.scaled_dot_product_attention])
    def test_fused_sdpa_distributed(self, attn_fn):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        comm_mode = CommDebugMode()

        # bsz, n_heads, slen, head_dim
        query = torch.rand(
            (4, 8, 8, 8),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.rand(
            (4, 8, 8, 8),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.rand(
            (4, 8, 8, 8),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # Test different sharding strategies
        sharding_strategies = [
            ([Replicate()], "replicate"),
            ([Shard(0)], "batch_shard"),
            ([Shard(1)], "head_shard"),
        ]

        for placement_spec, strategy_name in sharding_strategies:
            # Reset gradients for each test
            if query.grad is not None:
                query.grad.zero_()
            if key.grad is not None:
                key.grad.zero_()
            if value.grad is not None:
                value.grad.zero_()

            # Create distributed tensors
            dist_query = distribute_tensor(query, device_mesh, placement_spec)
            dist_key = distribute_tensor(key, device_mesh, placement_spec)
            dist_value = distribute_tensor(value, device_mesh, placement_spec)

            # Test forward pass
            # Local FusedSDPA
            if attn_fn == F.scaled_dot_product_attention:
                with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
                    local_out = attn_fn(
                        query,
                        key,
                        value,
                        None,  # attn_mask
                        0.0,  # dropout_p
                        True,  # is_causal
                    )
            else:  # FusedSDPA.apply: direct Autograd Override
                local_out = attn_fn(
                    query,
                    key,
                    value,
                    None,  # attn_mask
                    0.0,  # dropout_p
                    True,  # is_causal
                )

            # Distributed FusedSDPA
            with comm_mode:
                if attn_fn == F.scaled_dot_product_attention:
                    with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
                        dist_out = attn_fn(
                            dist_query,
                            dist_key,
                            dist_value,
                            None,  # attn_mask
                            0.0,  # dropout_p
                            True,  # is_causal
                        )
                else:  # FusedSDPA.apply: direct Autograd Override
                    dist_out = attn_fn(
                        dist_query,
                        dist_key,
                        dist_value,
                        None,  # attn_mask
                        0.0,  # dropout_p
                        True,  # is_causal
                    )
                # For head sharding, we expect no communication
                if strategy_name == "head_shard":
                    assert comm_mode.get_total_counts() == 0, (
                        f"Expected no communication for head sharding, got {comm_mode.get_total_counts()}"
                    )

            # Verify output shapes and values
            assert dist_out.shape == local_out.shape, f"Shape mismatch: {dist_out.shape} vs {local_out.shape}"
            compare_tensors(dist_out.full_tensor().cpu(), local_out.cpu(), atol=1e-2, rtol=1e-2)

            # Verify sharding is preserved for head sharding
            if strategy_name == "head_shard":
                assert dist_out.placements[0].is_shard(dim=1), "Head sharding not preserved in output"

            # Test backward pass
            local_out.sum().backward()
            with comm_mode:
                dist_out.sum().backward()
                # For head sharding, backward should also have minimal communication BUT
                # [TODO]: backward here needs a all_gather_into_tensor collective for hpu, but
                # nn.functional.sdpa doesn't. Why is it used for hpu and not for cuda?
                # if strategy_name == "head_shard":
                #     assert comm_mode.get_total_counts() == 0

            # Verify gradients
            if strategy_name == "head_shard":
                assert dist_query.grad.placements[0].is_shard(dim=1), "Query grad sharding not preserved"
                assert dist_key.grad.placements[0].is_shard(dim=1), "Key grad sharding not preserved"
                assert dist_value.grad.placements[0].is_shard(dim=1), "Value grad sharding not preserved"

            compare_tensors(dist_query.grad.full_tensor().cpu(), query.grad.cpu(), atol=1e-2, rtol=1e-2)
            compare_tensors(dist_key.grad.full_tensor().cpu(), key.grad.cpu(), atol=1e-2, rtol=1e-2)
            compare_tensors(dist_value.grad.full_tensor().cpu(), value.grad.cpu(), atol=1e-2, rtol=1e-2)

    @pytest.mark.skipif(is_pytest_mode_lazy(), reason="DTensor test is not supported in lazy mode")
    @with_comms
    @skip_unless_torch_gpu
    @parametrize("attn_fn", [FusedSDPA.apply, F.scaled_dot_product_attention])
    def test_fused_sdpa_with_attention_mask_distributed(self, attn_fn):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # bsz, n_heads, slen, head_dim
        batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16
        query = torch.rand(
            (batch_size, num_heads, seq_len, head_dim),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key = torch.rand(
            (batch_size, num_heads, seq_len, head_dim),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        value = torch.rand(
            (batch_size, num_heads, seq_len, head_dim),
            device=self.device_type,
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # Create attention mask
        attn_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device_type, dtype=torch.bfloat16))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))

        # Test with head sharding (most efficient for attention)
        dist_query = distribute_tensor(query, device_mesh, [Shard(1)])
        dist_key = distribute_tensor(key, device_mesh, [Shard(1)])
        dist_value = distribute_tensor(value, device_mesh, [Shard(1)])
        dist_attn_mask = distribute_tensor(attn_mask, device_mesh, [Replicate()])

        # Local computation
        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            local_out = attn_fn(
                query,
                key,
                value,
                attn_mask,
                0.0,
                False,  # is_causal; Using explicit mask instead
            )

        # Distributed computation
        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            dist_out = attn_fn(dist_query, dist_key, dist_value, dist_attn_mask, 0.0, False)

        # Verify results
        assert dist_out.placements[0].is_shard(dim=1), "Head sharding not preserved with attention mask"
        compare_tensors(dist_out.full_tensor().cpu(), local_out.cpu(), atol=1e-2, rtol=1e-2)

    @pytest.mark.skipif(is_pytest_mode_lazy(), reason="DTensor test is not supported in lazy mode")
    @with_comms
    @skip_unless_torch_gpu
    @parametrize(
        "thaw_params, is_seq_parallel, dtype, exp_cnts, attn_fn",
        [
            (
                None,  # all require grad seq_parallel float32 baseline
                True,
                torch.float32,
                ExpCommCounts(bwd={reduce_scatter: 5, all_gather: 8}, optim={all_reduce: 30}),
                "FusedSDPA",
            ),
            (
                None,  # all require grad seq_parallel float32 baseline
                True,
                torch.float32,
                ExpCommCounts(bwd={reduce_scatter: 5, all_gather: 8}, optim={all_reduce: 30}),
                "SDPAOverrideable",
            ),
        ],
        name_fn=lambda thaw, seq, dtype, exp_cnts, attn_fn: f"{'seq_parallel_' if seq else ''}"
        + f"{str(dtype).split('.')[-1]}_"
        + f"thaw_{'__'.join(sorted({n.rpartition('.')[0].replace('.', '_') for n in thaw})) if thaw else 'all'}"
        + f"_{attn_fn}",
    )
    def test_fused_sdpa_transformer_tensor_parallel(self, thaw_params, is_seq_parallel, dtype, exp_cnts, attn_fn):
        """Test FusedSDPA Tensor Parallel with various requires_grad patterns focused on attention layers"""
        # Disable dropout to facilitate single gpu to multi-device comparison
        # Disable weight-tying to enable more fine-tuning configurations
        model_args = ModelArgs(dropout_p=0.0, weight_tying=False)
        model = self._setup_single_gpu_fused_model(
            model_args, dtype, attn_fn
        )  # Step 1: Initialize single-gpu models with FusedSDPA.
        model_tp = self._setup_tp_fused_model(
            model,
            is_seq_parallel,
            dtype,
        )  # Step 2: Setup tp model, place onto device mesh.
        optim, optim_tp = self._setup_optimizer(model, model_tp)  # Step 3: Setup optimizers for both models
        DTensorFusedSDPATest._thaw_params(thaw_params, model, model_tp)  # Step 4: set `requires_grad` patterns

        # Initialize input and make sure all ranks have the same input.
        inp_size = [8, 8]  # [batch_size, seq_len]
        if is_seq_parallel:
            assert inp_size[1] % self.world_size == 0

        torch.manual_seed(0)
        inp = torch.randint(model_args.vocab_size, inp_size, device=self.device_type)
        output, output_tp = self._validate_fwd(model, model_tp, inp, check_comms=False)
        self._validate_bwd(model, model_tp, output, output_tp, exp_cnts.bwd, check_comms=True)
        self._validate_optim_step(model, model_tp, optim, optim_tp, exp_cnts.optim, check_comms=True)

    @pytest.mark.skipif(is_pytest_mode_lazy(), reason="FSDP test is not supported in lazy mode")
    @with_comms
    @skip_unless_torch_gpu
    @parametrize("attn_fn", [FusedSDPA.apply, F.scaled_dot_product_attention])
    def test_fused_sdpa_transformer_fsdp(self, attn_fn):
        """Compares losses across training runs between FSDP and non-FSDP models"""
        torch.manual_seed(42)
        bsz, seq_ln = 8, 8
        model_args = ModelArgs(n_layers=3, dropout_p=0.0, weight_tying=True)
        model = FusedTransformerHPU(model_args, attn_fn)  # transformer with FusedSDPA attention
        ref_model = copy.deepcopy(model).to(self.device_type)
        # ideally we should replicate the module across ranks
        # but on HPU it returns a RuntimeError: habana active device: 3 != 0
        # because of this we cannot set rank wise seeds below
        # replicate(
        #     ref_model,
        #     device_ids=[self.rank],
        # )
        # torch.manual_seed(42 + self.rank + 1)

        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                fully_shard(module)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        for iter_idx in range(3):
            inp = torch.randint(0, model_args.vocab_size, (bsz, seq_ln), device=self.device_type)
            losses: list[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])


# Instantiate parametrized tests
instantiate_parametrized_tests(DTensorFusedSDPATest)
