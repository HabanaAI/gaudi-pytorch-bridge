###############################################################################
# Copyright (c) 2021-2025 Intel Corporation
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


import importlib
import io
import json
import os
import sys
import types
from functools import partial
from typing import Any

import functorch
import habana_frameworks.torch.internal.bridge_config as bc
from habana_frameworks.torch.dynamo.debug_utils.logger import get_compile_backend_logger

import torch
from torch._dynamo import OptimizedModule
from torch._dynamo.backends.common import aot_autograd
from torch.ao.quantization.quantizer import Quantizer
from torch.export.exported_program import ExportedProgram
from torch.fx import GraphModule
from torch.fx.passes.utils.source_matcher_utils import (
    get_source_partitions,
)

from .pattern_matcher import (
    PatternMatchAndReplacer,
    get_dequant_node,
    is_node,
)
from .quantize_fsdpa import (
    handle_fsdpa_quantization,
)
from .quantize_kvcache import (
    check_kcache_or_vcache,
    handle_kvcache_quantization,
    prepare_for_inference,
    verify_kvcache_quant_effect,
)
from .torch_overwrites import _native_pt2e_quantization_interface

logger = get_compile_backend_logger()

habana_quantization_map_queue = []
export_model_record = {}
habana_pt2e_quant_context = None
param_id = 0
hash_counter = 0
json_counter = 0
use_json_counter_based_scale_filename = True


def calculate_hash(fx_graph_module):
    global hash_counter
    current_value = hash_counter
    hash_counter = hash_counter + 1
    key = hash(current_value)
    for idx, node in enumerate(fx_graph_module.graph.nodes):
        meta = node.meta.get("val", None)
        # ToDo: Add data type enum value instead of meta.dtype.itemsize
        # m.dtype is string which is not hashable due to hash randomization
        if isinstance(meta, tuple):
            for m in meta:
                key = key + hash((idx, m.dtype.itemsize, tuple(m.shape)))
        elif meta is not None:
            key = key + hash((idx, meta.dtype.itemsize, tuple(meta.shape)))
    logger.debug(f"FX graph hash : {key}")
    non_negative_key = key % 2**sys.hash_info.width
    return non_negative_key


def reset_counters():
    logger.debug("reset counters")
    global hash_counter
    hash_counter = 0
    global json_counter
    json_counter = 0


def get_scale_filename(key):
    filename = f"{key}.json"
    if use_json_counter_based_scale_filename or bc.get_pt_hpu_pt2eq_kvcq():
        global json_counter
        filename = f"pt2e_quant_scales_{json_counter}.json"
        json_counter = json_counter + 1
    return filename


def print_readable(
    module,
    print_output=True,
    include_stride=False,
    include_device=False,
    colored=False,
):
    if getattr(module, "graph", None) and (gms := module.graph.get_graph_modules()) != []:
        prefix = "Graphs:\n"
        module_code_list = [""]
        for idx, gm in enumerate(gms):
            module_code_list.append(f"graph[{idx}]:")
            module_code_list.append(
                gm.print_readable(
                    False,
                    include_stride,
                    include_device,
                    colored,
                )
            )
        module_code = "\n".join(module_code_list)
        suffix = "\nNote: Eager ops (if any) are not shown here."
        output = prefix + module_code + suffix
    elif hasattr(module, "l_print_readable"):
        output = module.l_print_readable
    else:
        output = HabanaPT2EQuantGraphManager.message_before_tracing
    if print_output:
        print(output, flush=True)
    return output


class HabanaPT2EQuantGraphManager:
    message_before_tracing = "If you haven't provided sample input during export, run the model at least once with actual input to capture the graphs."

    def __init__(self, gms: list[GraphModule] | Any):
        self.graph = gms
        self.l_graph = ""
        self.l_print_tabular = ""

    def get_graph_modules(self):
        return self.graph

    def set_loaded_graph_info(self, l_graph: str, l_print_tabular: str):
        self.l_graph = l_graph
        self.l_print_tabular = l_print_tabular

    def print_tabular(self):
        if self.graph:
            print("Graphs:")
            for idx, gm in enumerate(self.graph):
                print(f"graph[{idx}]:")
                gm.graph.print_tabular()
                print("\n")
            print("Note: Eager ops (if any) are not shown here.", flush=True)
        elif self.l_print_tabular:
            print(self.l_print_tabular, flush=True)
        else:
            print(self.message_before_tracing, flush=True)

    def __repr__(self):
        if self.graph:
            extra_repr_str = "Graphs:\n"
            for idx, gm in enumerate(self.graph):
                extra_repr_str = extra_repr_str + f"graph[{idx}]:\n{gm.graph}\n"
        elif self.l_graph:
            extra_repr_str = self.l_graph
        else:
            extra_repr_str = self.message_before_tracing
        return extra_repr_str


# ======================================================================================
# Habana's model level context manager for multi-graph PT2E-Quantization
# ======================================================================================
class HabanaPT2EQuantContext:
    """
    Habana's model level context manager for multi-graph PT2E-Quantization.
    """

    def __init__(self, model, model_key, input=None):
        super().__init__()
        self._original_model = model
        self._model_key = model_key
        self._input_for_tracing = input
        self._ep_dict = {}
        self._model = None
        self._quantizer = None
        self._quant_dtype = None
        self._args = []
        self._gm_hash = []
        self._preprocessed_gms = []
        self._prepared_gm = []
        self._converted_gm = []
        self._running_gm_cnt = 0
        self._kvcache_quant_details = {}
        self._kvcache_details_loaded = {}
        self._convert_settings = {}

    def set_quant_dtype(self):
        """
        It assumes that same quant dtype is used for all quantized nodes
        """
        converted_gms = self.get_all_transformed_gms(converted=True)
        for gm in converted_gms:
            for node in gm.graph.nodes:
                if is_node(node, "quantize_per_tensor.default"):
                    self._quant_dtype = node.args[5]
                    break
            if self._quant_dtype is not None:
                break

    def get_quant_dtype(self):
        if self._quant_dtype is None:
            self.set_quant_dtype()
        return self._quant_dtype

    def set_kvcache_quant_details(
        self,
        kvcache_quant_details=None,
        kvcache_allocation=None,
        kvcache_size=None,
        kvcache_size_prefill=None,
        kvcache_orig_dtype=None,
        kvcache_quant_dtype=None,
        loaded=False,
    ):
        if kvcache_quant_details:
            if not loaded:
                self._kvcache_quant_details = kvcache_quant_details
            else:
                self._kvcache_details_loaded = kvcache_quant_details
        else:
            if kvcache_allocation:
                self._kvcache_quant_details["kvcache_allocation"] = kvcache_allocation
            if kvcache_size:
                self._kvcache_quant_details["kvcache_size"] = kvcache_size
            if kvcache_size_prefill:
                self._kvcache_quant_details["kvcache_size_prefill"] = kvcache_size_prefill
            if kvcache_orig_dtype:
                self._kvcache_quant_details["kvcache_orig_dtype"] = kvcache_orig_dtype
            if kvcache_quant_dtype:
                self._kvcache_quant_details["kvcache_quant_dtype"] = kvcache_quant_dtype

    def get_kvcache_quant_details(self, loaded=False):
        if not loaded:
            return self._kvcache_quant_details
        else:
            return self._kvcache_details_loaded

    def set_misc_attributes(self, model, preprocessed=False, prepared=False, converted=False):
        model.traced = self._model.traced
        gms = []
        if converted or len(self._converted_gm) > 1:
            gms = self._converted_gm
        elif prepared or len(self._prepared_gm) > 1:
            gms = self._prepared_gm
        elif preprocessed or len(self._preprocessed_gms) > 1:
            gms = self._preprocessed_gms
        model.graph = HabanaPT2EQuantGraphManager(gms)
        model.print_readable = types.MethodType(print_readable, model)
        model.module = types.MethodType(lambda m: m, model)

    def record_transformed_gm(self, transformed_gm, preprocessed=False, prepared=False, converted=False):
        self._model.traced = True
        if preprocessed:
            self._preprocessed_gms.append(transformed_gm)
        elif prepared:
            self._prepared_gm.append(transformed_gm)
        elif converted:
            self._converted_gm.append(transformed_gm)

    def replace_transformed_gm(self, old_transformed_gm, new_transformed_gm, prepared=False, converted=False):
        if prepared:
            self._prepared_gm[self._prepared_gm.index(old_transformed_gm)] = new_transformed_gm
        if converted:
            self._converted_gm[self._converted_gm.index(old_transformed_gm)] = new_transformed_gm

    def get_all_transformed_gms(self, preprocessed=False, prepared=False, converted=False):
        if preprocessed:
            return self._preprocessed_gms
        elif prepared:
            return self._prepared_gm
        elif converted:
            return self._converted_gm

    def extract_graph_info_from_loaded_model(self):
        self._model.graph.set_loaded_graph_info(
            getattr(self._model, "l_graph", ""),
            getattr(self._model, "l_print_tabular", ""),
        )
        assert hasattr(self._model, "l_print_readable")

    def get_transformed_gm(self, prepared=False, converted=False):
        gm = None
        if prepared:
            logger.debug(f"prepared graph module list: len={len(self._prepared_gm)}, idx={self._running_gm_cnt}")
            if len(self._prepared_gm) > self._running_gm_cnt:
                gm = self._prepared_gm[self._running_gm_cnt]
        if converted:
            logger.debug(f"converted graph module list: len={len(self._converted_gm)}, idx={self._running_gm_cnt}")
            if len(self._converted_gm) > self._running_gm_cnt:
                gm = self._converted_gm[self._running_gm_cnt]
        self._running_gm_cnt = self._running_gm_cnt + 1
        return gm

    def record_args(self, args):
        self._args.append(args)

    def get_args_list(self):
        return self._args

    def record_hash(self, hash):
        self._gm_hash.append(hash)

    def get_hash_list(self):
        return self._gm_hash

    def initialize_ep_dict(self, ep_dict={}):
        self._ep_dict = ep_dict

    def get_ep(self, key):
        return self._ep_dict[key]

    def get_total_number_of_graphs(self):
        return self._running_gm_cnt

    def reset_graph_counter(self):
        self._running_gm_cnt = 0

    def get_input_for_tracing(self):
        return self._input_for_tracing

    def set_model(self, model):
        model.graph = getattr(model, "graph", HabanaPT2EQuantGraphManager([]))
        model.traced = getattr(model, "traced", False)
        self._model = model

    def get_original_model(self):
        import contextlib
        import io

        s = io.StringIO()
        with contextlib.redirect_stdout(s):
            print(self._original_model.graph)
        self._original_model.l_graph = s.getvalue()
        s.seek(0)
        s.truncate(0)

        with contextlib.redirect_stdout(s):
            self._original_model.graph.print_tabular()
        self._original_model.l_print_tabular = s.getvalue()
        s.seek(0)
        s.truncate(0)

        with contextlib.redirect_stdout(s):
            self._original_model.print_readable()
        self._original_model.l_print_readable = s.getvalue()
        s.seek(0)
        s.truncate(0)
        s.close()

        if hasattr(self._original_model, "graph"):
            delattr(self._original_model, "graph")
        if hasattr(self._original_model, "print_readable"):
            delattr(self._original_model, "print_readable")
        if hasattr(self._original_model, "module"):
            delattr(self._original_model, "module")
        if hasattr(self._original_model, "traced"):
            delattr(self._original_model, "traced")

        return self._original_model

    def set_quantizer(self, quantizer):
        self._quantizer = quantizer

    def get_quantizer(self):
        return self._quantizer

    def set_convert_settings(self, use_reference_representation, fold_quantize):
        self._convert_settings = {
            "use_reference_representation": use_reference_representation,
            "fold_quantize": fold_quantize,
        }

    def get_convert_settings(self):
        return self._convert_settings


# ======================================================================================
# Habana's torch.compile based graph break detector
# ======================================================================================
def check_if_pt2e_quant_backend_needed(
    f: torch.nn.Module,
    args: tuple[Any],
    kwargs: dict[str, Any] | None = None,
) -> bool:
    """
    Graph break detector for Habana's PT2E-Quantization flow.
    If graph breaks, Habana's PT2E-Quant flow is used. Else, native PT2E-Quant flow is used.
    """
    # If sample input is not specified during export, Habana's PT2E-Quant flow is used
    if args is None:
        return True

    # Next, check for user instruction, if any. 3 possibilities:
    # a. graph_break_present: False [Can be set only when user is sure about no graph breaks]
    # b. graph_break_present: True  [Can be set only when user is sure about graph breaks]
    # c. graph_break_present: unspecified
    if kwargs is not None and "graph_break_present" in kwargs:
        return kwargs["graph_break_present"]

    # Finally, try and figure out if there is any graph break.
    try:

        def detect_graph_break(model: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
            return model

        torch._dynamo.config.suppress_errors = True
        model = torch.compile(f, backend=detect_graph_break, fullgraph=True)
        model(*args)
        logger.info("......................................................................!")
        logger.info("NO GRAPH BREAK DETECTED............USING Native PT2E-QUANT FLOW.......!")
        logger.info("......................................................................!")
        return False
    except:
        logger.info("......................................................................!")
        logger.info("GRAPH BREAK DETECTED...............USING HABANA PT2E-QUANT FLOW.......!")
        logger.info("......................................................................!")
        return True


# ======================================================================================
# Habana Quantization Manager defined for torch.compile backend
# This is the module we use for actual support of quantization
# ======================================================================================
class HabanaQuantWrapperModule(torch.nn.Module):
    """
    Habana's fx-graph level context and control manager for multi-graph PT2E-Quantization.
    """

    def __init__(self, graph_module, module_key, pt2e_quant_context):
        super().__init__()
        self._module_key = module_key
        self._preprocessed = False
        self._prepared = False
        self._converted = False
        self._fx_module = graph_module
        self._fx_graph_hash = 0
        self._prepared_module = None
        self._observed_module = None
        self._converted_module = None
        self._pre_converted_module = None
        self._pt2e_quant_context = pt2e_quant_context
        self._kvcache_input_quantized = False

    def preprocess(self, *args):
        self._kvcache_input_quantized = discover_and_materialize_params(
            self._pt2e_quant_context, self._fx_module, *args
        )
        self._pt2e_quant_context.record_args(args)
        self._fx_graph_hash = calculate_hash(self._fx_module)
        self._pt2e_quant_context.record_hash(self._fx_graph_hash)
        self._preprocessed = True

    def __call__(self, *args, **kwargs):
        logger.debug(
            f"HabanaQuantWrapperModule::__call__ [{self._module_key}] ID:",
            id(self),
            f"\tpreprocessed={self._preprocessed}\tprepared={self._prepared}\tconverted={self._converted}",
        )

        assert self._pt2e_quant_context is not None
        if not self._preprocessed:
            self.preprocess(*args)

        if habana_quantization_map_queue[self._module_key] == []:
            self._pt2e_quant_context.record_transformed_gm(self._fx_module, preprocessed=True)
            return self._fx_module(*args, **kwargs)

        assert len(habana_quantization_map_queue[self._module_key]) == 1
        queue_element = habana_quantization_map_queue[self._module_key][0]
        if queue_element["task"] == "prepare_for_calibration":
            if not self._prepared:
                # Get already prepared fx graph
                self._prepared_module = self._pt2e_quant_context.get_transformed_gm(prepared=True)
                if not self._prepared_module:
                    # Apply pytorch prepare_pt2e on preprocessed fx graph
                    self._prepared_module = _native_pt2e_quantization_interface("prepare_pt2e")(
                        self._fx_module, self._pt2e_quant_context.get_quantizer()
                    )
                    self._pt2e_quant_context.record_transformed_gm(self._prepared_module, prepared=True)

                # Now we use torch.compilation with hpu_backend.
                # hpu_backend internally uses aot_autograd which extracts the forward definition of
                # observer class and replaces the observer specific call_module nodes with corresponding
                # inlined forward definitions.
                # However, as the same storage is still used for holding the observer state, the
                # result of calibration (i.e. all stat updates) remains available from the original
                # _prepared_module that we use later at conversion stage.
                with torch.no_grad():
                    self._observed_module = torch.compile(self._prepared_module, backend="hpu_backend")

                self._prepared = True

            return self._observed_module(*args, **kwargs)

        elif queue_element["task"] == "inference_after_calibration":
            if not self._converted:
                if not self._prepared:
                    if not self._kvcache_input_quantized:
                        logger.error(
                            "Attempt to convert an unprepared module!. Please use PT2E quant flow, i.e."
                            "Export -> prepare_pt2e -> calibrate -> convert_pt2e -> Ref_Quantized_Model, as recommended in"
                            "https://pytorch.org/tutorials/prototype/quantization_in_pytorch_2_0_export_tutorial.html"
                        )
                        raise
                    # Apply prepare_pt2e + convert_pt2e + kvcq pattern matching
                    self._converted_module = prepare_for_inference(
                        self._pt2e_quant_context, self._fx_module, update_scale=True
                    )
                else:
                    # Get already converted fx graph
                    self._converted_module = self._pt2e_quant_context.get_transformed_gm(converted=True)
                assert self._converted_module is not None

                if bc.get_pt_hpu_pt2eq_scale_load_path() != "":
                    logger.debug("Start quantization scale loading.")
                    load_scale(self._converted_module)

                if bc.get_pt_hpu_pt2eq_scale_dump_path() != "":
                    logger.debug("Start quantization scale dumping.")
                    dump_scale(self._converted_module, True)

                quant_dtype_is_fp8 = habana_pt2e_quant_context.get_quant_dtype() in [
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ]
                # Skip fx-graph pattern matching if quant_dtype is not fp8
                if quant_dtype_is_fp8 and bc.get_pt_hpu_pt2eq_fx_graph_pattern_matching():
                    replacer = PatternMatchAndReplacer(self._converted_module)
                    replacer.run()

                if bc.get_pt_hpu_pt2eq_fsdpa_quant() and (not bc.get_pt_hpu_pt2eq_fx_graph_pattern_matching()):
                    handle_fsdpa_quantization(self._converted_module)
                else:
                    logger.warn("Fp8 FSDPA supported only with synapse pattern matching!!!")

                # Skip torch compile if quant_dtype is not fp8
                if quant_dtype_is_fp8:
                    with torch.no_grad():
                        # Now we call hpu_inference_compiler to convert it into synapse graph.
                        self._converted_module = torch.compile(self._converted_module, backend="hpu_backend")

                self._converted = True

            if bc.get_pt_hpu_pt2eq_fx_graph_freezing():
                with torch._inductor.config.patch({"freezing": True}):
                    return self._converted_module(*args, **kwargs)
            else:
                return self._converted_module(*args, **kwargs)

        elif queue_element["task"] == "inference_after_load":
            self._preprocessed = True
            self._prepared = True

            use_export_program = bc.get_pt_hpu_pt2eq_use_export_program()
            if not self._converted:
                key = self._fx_graph_hash
                if use_export_program:
                    self._converted_module = self._pt2e_quant_context.get_ep(key).module()
                else:
                    # Apply prepare_pt2e + convert_pt2e + kvcq pattern matching
                    self._converted_module = prepare_for_inference(self._pt2e_quant_context, self._fx_module)

                    # Load scales from json file on each fx graph
                    extra_file = os.path.join(queue_element["dir_path"], get_scale_filename(key))
                    load_scale(self._converted_module, extra_file=extra_file)

                quant_dtype_is_fp8 = habana_pt2e_quant_context.get_quant_dtype() in [
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ]
                # Skip fx-graph pattern matching if quant_dtype is not fp8
                if quant_dtype_is_fp8 and bc.get_pt_hpu_pt2eq_fx_graph_pattern_matching():
                    replacer = PatternMatchAndReplacer(self._converted_module)
                    replacer.run()

                if bc.get_pt_hpu_pt2eq_fsdpa_quant() and (not bc.get_pt_hpu_pt2eq_fx_graph_pattern_matching()):
                    handle_fsdpa_quantization(self._converted_module)
                else:
                    logger.warn("Fp8 FSDPA supported only with synapse pattern matching!!!")

                assert self._converted_module is not None
                self._pt2e_quant_context.record_transformed_gm(self._converted_module, converted=True)

                # Skip torch compile if quant_dtype is not fp8
                if quant_dtype_is_fp8:
                    with torch.no_grad():
                        # We call hpu_inference_compiler to convert it into synapse graph.
                        self._converted_module = torch.compile(self._converted_module, backend="hpu_backend")

                self._converted = True

            if bc.get_pt_hpu_pt2eq_fx_graph_freezing():
                with torch._inductor.config.patch({"freezing": True}):
                    return self._converted_module(*args, **kwargs)
            else:
                return self._converted_module(*args, **kwargs)


def habana_quant_compiler_fw(
    module: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    module_key: torch.fx.GraphModule,
    pt2e_quant_context: HabanaPT2EQuantContext,
):
    """
    This function implements forward compiler backend used in Habana's PT2E quantization backend interface.
    It only sets up runtime wrapper to run real compilation once we have real tensors.
    """
    return functorch.compile.make_boxed_func(HabanaQuantWrapperModule(module, module_key, pt2e_quant_context))


def habana_quant_compiler_bw_raise(graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    """
    This function implements backward compiler backend used in Habana's PT2E quantization backend interface.
    As QAT is not supported yet, this function is now used to raise exception if backward pass compiler is called.
    """
    raise Exception("tried to call backward pass compiler in inference backend")


def habana_quant_backend(
    graph_module: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
    module_key: torch.fx.GraphModule,
    pt2e_quant_context: HabanaPT2EQuantContext,
    **kwargs,
):
    """
    This function implements interface for Habana's PT2E quantization backend.
    """
    from habana_frameworks.torch.dynamo.compile_backend import (
        config as habana_quant_backend_config,
    )
    from habana_frameworks.torch.dynamo.compile_backend.decomposition import (
        get_hpu_decompositions,
        override_composite_ops,
    )

    options = kwargs.get("options")
    with habana_quant_backend_config.patch(options), override_composite_ops():
        return aot_autograd(
            fw_compiler=habana_quant_backend_config.patch(options)(
                partial(
                    habana_quant_compiler_fw,
                    module_key=module_key,
                    pt2e_quant_context=pt2e_quant_context,
                )
            ),
            bw_compiler=habana_quant_compiler_bw_raise,
            decompositions=get_hpu_decompositions(),
            keep_inference_input_mutations=habana_quant_backend_config.keep_input_mutations,
        )(graph_module, example_inputs)


# ======================================================================================
# Habana's implementation of PT2E like multi-graph export
# Note: It uses torch.compile based approach with custom quantization backend
# ======================================================================================
def export(
    f: torch.nn.Module,
    args: tuple[Any, ...] = None,
    kwargs: dict[str, Any] | None = None,
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
    strict: bool = True,
    preserve_module_call_signature: tuple[str, ...] = (),
) -> torch.nn.Module | ExportedProgram | GraphModule:
    """
    Habana's implementation of PT2E like multi-graph export.
    Note: It uses torch.compile based approach with custom quantization backend.
    """
    logger.debug("Habana's implementation of PT2E based quantization flow: [export]")

    global export_model_record
    global habana_pt2e_quant_context

    # Check if we already have previous export result of the given module
    # If so, retrieve the same and return the previous export result

    if isinstance(f, OptimizedModule | GraphModule) and hasattr(f, "_orig_mod"):
        f = f._orig_mod

    id_model = hash((id(f), type(f).__name__))
    if id_model in export_model_record.keys():
        habana_pt2e_quant_context = export_model_record[id_model][1]
        model = export_model_record[id_model][0]
        assert getattr(model, "pt2e_quant_backend", False)
        if not getattr(model, "traced", False) and args is not None:
            model(*args)
            habana_pt2e_quant_context.set_misc_attributes(model)
        return model
    if kwargs:
        export_type = kwargs.get("export_type", "export.export_for_training")
        id_model = hash((id(f), type(f).__name__, export_type))
        if id_model in export_model_record.keys():
            model = export_model_record[id_model][0]
            assert not getattr(model, "pt2e_quant_backend", False)
            return model

    # Now that we have reached here, we need to check if habana_quant_backend
    # needs to be used or not
    use_pt2e_quant_backend = check_if_pt2e_quant_backend_needed(f, args, kwargs)

    if use_pt2e_quant_backend:
        habana_pt2e_quant_context = HabanaPT2EQuantContext(f, id_model, args)
        global habana_quantization_map_queue
        model_key = len(habana_quantization_map_queue)
        habana_quantization_map_queue = {model_key: []}
        torch._dynamo.reset()
        model = torch.compile(
            f,
            backend=partial(
                habana_quant_backend,
                module_key=model_key,
                pt2e_quant_context=habana_pt2e_quant_context,
            ),
            dynamic=False,
            options={"keep_input_mutations": True},
        )
        model.meta_hb_quant_id = model_key
        habana_pt2e_quant_context.reset_graph_counter()
        habana_pt2e_quant_context.set_model(model)
        if args is not None:
            model(*args)

        habana_pt2e_quant_context.set_misc_attributes(model, preprocessed=True)
        logger.debug(f"Graph after pt2e kind of export:\n {model.graph}")

        model.pt2e_quant_backend = True
        id_model = hash((id(f), type(f).__name__))
        export_model_record[id_model] = [model, habana_pt2e_quant_context]
        return model
    else:
        assert args, "[Native flow] input missing!"
        export_type = "export.export_for_training"
        if kwargs:
            if "graph_break_present" in kwargs:
                kwargs.pop("graph_break_present")
            if "export_type" in kwargs:
                export_type = kwargs["export_type"]
                kwargs.pop("export_type")

        if isinstance(f, OptimizedModule | GraphModule) and hasattr(f, "_orig_mod"):
            f = f._orig_mod

        model = _native_pt2e_quantization_interface(export_type)(
            f,
            args,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
            preserve_module_call_signature=preserve_module_call_signature,
        )
        logger.debug(f"Graph after pt2 export:\n {model.graph}")

        model.pt2e_quant_backend = False
        id_model = hash((id(f), type(f).__name__, export_type))
        export_model_record[id_model] = [model, None]
        return model


# ======================================================================================
# Habana's implementation of prepare_pt2e for multi-graph scenario
# ======================================================================================
def prepare_pt2e(
    model: GraphModule,
    quantizer: Quantizer,
) -> GraphModule:
    """
    Habana's implementation of prepare_pt2e for multi-graph scenario.
    """
    logger.debug("Habana's implementation of PT2E based quantization flow: [prepare_pt2e]")

    global export_model_record
    use_pt2e_quant_backend = getattr(model, "pt2e_quant_backend", False)
    if use_pt2e_quant_backend:
        reset_counters()
        # Set "prepare_pt2e" cmd for HabanaQuantWrapperModule
        global habana_quantization_map_queue
        model_key = model.meta_hb_quant_id
        habana_quantization_map_queue[model_key] = []
        habana_quantization_map_queue[model_key].append({"task": "prepare_for_calibration"})

        # If user provides example input during export() call,
        # we are sure about the availability of the preprocessed graph modules here.
        # And, we can prepare all the preprocessed graph modules in one go.
        global habana_pt2e_quant_context
        preprocessed_gms = habana_pt2e_quant_context.get_all_transformed_gms(preprocessed=True)

        if len(preprocessed_gms) > 0:
            habana_pt2e_quant_context.reset_graph_counter()
            logger.debug("Graphs after prepare_pt2e:")
            for p_gm in preprocessed_gms:
                # Preparation of each preprocessed fx_graph
                prepared_module = _native_pt2e_quantization_interface("prepare_pt2e")(p_gm, quantizer)
                habana_pt2e_quant_context.record_transformed_gm(prepared_module, prepared=True)
                logger.debug(f"{prepared_module.graph}")

        habana_pt2e_quant_context.reset_graph_counter()
        habana_pt2e_quant_context.set_model(model)
        habana_pt2e_quant_context.set_quantizer(quantizer)

        habana_pt2e_quant_context.set_misc_attributes(model, prepared=True)
        model.pt2e_quant_backend = True
        return model
    else:
        if isinstance(model, OptimizedModule | GraphModule) and hasattr(model, "_orig_mod"):
            model = model._orig_mod

        model = _native_pt2e_quantization_interface("prepare_pt2e")(model, quantizer)
        logger.debug(f"Graph after prepare_pt2e:\n {model.graph}")

        # Now we call hpu_inference_compiler to convert it into synapse graph.
        with torch.no_grad():
            model = torch.compile(model, backend="hpu_backend")

        model.pt2e_quant_backend = False
        return model


# ======================================================================================
# Habana's implementation of convert_pt2e for multi-graph scenario
# ======================================================================================
def convert_pt2e(
    model: GraphModule,
    use_reference_representation: bool = False,
    fold_quantize: bool = True,
) -> GraphModule:
    """
    Habana's implementation of convert_pt2e for multi-graph scenario.
    """
    logger.debug("Habana's implementation of PT2E based quantization flow: [convert_pt2e]")

    global export_model_record
    use_pt2e_quant_backend = getattr(model, "pt2e_quant_backend", False)
    if use_pt2e_quant_backend:
        reset_counters()
        # Set "convert_pt2e" cmd for HabanaQuantWrapperModule
        global habana_quantization_map_queue
        model_key = model.meta_hb_quant_id
        habana_quantization_map_queue[model_key] = []
        habana_quantization_map_queue[model_key].append(
            {
                "task": "inference_after_calibration",
            }
        )

        # At this point, we are sure that calibration is done.
        # So, we can convert all calibrated graph modules.
        global habana_pt2e_quant_context
        calibrated_gms = habana_pt2e_quant_context.get_all_transformed_gms(prepared=True)

        assert len(calibrated_gms) > 0, "Please run calibration before calling convert_pt2e()."

        habana_pt2e_quant_context.reset_graph_counter()
        logger.debug("Graphs after convert_pt2e:")
        for gm in calibrated_gms:
            # Conversion of each prepared + calibrated fx_graph
            converted_module = _native_pt2e_quantization_interface("convert_pt2e")(
                gm,
                use_reference_representation=use_reference_representation,
                fold_quantize=False,
            )
            habana_pt2e_quant_context.record_transformed_gm(converted_module, converted=True)
            logger.debug(f"{converted_module.graph}")

        # Find out low precision dtype
        habana_pt2e_quant_context.set_quant_dtype()

        # Try to use kv-cache quantization.
        # This takes effect only if kv-cache allocation is done as part of model forward method.
        handle_kvcache_quantization(habana_pt2e_quant_context)

        habana_pt2e_quant_context.reset_graph_counter()
        habana_pt2e_quant_context.set_model(model)
        habana_pt2e_quant_context.set_convert_settings(use_reference_representation, False)

        habana_pt2e_quant_context.set_misc_attributes(model, converted=True)
        model.pt2e_quant_backend = True
        return model
    else:
        if isinstance(model, OptimizedModule | GraphModule) and hasattr(model, "_orig_mod"):
            model = model._orig_mod

        model = _native_pt2e_quantization_interface("convert_pt2e")(
            model, use_reference_representation, fold_quantize=fold_quantize
        )
        logger.debug(f"Graph after convert_pt2e:\n {model.graph}")

        if bc.get_pt_hpu_pt2eq_fx_graph_pattern_matching():
            replacer = PatternMatchAndReplacer(model, quant_dtype_checked=False)
            replacer.run()

        # Now we call hpu_inference_compiler to convert it into synapse graph.
        with torch.no_grad():
            model = torch.compile(model, backend="hpu_backend")

        model.pt2e_quant_backend = False
        return model


def convert_to_module_name(input_str):
    output_str = input_str.replace("L['self']", "model")
    idx = output_str.find("_modules['layers']")
    if idx != -1:
        # modules layers found
        output_str = output_str.replace("_modules['layers']", "layers")
        idx1 = output_str.find("_modules['")
        idx2 = output_str.find("']")
        output_str = output_str[:idx1] + output_str[idx1 + len("_modules['") : idx2] + output_str[idx2 + len("']") :]
    else:
        # no modules layers found, skip the model in str
        output_str = output_str.replace("model.", "")

    return output_str


def create_kvcache_module_name(input_str, annotation):
    assert input_str != "" and annotation != ""
    string_array = input_str.split(".")
    assert len(string_array) >= 1

    kvcache_module_name = ".".join(string_array[:-1]) + ".self_attn." + annotation
    return kvcache_module_name


def dump_scale(module: torch.fx.GraphModule, save_to_file: bool, extra_file: str | None = None):
    graph = module.graph
    dump_json_output = {
        "GlobalRank": None,
        "LocalRank": -1,
        "Mode": "Scale",
        "Nodes": {},
    }

    module_params = module.state_dict()
    with torch.no_grad():
        for node in graph.nodes:
            dump_info = []
            nn_module_stack = node.meta.get("nn_module_stack", None)
            if is_node(node, "mm.default") or is_node(node, "addmm.default"):
                gemm_node = node
                is_addmm_node = is_node(gemm_node, "addmm.default")
                weight_idx = 2 if is_addmm_node else 1
                input_idx = 1 if is_addmm_node else 0
                weight_transpose_node = (
                    gemm_node.args[weight_idx] if is_node(gemm_node.args[weight_idx], "transpose.int") else None
                )
                weight_dequant_node = (
                    weight_transpose_node.args[0]
                    if weight_transpose_node and is_node(weight_transpose_node.args[0], "dequantize_per_tensor.default")
                    else None
                )
                # Try dequantize_per_channel for weights, if dequantize_per_tensor for weights is None
                if not weight_dequant_node:
                    weight_dequant_node = (
                        weight_transpose_node.args[0]
                        if weight_transpose_node
                        and is_node(weight_transpose_node.args[0], "dequantize_per_channel.default")
                        else None
                    )
                    if not weight_dequant_node:
                        logger.debug("dump_scale: Weight pattern match failed")
                        continue
                weight_quant_node = weight_dequant_node.args[0]
                input_dequant_node = None
                input_view_node = (
                    gemm_node.args[input_idx] if is_node(gemm_node.args[input_idx], "view.default") else None
                )
                if input_view_node and is_node(input_view_node.args[0], "dequantize_per_tensor.default"):
                    input_dequant_node = input_view_node.args[0]
                elif is_node(gemm_node.args[input_idx], "dequantize_per_tensor.default"):
                    input_dequant_node = gemm_node.args[input_idx]
                if not input_dequant_node:
                    logger.debug("dump_scale: Input pattern match failed")
                    continue
                input_quant_node = input_dequant_node.args[0]
                dump_input_scale_attr = torch.tensor(input_quant_node.args[1], device="hpu")
                if is_node(weight_quant_node, "quantize_per_tensor.default"):
                    dump_weight_scale = torch.tensor(weight_quant_node.args[1], device="hpu").item()
                else:
                    # get dequantize_per_channel values from model param
                    dump_weight_scale = module_params[weight_quant_node.args[1].name].tolist()
                dump_info = [dump_input_scale_attr.item(), dump_weight_scale]
                dump_key = list(nn_module_stack.values())[-1][0]
                dump_key = convert_to_module_name(dump_key)
                dump_json_output["Nodes"][dump_key] = {
                    "inputs": [dump_info[0]],
                    "params": {"weight": dump_info[1]},
                }
            if is_node(node, "bmm.default"):
                input0_dequant_node = get_dequant_node(node.args[0])
                input1_dequant_node = get_dequant_node(node.args[1])
                if input0_dequant_node and input1_dequant_node:
                    input0_node = (
                        input0_dequant_node.args[0]
                        if is_node(input0_dequant_node.args[0], "quantize_per_tensor.default")
                        else input0_dequant_node
                    )
                    input1_node = (
                        input1_dequant_node.args[0]
                        if is_node(input1_dequant_node.args[0], "quantize_per_tensor.default")
                        else input1_dequant_node
                    )
                    dump_input0_scale_attr = torch.tensor(input0_node.args[1], device="hpu")
                    dump_input1_scale_attr = torch.tensor(input1_node.args[1], device="hpu")
                    dump_info = [dump_input0_scale_attr, dump_input1_scale_attr]
                    dump_key = list(nn_module_stack.values())[-1][0]
                    dump_key = convert_to_module_name(dump_key)
                    dump_json_output["Nodes"][dump_key] = {
                        "inputs": [dump_info[0].item(), dump_info[1].item()],
                        "params": {},
                    }
            if is_node(node, "sdpa_recomp_fwd_non_dropout.default"):
                input0_dequant_node = get_dequant_node(node.args[0])
                input1_dequant_node = get_dequant_node(node.args[1])
                input2_dequant_node = get_dequant_node(node.args[2])
                if input0_dequant_node and input1_dequant_node and input2_dequant_node:
                    input0_node = (
                        input0_dequant_node.args[0]
                        if is_node(input0_dequant_node.args[0], "quantize_per_tensor.default")
                        else input0_dequant_node
                    )
                    input1_node = (
                        input1_dequant_node.args[0]
                        if is_node(input1_dequant_node.args[0], "quantize_per_tensor.default")
                        else input1_dequant_node
                    )
                    input2_node = (
                        input2_dequant_node.args[0]
                        if is_node(input2_dequant_node.args[0], "quantize_per_tensor.default")
                        else input2_dequant_node
                    )
                    dump_input0_scale_attr = torch.tensor(input0_node.args[1], device="hpu")
                    dump_input1_scale_attr = torch.tensor(input1_node.args[1], device="hpu")
                    dump_input2_scale_attr = torch.tensor(input2_node.args[1], device="hpu")
                    dump_info = [
                        dump_input0_scale_attr,
                        dump_input1_scale_attr,
                        dump_input2_scale_attr,
                    ]
                    dump_key = list(nn_module_stack.values())[-1][0]
                    dump_key = convert_to_module_name(dump_key)
                    dump_json_output["Nodes"][dump_key] = {
                        "inputs": [
                            dump_info[0].item(),
                            dump_info[1].item(),
                            dump_info[2].item(),
                        ],
                        "params": {},
                    }
            if is_node(node, "full.default"):
                logger.debug(f"Found full.default node: {node.name}")
                from .quantize_kvcache import search_node

                assert len(node.users) == 1
                full_user_node = next(iter(node.users), None)
                if is_node(full_user_node, "copy.default") and is_node(
                    full_user_node.args[1], "quantize_per_tensor.default"
                ):
                    copy_src_quant_node = full_user_node.args[1]
                    nn_module_stack = copy_src_quant_node.meta.get("nn_module_stack", None)

                    result, k_proj_v_proj_node_meta = check_kcache_or_vcache(copy_src_quant_node.args[0])
                    assert result in ("k_cache", "v_cache")

                    if not nn_module_stack:
                        assert k_proj_v_proj_node_meta
                        nn_module_stack = k_proj_v_proj_node_meta

                    dump_key = list(nn_module_stack.values())[-1][0]
                    dump_key = convert_to_module_name(dump_key)
                    dump_key = create_kvcache_module_name(dump_key, f"prompt.{result}")

                    dump_input1_scale_attr = torch.tensor(copy_src_quant_node.args[1], device="hpu")
                    dump_input0_scale_attr = torch.ones_like(dump_input1_scale_attr)
                    dump_info = [dump_input0_scale_attr, dump_input1_scale_attr]
                    dump_json_output["Nodes"][dump_key] = {
                        "inputs": [dump_info[0].item(), dump_info[1].item()],
                        "params": {},
                    }
            if is_node(node, "index_copy.default"):
                logger.debug(f"Found index_copy.default node: {node.name}")
                from .quantize_kvcache import search_node

                result = search_node(node.args[3], "quantize_per_tensor.default")
                if result["found"]:
                    input3_quant_node = result["fx_node"]
                    nn_module_stack = input3_quant_node.meta.get("nn_module_stack", None)

                    result, k_proj_v_proj_node_meta = check_kcache_or_vcache(input3_quant_node.args[0])
                    assert result in ("k_cache", "v_cache")

                    if not nn_module_stack:
                        assert k_proj_v_proj_node_meta
                        nn_module_stack = k_proj_v_proj_node_meta

                    dump_key = list(nn_module_stack.values())[-1][0]
                    dump_key = convert_to_module_name(dump_key)
                    dump_key = create_kvcache_module_name(dump_key, result)

                    dump_input3_scale_attr = torch.tensor(input3_quant_node.args[1], device="hpu")
                    dump_inputs_scale_attr = torch.ones_like(dump_input3_scale_attr)
                    dump_info = [
                        dump_inputs_scale_attr,
                        dump_inputs_scale_attr,
                        dump_inputs_scale_attr,
                        dump_input3_scale_attr,
                    ]
                    dump_json_output["Nodes"][dump_key] = {
                        "inputs": [
                            dump_info[0].item(),
                            dump_info[1].item(),
                            dump_info[2].item(),
                            dump_info[3].item(),
                        ],
                        "params": {},
                    }
                    for user_node in node.users:
                        if is_node(user_node, "dequantize_per_tensor.default"):
                            dump_output_scale_attr = torch.tensor(user_node.args[1], device="hpu")
                            output_dump_info = [dump_output_scale_attr]
                            dump_json_output["Nodes"][dump_key]["outputs"] = [output_dump_info[0].item()]
                            break

    graph.lint()
    if save_to_file:
        file_path = os.getenv("PT2E_QUANT_SCALE_DUMP_PATH", "0")
        if ".json" not in file_path:
            file_path = file_path + "/pt2e_quant_dumped_scale.json"
        if extra_file is not None:
            file_path = extra_file
        with open(file_path, "w") as json_file:
            json.dump(dump_json_output, json_file, indent=4)
        logger.debug(f"PT2E scale info dumped to file: {file_path}")

    return dump_json_output


def load_scale(module: torch.fx.GraphModule, scale_info_json=None, extra_file: str | None = None):
    graph = module.graph

    if not scale_info_json:
        file_path = os.getenv("PT2E_QUANT_SCALE_LOAD_PATH", "0")
        if extra_file is not None:
            file_path = extra_file
        assert file_path is not None
        with open(file_path) as file:
            scale_info_json = json.load(file)

    count = 0
    module_params = module.state_dict()
    with torch.no_grad():
        for node in graph.nodes:
            if is_node(node, "bmm.default"):
                input0_dequant_node = get_dequant_node(node.args[0])
                input1_dequant_node = get_dequant_node(node.args[1])
                if input0_dequant_node is not None:
                    logger.debug("Dequant node found for input0")
                if input1_dequant_node is not None:
                    logger.debug("Dequant node found for input1")
                if input0_dequant_node and input1_dequant_node:
                    count = count + 1
                    nn_module_stack = node.meta.get("nn_module_stack", None)
                    module_name = list(nn_module_stack.values())[-1][0]
                    module_name = convert_to_module_name(module_name)
                    input0_dequant_node_args = list(input0_dequant_node.args)
                    input1_dequant_node_args = list(input1_dequant_node.args)
                    input0_dequant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][0]
                    input1_dequant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][1]
                    input0_dequant_node.args = tuple(input0_dequant_node_args)
                    input1_dequant_node.args = tuple(input1_dequant_node_args)
                    input0_quant_node = input0_dequant_node.args[0]
                    if is_node(input0_quant_node, "quantize_per_tensor.default"):
                        input0_quant_node_args = list(input0_quant_node.args)
                        input0_quant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][0]
                        input0_quant_node.args = tuple(input0_quant_node_args)
                    input1_quant_node = input1_dequant_node.args[0]
                    if is_node(input1_quant_node, "quantize_per_tensor.default"):
                        input1_quant_node_args = list(input1_quant_node.args)
                        input1_quant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][1]
                        input1_quant_node.args = tuple(input1_quant_node_args)
            if is_node(node, "sdpa_recomp_fwd_non_dropout.default"):
                input0_dequant_node = get_dequant_node(node.args[0])
                input1_dequant_node = get_dequant_node(node.args[1])
                input2_dequant_node = get_dequant_node(node.args[2])
                if input0_dequant_node is not None:
                    logger.debug("Dequant node found for input0")
                if input1_dequant_node is not None:
                    logger.debug("Dequant node found for input1")
                if input2_dequant_node is not None:
                    logger.debug("Dequant node found for input1")
                if input0_dequant_node and input1_dequant_node and input2_dequant_node:
                    count = count + 1
                    nn_module_stack = node.meta.get("nn_module_stack", None)
                    module_name = list(nn_module_stack.values())[-1][0]
                    module_name = convert_to_module_name(module_name)
                    input0_dequant_node_args = list(input0_dequant_node.args)
                    input1_dequant_node_args = list(input1_dequant_node.args)
                    input2_dequant_node_args = list(input2_dequant_node.args)
                    input0_dequant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][0]
                    input1_dequant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][1]
                    input2_dequant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][2]
                    input0_dequant_node.args = tuple(input0_dequant_node_args)
                    input1_dequant_node.args = tuple(input1_dequant_node_args)
                    input2_dequant_node.args = tuple(input2_dequant_node_args)
                    input0_quant_node = input0_dequant_node.args[0]
                    if is_node(input0_quant_node, "quantize_per_tensor.default"):
                        input0_quant_node_args = list(input0_quant_node.args)
                        input0_quant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][0]
                        input0_quant_node.args = tuple(input0_quant_node_args)
                    input1_quant_node = input1_dequant_node.args[0]
                    if is_node(input1_quant_node, "quantize_per_tensor.default"):
                        input1_quant_node_args = list(input1_quant_node.args)
                        input1_quant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][1]
                        input1_quant_node.args = tuple(input1_quant_node_args)
                    input2_quant_node = input2_dequant_node.args[0]
                    if is_node(input2_quant_node, "quantize_per_tensor.default"):
                        input2_quant_node_args = list(input2_quant_node.args)
                        input2_quant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][2]
                        input2_quant_node.args = tuple(input2_quant_node_args)
                else:
                    logger.debug("FSDPA dequant inputs not found")
            if is_node(node, "mm.default") or is_node(node, "addmm.default"):
                gemm_node = node
                is_addmm_node = is_node(gemm_node, "addmm.default")
                weight_idx = 2 if is_addmm_node else 1
                input_idx = 1 if is_addmm_node else 0
                weight_transpose_node = (
                    gemm_node.args[weight_idx] if is_node(gemm_node.args[weight_idx], "transpose.int") else None
                )
                weight_dequant_node = (
                    weight_transpose_node.args[0]
                    if weight_transpose_node and is_node(weight_transpose_node.args[0], "dequantize_per_tensor.default")
                    else None
                )
                # Try dequantize_per_channel for weights, if dequantize_per_tensor for weights is None
                if not weight_dequant_node:
                    weight_dequant_node = (
                        weight_transpose_node.args[0]
                        if weight_transpose_node
                        and is_node(weight_transpose_node.args[0], "dequantize_per_channel.default")
                        else None
                    )
                    if not weight_dequant_node:
                        logger.debug("load_scale: Weight pattern match failed")
                        continue
                weight_quant_node = weight_dequant_node.args[0]

                input_dequant_node = None
                input_view_node = (
                    gemm_node.args[input_idx] if is_node(gemm_node.args[input_idx], "view.default") else None
                )
                if input_view_node and is_node(input_view_node.args[0], "dequantize_per_tensor.default"):
                    input_dequant_node = input_view_node.args[0]
                elif is_node(gemm_node.args[input_idx], "dequantize_per_tensor.default"):
                    input_dequant_node = gemm_node.args[input_idx]

                if not input_dequant_node:
                    logger.debug("load_scale: Input pattern match failed")
                    continue

                count = count + 1
                nn_module_stack = node.meta.get("nn_module_stack", None)
                module_name = list(nn_module_stack.values())[-1][0]
                module_name = convert_to_module_name(module_name)
                if is_node(weight_quant_node, "quantize_per_tensor.default"):
                    weight_quant_node_args = list(weight_quant_node.args)
                    weight_quant_node_args[1] = scale_info_json["Nodes"][module_name]["params"]["weight"]
                    weight_quant_node.args = tuple(weight_quant_node_args)
                    weight_dequant_node_args = list(weight_dequant_node.args)
                    weight_dequant_node_args[1] = scale_info_json["Nodes"][module_name]["params"]["weight"]
                    weight_dequant_node.args = tuple(weight_dequant_node_args)
                else:
                    # get dequantize_per_channel values and update module params
                    scales_list = scale_info_json["Nodes"][module_name]["params"]["weight"]
                    scales_tensor = torch.tensor(scales_list, device="hpu")
                    offsets_tensor = torch.zeros_like(scales_tensor, device="hpu")
                    module_params[weight_quant_node.args[1].name] = scales_tensor
                    module_params[weight_quant_node.args[2].name] = offsets_tensor

                    with module.graph.inserting_before(weight_quant_node):
                        scales_attr_name = weight_quant_node.args[1].name + str(count)
                        setattr(module, scales_attr_name, torch.nn.parameter.Parameter(scales_tensor.detach()))
                        new_scales_attr_node = module.graph.create_node("get_attr", scales_attr_name)
                        weight_quant_node.replace_input_with(weight_quant_node.args[1], new_scales_attr_node)
                        weight_dequant_node.replace_input_with(weight_dequant_node.args[1], new_scales_attr_node)

                        offsets_attr_name = weight_quant_node.args[2].name + str(count)
                        setattr(module, offsets_attr_name, torch.nn.parameter.Parameter(offsets_tensor.detach()))
                        new_offsets_attr_node = module.graph.create_node("get_attr", offsets_attr_name)
                        weight_quant_node.replace_input_with(weight_quant_node.args[2], new_offsets_attr_node)
                        weight_dequant_node.replace_input_with(weight_dequant_node.args[2], new_offsets_attr_node)

                input_quant_node = input_dequant_node.args[0]
                input_quant_node_args = list(input_quant_node.args)
                input_quant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][0]
                input_quant_node.args = tuple(input_quant_node_args)
                input_dequant_node_args = list(input_dequant_node.args)
                input_dequant_node_args[1] = scale_info_json["Nodes"][module_name]["inputs"][0]
                input_dequant_node.args = tuple(input_dequant_node_args)
            if is_node(node, "full.default"):
                logger.debug(f"Found full.default node: {node.name}")
                from .quantize_kvcache import search_node

                assert len(node.users) == 1
                full_user_node = next(iter(node.users), None)
                if is_node(full_user_node, "copy.default") and is_node(
                    full_user_node.args[1], "quantize_per_tensor.default"
                ):
                    copy_src_quant_node = full_user_node.args[1]
                    nn_module_stack = copy_src_quant_node.meta.get("nn_module_stack", None)

                    result, k_proj_v_proj_node_meta = check_kcache_or_vcache(copy_src_quant_node.args[0])
                    assert result in ("k_cache", "v_cache")

                    if not nn_module_stack:
                        assert k_proj_v_proj_node_meta
                        nn_module_stack = k_proj_v_proj_node_meta

                    load_key = list(nn_module_stack.values())[-1][0]
                    load_key = convert_to_module_name(load_key)
                    load_key = create_kvcache_module_name(load_key, f"prompt.{result}")

                    count = count + 1
                    input1_quant_node_args = list(copy_src_quant_node.args)
                    input1_quant_node_args[1] = scale_info_json["Nodes"][load_key]["inputs"][1]
                    copy_src_quant_node.args = tuple(input1_quant_node_args)
            if is_node(node, "index_copy.default"):
                logger.debug(f"Found index_copy.default node: {node.name}")
                from .quantize_kvcache import search_node

                result = search_node(node.args[3], "quantize_per_tensor.default")
                if result["found"]:
                    input3_quant_node = result["fx_node"]
                    nn_module_stack = input3_quant_node.meta.get("nn_module_stack", None)

                    result, k_proj_v_proj_node_meta = check_kcache_or_vcache(input3_quant_node.args[0])
                    assert result in ("k_cache", "v_cache")

                    if not nn_module_stack:
                        assert k_proj_v_proj_node_meta
                        nn_module_stack = k_proj_v_proj_node_meta

                    load_key = list(nn_module_stack.values())[-1][0]
                    load_key = convert_to_module_name(load_key)
                    load_key = create_kvcache_module_name(load_key, result)

                    count = count + 1
                    input3_quant_node_args = list(input3_quant_node.args)
                    input3_quant_node_args[1] = scale_info_json["Nodes"][load_key]["inputs"][3]
                    input3_quant_node.args = tuple(input3_quant_node_args)

                    for user_node in list(node.users):
                        if is_node(user_node, "dequantize_per_tensor.default"):
                            output_dequant_node_args = list(user_node.args)
                            output_dequant_node_args[1] = scale_info_json["Nodes"][load_key]["outputs"][0]
                            user_node.args = tuple(output_dequant_node_args)

        assert count == len(scale_info_json["Nodes"])
        module.graph.lint()
        module.recompile()


def get_dir_file_name(path):
    """
    From user provided path, derive the folder and file names.
    """
    dir_name = os.path.dirname(path)
    if dir_name == "":
        dir_name = os.path.join(dir_name, "pt2e_quant")
    filename = os.path.basename(path)
    if filename == "":
        filename = "pt2e_quant_model.pt2"
    filename = os.path.join(dir_name, filename)

    return dir_name, filename


# ======================================================================================
# Habana's implementation of torch.export.save for multi-graph scenario
# ======================================================================================
def save_pt2e(
    model: Any,  # e.g. torch.nn.Module, GraphModule, ExportedProgram
    f: str | os.PathLike | io.BytesIO,
    *,
    extra_files: dict[str, Any] | None = None,
    opset_version: dict[str, int] | None = None,
) -> None:
    """
    Habana's implementation of torch.export.save for multi-graph scenario.
    """
    logger.debug("Habana's implementation of PT2E based quantization flow: [save_pt2e]")

    use_pt2e_quant_backend = getattr(model, "pt2e_quant_backend", False)
    logger.debug(f"[save_pt2e] Habana PT2E Quant backend used? {use_pt2e_quant_backend}")

    dir_name, saved_model_filename = get_dir_file_name(f)
    os.makedirs(dir_name, exist_ok=True)

    if use_pt2e_quant_backend:
        reset_counters()
        global habana_pt2e_quant_context

        quant_dtype = habana_pt2e_quant_context.get_quant_dtype()
        if quant_dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]:
            raise NotImplementedError(
                f"[save_pt2e] Serialization/Deserialization in multi-graph scenario not implemented for quant_dtype: {quant_dtype}"
            )

        org_model = habana_pt2e_quant_context.get_original_model()
        org_model.pt2e_quant_backend = True

        # save original model
        # ToDo: explore to save org model with out state dict
        torch.save(org_model, saved_model_filename)

        use_export_program = bc.get_pt_hpu_pt2eq_use_export_program()
        if not use_export_program:
            pt2e_quant_info = {}
            # save quantizer used
            pt2e_quant_info["quantizer"] = habana_pt2e_quant_context.get_quantizer()
            # save convert_pt2e settings used
            pt2e_quant_info["convert_settings"] = habana_pt2e_quant_context.get_convert_settings()
            # save kv-cache quant details
            pt2e_quant_info["kvcache_quant_details"] = habana_pt2e_quant_context.get_kvcache_quant_details()
            pt2e_quant_info_filename = os.path.join(dir_name, "pt2e_quant_info.pt2")
            torch.save(pt2e_quant_info, pt2e_quant_info_filename)
            # save scale information
            fx_module_hashkeys = habana_pt2e_quant_context.get_hash_list()
            converted_gms = habana_pt2e_quant_context.get_all_transformed_gms(converted=True)
            for hashkey, gm_2 in zip(fx_module_hashkeys, converted_gms, strict=False):
                # get scale information
                scale_info_json = dump_scale(gm_2, False)
                scale_filename = os.path.join(dir_name, get_scale_filename(hashkey))
                with open(scale_filename, "w") as json_file:
                    json.dump(scale_info_json, json_file, indent=4)
            logger.debug("[save_pt2e] Completed.")
        else:
            # save export programs
            fx_module_hashkeys = habana_pt2e_quant_context.get_hash_list()
            converted_gms = habana_pt2e_quant_context.get_all_transformed_gms(converted=True)
            args_list = habana_pt2e_quant_context.get_args_list()
            assert len(args_list) == len(fx_module_hashkeys)
            assert len(args_list) == len(converted_gms)
            for hashkey, gm_2, arg in zip(fx_module_hashkeys, converted_gms, args_list, strict=False):
                # get export program
                exported_fx_graph = None
                with torch.no_grad():
                    exported_fx_graph = torch.export.export(gm_2, arg)
                logger.debug(f"exported program: {exported_fx_graph}")

                assert exported_fx_graph is not None
                # clear export program example inputs to reduce export program disk size
                exported_fx_graph._example_inputs = ()

                # save each exported converted fx graph
                exported_program_filename = os.path.join(dir_name, f"{hashkey}.pt2")
                with torch.no_grad():
                    _native_pt2e_quantization_interface("save_pt2e")(exported_fx_graph, exported_program_filename)
            # save hashkeys
            hashkeys_filename = os.path.join(dir_name, "hashkeys.pt2")
            torch.save(fx_module_hashkeys, hashkeys_filename)
            logger.debug("[save_pt2e] Export Program Dump Completed.")
    else:
        if isinstance(model, OptimizedModule | GraphModule) and hasattr(model, "_orig_mod"):
            model = model._orig_mod

        with torch.no_grad():
            _native_pt2e_quantization_interface("save_pt2e")(
                model, saved_model_filename, extra_files=extra_files, opset_version=opset_version
            )
        return


# ======================================================================================
# Habana's implementation of torch.export.load for multi-graph scenario
# ======================================================================================
def load_pt2e(
    f: str | os.PathLike | io.BytesIO,
    *,
    extra_files: dict[str, Any] | None = None,
    expected_opset_version: dict[str, int] | None = None,
) -> Any:
    """
    Habana's implementation of torch.export.load for multi-graph scenario.
    """
    logger.debug("Habana's implementation of PT2E based quantization flow: [load_pt2e]")

    dir_name, saved_model_filename = get_dir_file_name(f)

    try:
        # load original model
        # weights_only flag is True by default from PT2.6 onwards, Hence explicitly setting it as False
        model = torch.load(saved_model_filename, weights_only=False).to(device="hpu")
        use_pt2e_quant_backend = getattr(model, "pt2e_quant_backend", False)
    except:
        use_pt2e_quant_backend = False

    logger.debug(f"[load_pt2e] Habana PT2E Quant backend used? {use_pt2e_quant_backend}")

    if use_pt2e_quant_backend:
        reset_counters()
        org_model = model
        id_org_model = hash((id(org_model), type(org_model).__name__))

        # ToDo: explore to save org model with out state dict
        # Currently state dict is deleted after loading the model
        del org_model._state_dict_pre_hooks

        global habana_pt2e_quant_context
        habana_pt2e_quant_context = HabanaPT2EQuantContext(f, id_org_model)

        global habana_quantization_map_queue
        model_key = len(habana_quantization_map_queue)
        habana_quantization_map_queue = {model_key: []}
        torch._dynamo.reset()
        model = torch.compile(
            org_model,
            backend=partial(
                habana_quant_backend,
                module_key=model_key,
                pt2e_quant_context=habana_pt2e_quant_context,
            ),
            dynamic=False,
            options={"keep_input_mutations": True},
        )
        model.meta_hb_quant_id = model_key
        habana_pt2e_quant_context.set_model(model)
        habana_pt2e_quant_context.extract_graph_info_from_loaded_model()

        use_export_program = bc.get_pt_hpu_pt2eq_use_export_program()
        if use_export_program:
            # load all exported converted fx graphs and create a dictionary
            # try loading hashkeys
            hashkeys_filename = os.path.join(dir_name, "hashkeys.pt2")
            fx_module_hashkeys = torch.load(hashkeys_filename, weights_only=False)
            assert len(fx_module_hashkeys) != 0
            ep_dict = {}
            for key in fx_module_hashkeys:
                exported_program_filename = os.path.join(dir_name, f"{key}.pt2")
                with torch.no_grad():
                    exported_fx_graph = _native_pt2e_quantization_interface("load_pt2e")(exported_program_filename)
                exported_fx_graph._example_inputs = ()
                ep_dict[key] = exported_fx_graph
                logger.debug(f"loading exported program from file {exported_program_filename}")
            habana_pt2e_quant_context.initialize_ep_dict(ep_dict)
        else:
            pt2e_quant_info_filename = os.path.join(dir_name, "pt2e_quant_info.pt2")
            pt2e_quant_info = torch.load(pt2e_quant_info_filename, weights_only=False)
            # load quantizer for original model if export use scale
            quantizer = pt2e_quant_info["quantizer"]
            assert quantizer is not None
            habana_pt2e_quant_context.set_quantizer(quantizer)
            # load convert_pt2e settings
            convert_settings = pt2e_quant_info["convert_settings"]
            assert convert_settings is not None
            habana_pt2e_quant_context.set_convert_settings(
                convert_settings["use_reference_representation"],
                convert_settings["fold_quantize"],
            )
            # load kv-cache quant details
            kvcache_quant_details = pt2e_quant_info["kvcache_quant_details"]
            assert kvcache_quant_details is not None
            habana_pt2e_quant_context.set_kvcache_quant_details(kvcache_quant_details, loaded=True)

        habana_quantization_map_queue[model_key].append({"task": "inference_after_load", "dir_path": dir_name})

        model.pt2e_quant_backend = True
        logger.debug("LOADING All export program COMPLETED !!!")

        habana_pt2e_quant_context.set_misc_attributes(model, converted=True)
        return model
    else:
        with torch.no_grad():
            model = _native_pt2e_quantization_interface("load_pt2e")(
                saved_model_filename,
                extra_files=extra_files,
                expected_opset_version=expected_opset_version,
            )

        logger.debug(f"Graph after pt2e load:\n {model.graph}")

        # Now we call hpu_inference_compiler to convert it into synapse graph.
        with torch.no_grad():
            model = torch.compile(model.module(), backend="hpu_backend")

        model.module = types.MethodType(lambda m: m, model)
        model.pt2e_quant_backend = False
        return model


# ======================================================================================
# Freeze parameters for linear op, as is done in case of torch.export()
# ======================================================================================
def preprocess_linears(placeholder_map, model: torch.fx.GraphModule, tupled_args, *args):
    """
    This function un-lifts linear node parameters.
    """
    linear_module_partitions = get_source_partitions(model.graph, [torch.nn.Linear, torch.nn.functional.linear])

    if len(linear_module_partitions) == 0:
        return

    global param_id
    model_changed = False
    for module_or_fn_type, partitions in linear_module_partitions.items():
        if module_or_fn_type in (torch.nn.Linear, torch.nn.functional.linear):
            for p in partitions:
                weight_node = None
                bias_node = None
                compute_node = None
                for node in p.nodes:
                    if node.op == "call_function":
                        if node.target.__name__ == "linear.default":
                            weight_node = node.args[1]
                            if len(node.args) > 2:
                                bias_node = node.args[2]
                            compute_node = node
                            break
                        elif node.target.__name__ == "addmm.default":
                            weight_node = node.args[0]
                            bias_node = node.args[2]
                            compute_node = node
                            break
                        elif node.target.__name__ == "mm.default":
                            weight_node = node.args[1]
                            compute_node = node
                            break

                if compute_node is None:
                    logger.warn("Ignoring cases, where linear is decomposed into (t + bmm).")
                    continue

                assert weight_node is not None

                # Now let's follow addmm node inputs till we find nodes on partition list to get
                # original primals. We do that to go before any view/t ops we could have here.
                # We assume that all ops in such chain take single input.
                if weight_node in p.input_nodes:
                    # Already a primal.
                    weight_node_first_user = compute_node
                else:
                    weight_node_first_user = weight_node
                    while True:
                        if weight_node in p.input_nodes:
                            break
                        assert len(weight_node.args) >= 1
                        weight_node_first_user = weight_node
                        weight_node = weight_node.args[0]

                if bias_node is not None:
                    if bias_node in p.input_nodes:
                        # Already a primal.
                        bias_node_first_user = compute_node
                    else:
                        bias_node_first_user = bias_node
                        while True:
                            if bias_node in p.input_nodes:
                                break
                            assert len(bias_node.args) >= 1
                            bias_node_first_user = bias_node
                            bias_node = bias_node.args[0]

                # Now, clone original parameters primals into actual params within self and add
                # FX graph nodes to use them instead of inputs.
                with model.graph.inserting_before(weight_node_first_user):
                    model_changed = model_changed or True
                    attr_name = "_param_constant_l" + str(param_id)
                    param_tensor = tupled_args[placeholder_map[weight_node.name]]
                    setattr(
                        model,
                        attr_name,
                        torch.nn.parameter.Parameter(param_tensor.detach()),
                    )
                    new_attr_node = model.graph.create_node("get_attr", attr_name)
                    weight_node_first_user.replace_input_with(weight_node, new_attr_node)
                    param_id = param_id + 1

                    # Fix source code meta for annotations detection.
                    new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                    new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                    new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                    new_attr_node.meta["val"] = compute_node.meta.get("val", None)

                if bias_node is not None:
                    with model.graph.inserting_before(bias_node_first_user):
                        model_changed = model_changed or True
                        attr_name = "_param_constant_l" + str(param_id)
                        param_tensor = tupled_args[placeholder_map[bias_node.name]]
                        setattr(
                            model,
                            attr_name,
                            torch.nn.parameter.Parameter(param_tensor.detach()),
                        )
                        new_attr_node = model.graph.create_node("get_attr", attr_name)
                        bias_node_first_user.replace_input_with(bias_node, new_attr_node)
                        param_id = param_id + 1

                        # Fix source code meta for annotations detection.
                        new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                        new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                        new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                        new_attr_node.meta["val"] = compute_node.meta.get("val", None)

    if model_changed:
        model.graph.lint()
        model.recompile()


# ======================================================================================
# Freeze parameters for conv op, as is done in case of torch.export()
# ======================================================================================
def preprocess_convs(placeholder_map, model: torch.fx.GraphModule, tupled_args):
    """
    This function un-lifts convolution node parameters.
    """
    conv_module_partitions = get_source_partitions(model.graph, [torch.nn.Conv2d, torch.nn.functional.conv2d])

    if len(conv_module_partitions) == 0:
        return

    # TODO add support for convs without bias.

    global param_id
    for module_or_fn_type, partitions in conv_module_partitions.items():
        if module_or_fn_type in (torch.nn.Conv2d, torch.nn.functional.conv2d):
            for p in partitions:
                weight_node = None
                bias_node = None
                compute_node = None
                for node in p.nodes:
                    # Find addmm node and get first input. We cannot use partitions input list
                    # to get params as it is changing inputs order.
                    if node.op == "call_function" and node.target.__name__ == "convolution.default":
                        weight_node = node.args[1]
                        bias_node = node.args[2]
                        compute_node = node
                        break

                assert weight_node is not None and compute_node is not None

                # Now let's follow addmm node inputs till we find nodes on partition list to get
                # original primals. We do that to go before any view/t ops we could have here.
                # We assume that all ops in such chain take single input.
                if weight_node in p.input_nodes:
                    # Already a primal.
                    weight_node_first_user = compute_node
                else:
                    weight_node_first_user = weight_node
                    while True:
                        if weight_node in p.input_nodes:
                            break
                        assert len(weight_node.args) >= 1
                        weight_node_first_user = weight_node
                        weight_node = weight_node.args[0]

                if bias_node in p.input_nodes:
                    # Already a primal.
                    bias_node_first_user = compute_node
                else:
                    bias_node_first_user = bias_node
                    while True:
                        if bias_node in p.input_nodes:
                            break
                        assert len(bias_node.args) >= 1
                        bias_node_first_user = bias_node
                        bias_node = bias_node.args[0]

                # Now, clone original parameters primals into actual params within self and add
                # FX graph nodes to use them instead of inputs.
                with model.graph.inserting_before(weight_node_first_user):
                    attr_name = "_param_constant_c" + str(param_id)
                    param_tensor = tupled_args[placeholder_map[weight_node.name]]
                    setattr(
                        model,
                        attr_name,
                        torch.nn.parameter.Parameter(param_tensor.detach()),
                    )
                    new_attr_node = model.graph.create_node("get_attr", attr_name)
                    weight_node_first_user.replace_input_with(weight_node, new_attr_node)
                    param_id = param_id + 1

                    # Fix source code meta for annotations detection.
                    new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                    new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                    new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                    new_attr_node.meta["val"] = compute_node.meta.get("val", None)

                with model.graph.inserting_before(bias_node_first_user):
                    attr_name = "_param_constant_c" + str(param_id)
                    param_tensor = tupled_args[placeholder_map[bias_node.name]]
                    setattr(
                        model,
                        attr_name,
                        torch.nn.parameter.Parameter(param_tensor.detach()),
                    )
                    new_attr_node = model.graph.create_node("get_attr", attr_name)
                    bias_node_first_user.replace_input_with(bias_node, new_attr_node)
                    param_id = param_id + 1

                    # Fix source code meta for annotations detection.
                    new_attr_node.meta["source_fn_stack"] = compute_node.meta.get("source_fn_stack", None)
                    new_attr_node.meta["stack_trace"] = compute_node.meta.get("stack_trace", None)
                    new_attr_node.meta["tensor_meta"] = compute_node.meta.get("tensor_meta", None)
                    new_attr_node.meta["val"] = compute_node.meta.get("val", None)

    model.graph.lint()
    model.recompile()


# ======================================================================================
# Change FX graph so that it resembles one that would be generated by torch.export()
# ======================================================================================
def discover_and_materialize_params(pt2eq_context, model: torch.fx.GraphModule, *args):
    """
    This function changes FX graph so that it resembles one that would be generated by torch.export().
    """
    kvcache_input_quantized = verify_kvcache_quant_effect(pt2eq_context, model)

    # Get placeholder map from FX graph.
    placeholder_map = {}
    placeholder_count = 0
    for node in model.graph.nodes:
        if node.op == "placeholder":
            placeholder_map[node.name] = placeholder_count
            placeholder_count = placeholder_count + 1

    tupled_args = tuple(args)

    # Handle following custom linear modules in deepspeed
    def handle_custom_linear_modules(model):
        for node in model.graph.nodes:
            source_fn_stack = node.meta.get("source_fn_stack", None)
            nn_module_stack = node.meta.get("nn_module_stack", None)
            if source_fn_stack is not None and nn_module_stack is not None:
                node.meta["source_fn_stack_original"] = source_fn_stack
                nn_module_stack_last_value = str(list(nn_module_stack.values())[-1])
                custom_linear_modules = [
                    "LinearLayer",
                    "LinearAllreduce",
                    "ScopedLinearAllReduce",
                    "LmHeadLinearAllreduce",
                ]
                if any(substring in nn_module_stack_last_value for substring in custom_linear_modules):
                    del source_fn_stack[-1]
                    source_fn_stack.append((list(nn_module_stack.keys())[-1], torch.nn.Linear))
                    node.meta["source_fn_stack"] = source_fn_stack

    # Due to custom linear modules in deepspeed, "source_fn_stack" node meta
    # of post-decomposition "mm" nodes does not include the original source
    # information. Hence, pytorch's get_source_partitions() utility fails to
    # to identify the "mm" nodes that originally belong to linear modules.
    # Till we have a proper 'parameter freezing' mechanism in place, we can
    # use "nn_module_stack" node meta to refill the missing information.
    if importlib.util.find_spec("deepspeed") and os.getenv("WORLD_SIZE", "0") != "0":
        handle_custom_linear_modules(model)

    preprocess_linears(placeholder_map, model, tupled_args, *args)
    preprocess_convs(placeholder_map, model, tupled_args)

    return kvcache_input_quantized
