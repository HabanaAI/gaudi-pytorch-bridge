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

import datetime
import os
import pickle
import threading
from collections import deque
from functools import wraps
from os import environ, path
from typing import IO, Any, BinaryIO, Dict, Generator, Optional, Tuple, Union

import habana_frameworks.torch.hpu as ht
import habana_frameworks.torch.hpu.random as rand_hpu
import habana_frameworks.torch.internal.bridge_config as bc
import habana_frameworks.torch.utils.debug as htdebug
import torch
from habana_frameworks.torch.utils import _weights_only_unpickler
from habana_frameworks.torch.utils.internal import is_lazy
from torch.distributed.constants import default_pg_timeout
from torch.functional import Tensor

LAZY_DEFAULT_PROTOCOL = 4

_name_stack = deque()
_module_dict = dict()
_lock = threading.Lock()


def _is_inference():
    return environ.get("PT_HPU_INFERENCE_MODE")


def _names_hook_already_registered(module):
    if hasattr(module, "names_hook") and module.names_hook == True:
        return True
    return False


def _pre_fwd_hook(module, input):
    # handle the naming mismatch issue with a temp fix, till we
    # find a way to get unique module names from the calibration tool
    with _lock:
        try:
            if _is_inference() and _name_stack and "relu" in module.custom_name:
                ns = str(_name_stack[-1])
                if ns in _module_dict.keys():
                    _module_dict[ns] += 1
                    new_name = _name_stack[-1] + "/" + module.custom_name + "." + str(_module_dict[ns])
                else:
                    _module_dict[ns] = 0
                    new_name = _name_stack[-1] + "/" + module.custom_name if _name_stack else module.custom_name
            else:
                new_name = _name_stack[-1] + "/" + module.custom_name if _name_stack else module.custom_name

            _name_stack.append(new_name)
            htdebug._set_module_name(new_name)
        except:
            new_name = module.custom_name


def _gen_grad_hook(name):
    def grad_hook(grad):
        htdebug._set_module_name(name)
        return grad

    return grad_hook


def _post_fwd_hook(module, input, output):
    with _lock:
        module_name = _name_stack.pop()
        if _is_inference() and module_name in _module_dict.keys():
            del _module_dict[module_name]
        if (_name_stack) and len(_name_stack):
            name = _name_stack[-1]
        else:
            name = ""
        grad_name = "gradient/" + module_name
        htdebug._set_module_name(name)
        try:
            if isinstance(output, Tensor):
                if output.requires_grad and not _names_hook_already_registered(output):
                    output.register_hook(_gen_grad_hook(grad_name))
                    output.names_hook = True
            else:
                for o in output:
                    if isinstance(o, Tensor) and o.requires_grad and not _names_hook_already_registered(o):
                        o.register_hook(_gen_grad_hook(grad_name))
                        o.names_hook = True
        except:
            pass


class HpuMarker:
    pass


def _deserialize_habana_wrapper(func, marker, *args):
    return func(*args)


def overwrite_torch_functions():
    # wrap torch.distributed.distributed_c10d._get_pg_default_device
    get_pg_default_device_orig = torch.distributed.distributed_c10d._get_pg_default_device

    @wraps(torch.distributed.distributed_c10d._get_pg_default_device)
    def wrap_get_pg_default_device(group):
        backend_name = group._get_backend_name() if group is not None else torch.distributed.get_backend()
        if backend_name == "hccl":
            return torch.device("hpu")

        return get_pg_default_device_orig(group)

    torch.distributed.distributed_c10d._get_pg_default_device = wrap_get_pg_default_device

    # wrap torch.manual_seed

    manual_seed_orig = torch.manual_seed

    @wraps(torch.manual_seed)
    def wrap_manual_seed(seed):
        if not is_lazy():
            from habana_frameworks.torch.dynamo.compile_backend import _recipe_compiler_C
            from habana_frameworks.torch.utils import _debug_eager_C

            _debug_eager_C.join_pending_pipeline_threads()
            _recipe_compiler_C.reset_seeds()

        rand_hpu.manual_seed(seed)
        return manual_seed_orig(seed)

    torch.manual_seed = wrap_manual_seed

    # wrap torch.Tensor.record_stream

    record_orig = torch.Tensor.record_stream

    @wraps(torch.Tensor.record_stream)
    def wrap_record_stream(self, stream):
        if isinstance(self, Tensor) and self.device.type == "hpu":
            ht.record_stream(self, stream)
        else:
            record_orig(self, stream)

    torch.Tensor.record_stream = wrap_record_stream

    # wrap torch.nn.modules.Module.add_module

    add_module_orig = torch.nn.modules.Module.add_module

    @wraps(torch.nn.modules.Module.add_module)
    def wrap_add_module(self, name, module):
        if isinstance(module, torch.nn.Module) and not _names_hook_already_registered(module):
            try:
                module.custom_name = name
                module.register_forward_pre_hook(_pre_fwd_hook)
                module.register_forward_hook(_post_fwd_hook, always_call=True)
                module.names_hook = True
            except RuntimeError:
                pass
        add_module_orig(self, name, module)

    # Install NNModule Hooks that assign name to the module, what is used for
    # debugging purposes. These hooks shouldn't be used with torch.compile due
    # to potential performance regressions that may be introduced by including
    # them, reference: https://pytorch.org/docs/master/compile/nn-module.html.
    if is_lazy():
        torch.nn.modules.Module.add_module = wrap_add_module

    # wrap torch.distributed.new_group and torch.distributed.init_process_group

    ranks_cache = {}
    new_group_orig = torch.distributed.new_group
    init_process_group_orig = torch.distributed.init_process_group

    @wraps(torch.distributed.new_group)
    def wrap_new_group(ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None):
        nonlocal ranks_cache
        cache_enable = bc.get_pt_enable_comm_group_cache()
        hpu_backend_invoke = True if backend is None or "hccl" in backend else False

        if cache_enable and hpu_backend_invoke:
            nonlocal ranks_cache
            ranks_cache[backend] = {}
            if ranks == None:
                actual_world_size = torch.distributed.distributed_c10d.get_world_size()
                ranks_tuple = tuple(list(range(0, actual_world_size)))
            else:
                ranks_tuple = tuple(sorted(tuple(ranks)))
            if ranks_tuple in ranks_cache[backend]:
                return ranks_cache[backend][ranks_tuple]
            else:
                ranks_cache[backend][ranks_tuple] = new_group_orig(ranks, timeout, backend, pg_options)
                return ranks_cache[backend][ranks_tuple]
        else:
            return new_group_orig(ranks, timeout, backend, pg_options)

    @wraps(torch.distributed.init_process_group)
    def wrap_init_process_group(
        backend,
        init_method=None,
        timeout=datetime.timedelta(seconds=1800),
        world_size=-1,
        rank=-1,
        store=None,
        group_name="",
        pg_options=None,
    ):
        nonlocal ranks_cache
        cache_enable = bc.get_pt_enable_comm_group_cache()
        hpu_backend_invoke = True if backend is None or "hccl" in backend else False
        if cache_enable and hpu_backend_invoke:
            ranks_cache[backend] = {}
            if len(ranks_cache[backend]) == 0:
                init_process_group_orig(
                    backend,
                    init_method,
                    timeout,
                    world_size,
                    rank,
                    store,
                    group_name,
                    pg_options,
                )
            actual_world_size = torch.distributed.distributed_c10d.get_world_size()
            ranks_tuple = tuple(list(range(0, actual_world_size)))
            if ranks_tuple not in ranks_cache[backend]:
                ranks_cache[backend][ranks_tuple] = torch.distributed.distributed_c10d._get_default_group()
        else:
            return init_process_group_orig(
                backend,
                init_method,
                timeout,
                world_size,
                rank,
                store,
                group_name,
                pg_options,
            )

    torch.distributed.new_group = wrap_new_group
    torch.distributed.init_process_group = wrap_init_process_group

    # wrap torch.nn.Module.__setattr__

    module_set_attr_orig = torch.nn.Module.__setattr__

    # Ie9b966962fca23e5118047c3e7a47545d4c87f11 needs to be merged to resolve https://github.com/pytorch/pytorch/issues/107460
    # torch compile will have issues with log/metric attributes until then.
    @wraps(torch.nn.Module.__setattr__)
    def wrap_set_attr(self, name: str, value: Union[torch.Tensor, "torch.nn.Module"]) -> None:
        if isinstance(value, torch.nn.Module) and not _names_hook_already_registered(value):
            try:
                value.custom_name = name
                value.register_forward_pre_hook(_pre_fwd_hook)
                value.register_forward_hook(_post_fwd_hook, always_call=True)
                value.names_hook = True
            except RuntimeError:
                pass
        module_set_attr_orig(self, name, value)

    # Install NNModule Hooks that assign name to the module, what is used for
    # debugging purposes. These hooks shouldn't be used with torch.compile due
    # to potential performance regressions that may be introduced by including
    # them, reference: https://pytorch.org/docs/master/compile/nn-module.html.
    if is_lazy():
        torch.nn.Module.__setattr__ = wrap_set_attr

    # wrap torch.distributed.irecv

    dummy_mode = int(environ.get("P2P_DUMMY_MODE_PHASE", "0"))
    if dummy_mode == 1 or dummy_mode == 2:
        irecv_orig = torch.distributed.irecv
        distributed_c10d = torch.distributed.distributed_c10d

        def irecv_aux(tensor):
            dummy_mode = int(environ.get("P2P_DUMMY_MODE_PHASE", "0"))
            if not hasattr(irecv_aux, "dummy_mode_seq"):
                irecv_aux.dummy_mode_seq = 0  # it doesn't exist yet, so initialize it

            dummy_folder_path = (
                environ.get("P2P_DUMMY_MODE_PATH") if environ.get("P2P_DUMMY_MODE_PATH") != None else "./"
            )
            tensor_file = (
                dummy_folder_path + str(distributed_c10d.get_rank()) + "_" + str(irecv_aux.dummy_mode_seq) + ".pt"
            )

            if dummy_mode == 1:
                print("Dummy Mode: " + tensor_file + " saved.")
                torch.save(tensor.to("cpu"), tensor_file)

            if dummy_mode == 2:
                if path.exists(tensor_file):
                    tensor = torch.load(tensor_file).to("hpu")
                    print("Dummy Mode: " + tensor_file + " loaded.")
                else:
                    irecv_aux.dummy_mode_seq = 0
                    tensor_file = (
                        dummy_folder_path
                        + str(distributed_c10d.get_rank())
                        + "_"
                        + str(irecv_aux.dummy_mode_seq)
                        + ".pt"
                    )
                    if path.exists(tensor_file):
                        print("Dummy Mode: " + tensor_file + " loaded.")
                        tensor = torch.load(tensor_file).to("hpu")
                    else:
                        raise Exception(
                            "Attempting to run HPU Dummy Mode but needed file " + tensor_file + " does not exist!"
                        )

            irecv_aux.dummy_mode_seq += 1
            return tensor

        @wraps(torch.distributed.irecv)
        def wrap_irecv(
            tensor: torch.Tensor,
            src: Optional[int] = None,
            group: Optional[distributed_c10d.ProcessGroup] = None,
            tag: int = 0,
        ) -> distributed_c10d.Work:
            res = irecv_orig(tensor, src, group, tag)
            tensor.copy_(irecv_aux(tensor))
            return res

        torch.distributed.irecv = wrap_irecv

    # wrap torch.Tensor._reduce_ex_internal

    _original_reduce_ex_internal = torch.Tensor._reduce_ex_internal

    @wraps(torch.Tensor._reduce_ex_internal)
    def _reduce_ex_internal_habana(self, proto):
        if is_lazy():
            # Lazy tensor is storage less and will go numpy()
            # which will throw checked exception when tensor requires grad.
            with torch.no_grad():
                func, args = _original_reduce_ex_internal(self, proto)
        else:
            func, args = _original_reduce_ex_internal(self, proto)
        if self.device.type.startswith("hpu"):
            return _deserialize_habana_wrapper, (func, HpuMarker(), *args)

        return func, args

    torch.Tensor._reduce_ex_internal = _reduce_ex_internal_habana

    # wrap torch.random.fork_rng

    fork_rng_original = torch.random.fork_rng

    @wraps(torch.random.fork_rng)
    def wrap_fork_rng(
        devices=None,
        enabled=True,
        _caller="fork_rng",
        _devices_kw="devices",
        device_type=None,
    ) -> Generator:
        return fork_rng_original(
            devices=devices,
            enabled=enabled,
            _caller=_caller,
            _devices_kw=_devices_kw,
            device_type=device_type if device_type else "hpu",
        )

    torch.random.fork_rng = wrap_fork_rng

    # wrap torch.save for default to pickle protocol 4 for lazy mode
    # This should be removed if upstream pytorch moved the default to
    # pickle protocol 4.
    # Refer https://github.com/pytorch/pytorch/issues/97772

    save_orig = torch.save

    # move hpu tensor to cpu before save, to handle issues with numpy
    def convert_for_pickle(obj):
        if isinstance(obj, torch.Size):
            return obj
        elif isinstance(obj, dict):
            return {k: convert_for_pickle(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_pickle(e) for e in obj]
        elif isinstance(obj, tuple):
            return tuple([convert_for_pickle(e) for e in obj])
        else:
            if isinstance(obj, torch.Tensor):
                return obj.data.detach().clone().cpu()
            else:
                return obj

    @wraps(torch.save)
    def wrap_save(
        obj: object,
        f: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        pickle_module: Any = pickle,
        pickle_protocol: int = LAZY_DEFAULT_PROTOCOL,
        _use_new_zipfile_serialization: bool = True,
        _disable_byteorder_record: bool = False,
    ) -> None:
        data = convert_for_pickle(obj)
        save_orig(
            obj=data,
            f=f,
            pickle_module=pickle_module,
            pickle_protocol=pickle_protocol,
            _use_new_zipfile_serialization=_use_new_zipfile_serialization,
            _disable_byteorder_record=_disable_byteorder_record,
        )

    # For the storage less backend tensor such as Lazy mode tensor,
    # torch.save will use tensor.cpu().numpy() to getting a numpy
    # array and pickle the data array. This needs pickle protocol 4
    # for support tensors with size larger than 4GB. So we default
    # to pickle protocol 4 for lazy mode
    if is_lazy():
        torch.save = wrap_save

    # wrap torch.serialization._load for handling weights only
    # unpickler for lazy mode pickle protocol 4
    # This should be removed if upstream pytorch weights_only_unpickler
    # support pickle protocol 4.
    # Refer https://github.com/pytorch/pytorch/issues/118092

    serialization_load_internal_orig = torch.serialization._load

    @wraps(torch.serialization._load)
    def wrap_serialization_load_internal(
        zip_file, map_location, pickle_module, pickle_file="data.pkl", overall_storage=None, **pickle_load_args
    ):
        if pickle_module.__name__ == "torch._weights_only_unpickler":
            pickle_module = _weights_only_unpickler
        return serialization_load_internal_orig(
            zip_file=zip_file,
            map_location=map_location,
            pickle_module=pickle_module,
            pickle_file=pickle_file,
            overall_storage=overall_storage,
            **pickle_load_args
        )

    # Pickle protocol 4 is used by default for lazy mode.
    # We need the weights_only_unpickler to support pickle protocol 4.
    # So we have a lazy version of weights_only_unpickler.
    if is_lazy():
        torch.serialization._load = wrap_serialization_load_internal

    # wrap torch.serialization._legacy_load for handling weights only
    # unpickler for lazy mode pickle protocol 4
    # This should be removed if upstream pytorch weights_only_unpickler
    # support pickle protocol 4.
    # Refer https://github.com/pytorch/pytorch/issues/118092

    serialization_legacy_load_internal_orig = torch.serialization._legacy_load

    @wraps(torch.serialization._legacy_load)
    def wrap_serialization_legacy_load_internal(f, map_location, pickle_module, **pickle_load_args):
        if pickle_module.__name__ == "torch._weights_only_unpickler":
            pickle_module = _weights_only_unpickler
        return serialization_legacy_load_internal_orig(
            f=f, map_location=map_location, pickle_module=pickle_module, **pickle_load_args
        )

    # Pickle protocol 4 is used by default for lazy mode.
    # We need the weights_only_unpickler to support pickle protocol 4.
    # So we have a lazy version of weights_only_unpickler.
    if is_lazy():
        torch.serialization._legacy_load = wrap_serialization_legacy_load_internal

    # wrap torch.seed
    # In public PyTorch, torch.seed() generates a random seed and synchronizes it across available accelerators:
    # https://github.com/pytorch/pytorch/blob/b6a30bbfb6c1bcb9c785e7a853c2622c8bc17093/torch/random.py#L62

    seed_original = torch.seed

    @wraps(torch.seed)
    def wrap_seed() -> int:
        seed = seed_original()
        torch.hpu.manual_seed_all(seed)
        return seed

    torch.seed = wrap_seed


# wrap native pt2e-quant apis required to work on HPU with graph-breaks

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

org_export = torch._export.capture_pre_autograd_graph
org_prepare_pt2e = prepare_pt2e
org_convert_pt2e = convert_pt2e


def _native_pt2e_quantization_interface(name):
    if name == "export":
        global org_export
        return org_export
    elif name == "prepare_pt2e":
        global org_prepare_pt2e
        return org_prepare_pt2e
    elif name == "convert_pt2e":
        global org_convert_pt2e
        return org_convert_pt2e
    else:
        return None


def overwrite_native_pt2e_quantization_interface():

    # wrap capture_pre_autograd_graph
    @wraps(torch._export.capture_pre_autograd_graph)
    def wrap_capture_pre_autograd_graph(
        f: torch.nn.Module,
        args: Tuple[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ) -> torch.nn.Module:
        from habana_frameworks.torch.core.quantize_pt2e import export

        return export(f, args, kwargs, dynamic_shapes)

    torch._export.capture_pre_autograd_graph = wrap_capture_pre_autograd_graph

    import torch.ao.quantization.quantize_pt2e as quantize_pt2e
    from torch.ao.quantization.quantizer import Quantizer
    from torch.fx import GraphModule

    # wrap prepare_pt2e
    @wraps(quantize_pt2e.prepare_pt2e)
    def wrap_prepare_pt2e(
        model: GraphModule,
        quantizer: Quantizer,
    ) -> GraphModule:
        from habana_frameworks.torch.core.quantize_pt2e import prepare_pt2e

        return prepare_pt2e(model, quantizer)

    quantize_pt2e.prepare_pt2e = wrap_prepare_pt2e

    # wrap convert_pt2e
    @wraps(quantize_pt2e.convert_pt2e)
    def wrap_convert_pt2e(
        model: GraphModule,
        use_reference_representation: bool = False,
        fold_quantize: bool = True,
    ) -> GraphModule:
        from habana_frameworks.torch.core.quantize_pt2e import convert_pt2e

        return convert_pt2e(model, use_reference_representation, fold_quantize)

    quantize_pt2e.convert_pt2e = wrap_convert_pt2e
