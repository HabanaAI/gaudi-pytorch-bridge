import torch
from functools import wraps
from . import config


def vprint(*args, **kwds):
    if (config.verbose_mode):
        print(*args, **kwds)
    else:
        pass


def to_bf16(x):
    if (x.dtype == torch.float32):
        return x.type(torch.bfloat16)
    else:
        return x


def to_fp32(x):
    if (x.dtype == torch.bfloat16):
        return x.type(torch.float)
    else:
        return x


def get_list_from_file(file_path):
    with open(file_path) as file:
        ops_list = file.read().splitlines()

    return ops_list


def check_input(opt_level, bf16_file_path, fp32_file_path):
    assert ((opt_level == 'O1') or (opt_level == 'O2')), "Optlevel should be either O1 or O2"

    if ((opt_level == 'O2') and ((bf16_file_path != '') or (fp32_file_path != ''))):
        print("Input op list would be overridden in opt_level O2")


def decide_cast_fn(* args, **kwds):
    # Float if any of the input tensor float else bf16
    dtype_list = []
    for arg in args:
        if (torch.is_tensor(arg) or isinstance(arg, torch.autograd.Variable)):
            dtype_list.append(str(arg.dtype))

    for key, val in kwds.items():
        if (torch.is_tensor(val) or isinstance(val, torch.autograd.Variable)):
            dtype_list.append(str(val.dtype))

    if 'torch.float32' in dtype_list:
        cast_fn = to_fp32
    else:
        cast_fn = to_bf16

    vprint("Cast function decided", cast_fn.__name__)
    return cast_fn


def decide_cast_fn_inplace(* args, **kwds):
    # decide cast dtype based on first argument
    arg0 = args[0]
    assert (torch.is_tensor(arg0) or isinstance(arg0, torch.autograd.Variable)
            ), "Self should be a tensor in inplace op"
    if (str(arg0.dtype) == 'torch.float32'):
        cast_fn = to_fp32
    else:
        cast_fn = to_bf16

    vprint("Inplace Decided", cast_fn.__name__)

    return cast_fn


def get_new_args(cast_fn, args, kwds):
    # args is a tuple and hence immutable - create new tuple
    args_cast = []
    for arg in args:
        if (torch.is_tensor(arg) or isinstance(arg, torch.autograd.Variable)):
            args_cast.append(cast_fn(arg))
        else:
            args_cast.append(arg)

    # kwds modified inplace
    for key, val in kwds.items():
        if (torch.is_tensor(val) or isinstance(val, torch.autograd.Variable)):
            kwds[key] = cast_fn(val)
    return tuple(args_cast)


def op_wrap(op, cast_fn):
    '''cast all the tensors like objects within a op'''
    @wraps(op)
    def wrapper(*args, **kwds):
        vprint('casting ', op, cast_fn.__name__)

        args_cast = get_new_args(cast_fn, args, kwds)

        return op(*args_cast, **kwds)
    return wrapper


def op_wrap_dynamic(op):
    @wraps(op)
    def wrapper_dynamic(*args, **kwds):
        cast_fn = decide_cast_fn(*args, **kwds)
        vprint('casting ', op, " to ", cast_fn.__name__)

        args_cast = get_new_args(cast_fn, args, kwds)

        return op(*args_cast, **kwds)
    return wrapper_dynamic


def op_wrap_dynamic_inplace(op):
    @wraps(op)
    def wrapper_dynamic_inplace(*args, **kwds):
        cast_fn = decide_cast_fn_inplace(*args, **kwds)
        vprint('casting ', op, " to ", cast_fn.__name__)

        args_cast = get_new_args(cast_fn, args, kwds)

        return op(*args_cast, **kwds)
    return wrapper_dynamic_inplace


def cast_ops_list(ops_list, ops_dict, cast_fn=None):
    for op in ops_list:
        key = str(op)
        if key in ops_dict:
            for mod in ops_dict[key]:
                if (mod == torch.Tensor):
                    if (hasattr(mod, op)):
                        pt_op = getattr(mod, op)
                    else:
                        pt_op = getattr(mod, '__' + str(op) + '__')
                else:
                    pt_op = getattr(mod, op)
                if (cast_fn is not None):
                    wrapper = op_wrap(pt_op, cast_fn)
                    vprint("Wrapping ", pt_op, " ", cast_fn.__name__)
                else:
                    vprint("Deciding cast for", pt_op)
                    if (mod == torch.Tensor):
                        wrapper = op_wrap_dynamic_inplace(pt_op)
                    else:
                        wrapper = op_wrap_dynamic(pt_op)
                setattr(mod, op, wrapper)
        else:
            print(op, "is an invalid/unsupported op")
