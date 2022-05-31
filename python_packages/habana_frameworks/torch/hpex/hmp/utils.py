import contextlib
import torch
from functools import wraps
from . import config


class HmpState(object):
    def __init__(self):
        self.disable_cast = False


# Attribute store.  Can also just store things
# as global module attributes.
_hmp_state = HmpState()


@contextlib.contextmanager
def disable_casts():
    """This function is used to create a
       caller context where all casts are
       disabled.
       E.g. around optimizer.step()"""
    if config.verbose_mode:
        print("Disabled HMP casting")
    _hmp_state.disable_cast = True
    yield
    if config.verbose_mode:
        print("Enabled HMP casting")
    _hmp_state.disable_cast = False


def vprint(*args, **kwds):
    """Enable prints in verbose mode"""
    if config.verbose_mode:
        print(*args, **kwds)
    else:
        pass


def to_bf16(x):
    """Cast tensor to bf16"""
    if x.dtype == torch.float32:
        return x.type(torch.bfloat16)
    else:
        return x


def to_fp32(x):
    """Cast tensor to fp32"""
    if x.dtype == torch.bfloat16:
        return x.type(torch.float)
    else:
        return x


def inplace(x):
    """Return inplace version of input OP"""
    return x + "_"


def overrides(x):
    """Return override version of input OP"""
    return "__" + x + "__"


def get_list_from_file(file_path):
    """Get OP list from a txt file"""

    with open(file_path) as file:
        ops_list = file.read().splitlines()

    return ops_list


def check_input(opt_level, bf16_file_path, fp32_file_path):
    """Run some sanity checks on user provided inputs"""

    assert (opt_level == "O1") or (
        opt_level == "O2"
    ), "Optlevel should be either O1 or O2"

    if (opt_level == "O2") and ((bf16_file_path != "") or (fp32_file_path != "")):
        print("Input op list would be overridden in opt_level O2")


def decide_cast_fn(*args, **kwds):
    """Decides cast_fn as fp32 if any tensor is float, else cast_fn is bf16"""

    dtype_list = []
    for arg in args:
        if torch.is_tensor(arg) or isinstance(arg, torch.autograd.Variable):
            dtype_list.append(str(arg.dtype))

    for key, val in kwds.items():
        if torch.is_tensor(val) or isinstance(val, torch.autograd.Variable):
            dtype_list.append(str(val.dtype))

    if "torch.float32" in dtype_list:
        cast_fn = to_fp32
    else:
        cast_fn = to_bf16

    vprint("Cast function decided", cast_fn.__name__)
    return cast_fn


def decide_cast_fn_inplace(*args, **kwds):
    """Decides cast_fn based on first/inplace argument dtype"""

    arg0 = args[0]
    assert torch.is_tensor(arg0) or isinstance(
        arg0, torch.autograd.Variable
    ), "Self should be a tensor in inplace op"
    if str(arg0.dtype) == "torch.float32":
        cast_fn = to_fp32
    else:
        cast_fn = to_bf16

    vprint("Cast function Decided", cast_fn.__name__)

    return cast_fn


def get_new_args(cast_fn, args, kwds):
    """ Iterate and cast any tensors in args or kwds using cast_fn"""

    # args is a tuple and hence immutable - create new tuple
    args_cast = []
    for arg in args:
        if torch.is_tensor(arg) or isinstance(arg, torch.autograd.Variable):
            args_cast.append(cast_fn(arg))
        else:
            args_cast.append(arg)

    # kwds modified inplace
    for key, val in kwds.items():
        if torch.is_tensor(val) or isinstance(val, torch.autograd.Variable):
            kwds[key] = cast_fn(val)
    return tuple(args_cast)


def op_wrap_var_input_len(op, cast_fn, wrap_len):
    """Adds wrapper function to OPs. only 1st wrap_len
    tensor inputs for the OP are casted to type determined
    by cast_fn provided.

    Args:
    op (torch.nn.functional/torch/torch.Tensor): Input OP
    cast_fn (to_bf16/to_fp32): Fn to cast input tensors

    Returns:
    Wrapper function that shall be inserted back to
    corresponding module for this OP.
    """
    vprint("Wrapping ", op, " ", cast_fn.__name__)

    @wraps(op)
    def wrapper(*args, **kwds):
        if _hmp_state.disable_cast:
            return op(*args, **kwds)

        vprint("casting ", op, cast_fn.__name__)
        #Because batch_norm is different from layer_norm, kwds include tensors and scale parameters.
        #kwds should keep the orignal data type.
        args_out = get_new_args(cast_fn, args[0:wrap_len], dict())
        args_cast = args_out + args[wrap_len:]
        return op(*args_cast, **kwds)

    return wrapper


def op_wrap(op, cast_fn):
    """Adds wrapper function to OPs. All tensor inputs
    for the OP are casted to type determined by cast_fn
    provided.

    Args:
    op (torch.nn.functional/torch/torch.Tensor): Input OP
    cast_fn (to_bf16/to_fp32): Fn to cast input tensors

    Returns:
    Wrapper function that shall be inserted back to
    corresponding module for this OP.
    """
    vprint("Wrapping ", op, " ", cast_fn.__name__)

    @wraps(op)
    def wrapper(*args, **kwds):
        if _hmp_state.disable_cast:
            return op(*args, **kwds)

        vprint("casting ", op, cast_fn.__name__)
        args_cast = get_new_args(cast_fn, args, kwds)
        return op(*args_cast, **kwds)

    return wrapper


def op_wrap_dynamic(op):
    """Adds wrapper function for OPs with multiple
    tensor inputs (other than weight, bias etc.), This
    wrapper function looks for largest data type
    among all tensor inputs and promotes (casts) all
    tensor inputs to this type.

    Args:
    op (torch.nn.functional/torch/torch.Tensor): Input OP

    Returns:
    Wrapper function that shall be inserted back to
    corresponding module for this OP.
    """
    vprint("Deciding cast for", op)

    @wraps(op)
    def wrapper_dynamic(*args, **kwds):
        if _hmp_state.disable_cast:
            return op(*args, **kwds)

        if isinstance(args[0], list) or isinstance(args[0], tuple):
            # ops with tensorlist as input
            cast_fn = decide_cast_fn(*args[0], **kwds)
            vprint("casting ", op, " to ", cast_fn.__name__)
            args_cast = get_new_args(cast_fn, args[0], kwds)
            return op(args_cast, *args[1:], **kwds)
        else:
            # ops with tensors as input
            cast_fn = decide_cast_fn(*args, **kwds)
            vprint("casting ", op, " to ", cast_fn.__name__)
            args_cast = get_new_args(cast_fn, args, kwds)
            return op(*args_cast, **kwds)

    return wrapper_dynamic


def op_wrap_dynamic_inplace(op):
    """Adds wrapper function for inplace OPs. This wrapper
    function casts all the tensor inputs for the inplace OP
    to same type as inplace tensor (assumed to be 1st tensor
    in argument list).

    Args:
    op (torch.nn.functional/torch/torch.Tensor): Input OP

    Returns:
    Wrapper function that shall be inserted back to
    corresponding module for this OP.
    """

    vprint("Deciding cast for", op)

    @wraps(op)
    def wrapper_dynamic_inplace(*args, **kwds):
        if _hmp_state.disable_cast:
            return op(*args, **kwds)

        cast_fn = decide_cast_fn_inplace(*args, **kwds)
        vprint("casting ", op, " to ", cast_fn.__name__)

        args_cast = get_new_args(cast_fn, args, kwds)

        return op(*args_cast, **kwds)

    return wrapper_dynamic_inplace


def cast_ops_list(ops_list, ops_dict, cast_fn=None):
    """Takes a list of OPs as input and adds a wrapper function
    around each OP in the list based on the module type for an OP
    and the cast_fn provided.

    Args:
    ops_list (list): Input list of OPs
    ops_dict (dict): Dictionary with all OPs supported by HMP
                     package. Key is OP name, value is torch
                     module(s) to which OP belongs.
    cast_fn (to_bf16, to_fp32, None): cast function to be used
                     on OPs in input list
    """
    special_list = ["layer_norm", "batch_norm", "instance_norm", "group_norm"]

    for op in ops_list:
        key = str(op)
        if key in ops_dict:
            for mod in ops_dict[key]:
                # Handle ops from bf16, fp32, multi-input lists
                for op_in in [op, overrides(op)]:
                    if hasattr(mod, op_in):
                        pt_op = getattr(mod, op_in)
                        if cast_fn is not None:
                            if op_in not in special_list:
                                wrapper = op_wrap(pt_op, cast_fn)
                            else:
                                wrapper = op_wrap_var_input_len(pt_op, cast_fn, 1)
                        else:
                            wrapper = op_wrap_dynamic(pt_op)
                        setattr(mod, op_in, wrapper)

                # Handle inplace ops. For now inplace handling limited to
                # ops coming from multi-input list because it maybe tricky
                # to force strict casting of inplace ops
                if cast_fn is None:
                    if hasattr(mod, inplace(op)):
                        pt_op_inplace = getattr(mod, inplace(op))
                        wrapper = op_wrap_dynamic_inplace(pt_op_inplace)
                        setattr(mod, inplace(op), wrapper)

        else:
            print(op, "OP not supported. Please add it to OPs dict and try again")
