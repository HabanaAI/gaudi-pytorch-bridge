import torch
import os
import builtins
from .ops_dict import ops_dict
from .utils import *
from . import config


def convert(opt_level="O1", bf16_file_path="", fp32_file_path="", isVerbose=False):
    """Entry function to Habana Mixed Precision (HMP) tool.
    This tool inserts cast nodes to inputs of torch OPs based on provided
    optimization_level, list of always bf16 OPs and list of always fp32 OPs.
    Any torch OP not in bf16 or fp32 list will follow the type of preceding
    OP in the graph. Exceptions to this rule would be,
    - OPs with multiple tensor inputs (other than weight, bias) shall cast to bf16
    if all tensors are bf16 but will cast to fp32 if any tensor is fp32
    - Inplace OPs shall cast to type of inplace tensor argument

    Args:
    bf16_file_path : User provided file containing list of torch ops that needs to
                     operate on bf16 inputs. If null string, then default list
                     ops_bf16.txt would be used
    fp32_file_path : User provided file containing list of torch ops that needs to
                     operate on fp32 inputs. If null string, then default list
                     ops_fp32.txt would be used
    isVerbose : Enable/disable verbose mode
    opt_level : O1 - Two files for torch operators are taken as input namely for
                bf16 and fp32 dtypes. Dtypes for rest of the operators would
                be decided based on their input tensors
                O2 - GEMM and Conv kernels would operate on bf16. Rest of the
                operators in HMP ops_dict operate on fp32.
    """

    check_input(opt_level, bf16_file_path, fp32_file_path)

    config.verbose_mode = isVerbose
    config.opt_level = opt_level

    print("hmp:verbose_mode ", config.verbose_mode)
    print("hmp:opt_level", config.opt_level)

    # Use default files if user didn't provide
    if opt_level == "O1":
        dir_path = os.path.dirname(os.path.realpath(__file__))

        if bf16_file_path == "":
            bf16_file_path = os.path.join(dir_path, "ops_bf16.txt")

        if fp32_file_path == "":
            fp32_file_path = os.path.join(dir_path, "ops_fp32.txt")

        any_file_path = os.path.join(dir_path, "ops_multi_inputs.txt")

        ops_bf16_list = get_list_from_file(bf16_file_path)

        ops_fp32_list = get_list_from_file(fp32_file_path)

        ops_any_list = get_list_from_file(any_file_path)

        # Handle rest of the multi input ops
        # single input op will follow the previous node
        cast_ops_list(ops_any_list, ops_dict)

        # cast ops in the bf16 list
        cast_ops_list(ops_bf16_list, ops_dict, to_bf16)

        # cast ops in the fp32 list
        cast_ops_list(ops_fp32_list, ops_dict, to_fp32)

    elif opt_level == "O2":
        ops_bf16_list = [
            "conv1d",
            "conv2d",
            "conv3d",
            "bmm",
            "addmm",
            "mm",
            "mv",
            "dot",
        ]
        ops_fp32_list = []

        for key in ops_dict:
            if key not in ops_bf16_list:
                ops_fp32_list.append(key)

        # cast ops in the bf16 list
        cast_ops_list(ops_bf16_list, ops_dict, to_bf16)

        # cast ops in the fp32 list
        cast_ops_list(ops_fp32_list, ops_dict, to_fp32)
