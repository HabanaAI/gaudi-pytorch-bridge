import os
from .ops_dict import ops_dict
from .utils import *
from . import config


def convert(opt_level="O1", bf16_file_path="", fp32_file_path="", fp16_file_path="", isVerbose=False, low_precision_type = torch.bfloat16):
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
    fp16_file_path : User provided file containing list of torch ops that needs to
                     operate on fp16 inputs. If null string, then default list
                     ops_fp16.txt would be used
    fp32_file_path : User provided file containing list of torch ops that needs to
                     operate on fp32 inputs. If null string, then default list
                     ops_fp32.txt would be used
    low_precision_type: Type which is used for conversion to low precision. Could be
                     bfloat16 or float16.
    isVerbose : Enable/disable verbose mode
    opt_level : O1 - Two files for torch operators are taken as input namely for
                bf16 and fp32 dtypes. Dtypes for rest of the operators would
                be decided based on their input tensors
                O2 - GEMM and Conv kernels would operate on bf16. Rest of the
                operators in HMP ops_dict operate on fp32.
    """

    check_input(opt_level, bf16_file_path, fp32_file_path, fp16_file_path, low_precision_type)

    config.verbose_mode = isVerbose
    config.opt_level = opt_level

    print("hmp:verbose_mode ", config.verbose_mode)
    print("hmp:opt_level", config.opt_level)

    clp = ConvertLowPrecision(low_precision_type)

    # Use default files if user didn't provide
    if opt_level == "O1":
        dir_path = os.path.dirname(os.path.realpath(__file__))

        is_bf = low_precision_type == torch.bfloat16
        low_precision_file_path = bf16_file_path if is_bf else fp16_file_path
        low_precision_default_file = "ops_bf16.txt" if is_bf else "ops_fp16.txt"
        fp32_default_file = "ops_fp32_for_bf16.txt" if is_bf else "ops_fp32_for_fp16.txt"

        if low_precision_file_path == "":
            low_precision_file_path = os.path.join(dir_path, low_precision_default_file)

        if fp32_file_path == "":
            fp32_file_path = os.path.join(dir_path, fp32_default_file)

        any_file_path = os.path.join(dir_path, "ops_multi_inputs.txt")

        # Find a list of OPs common to bf16 and any list and remove
        # these from any list (=> higher prio to bf16 list). Do same
        # thing for fp32 and any list (=> higher prio to fp32 list).
        # E.g. if you have "add" in both multi-inputs OPs and bf16
        # lists, then "add" inputs will be casted to bf16.

        ops_low_precision_list = get_list_from_file(low_precision_file_path)
        if config.verbose_mode:
            print("ops_low_precision_list = ", ops_low_precision_list)

        ops_fp32_list = get_list_from_file(fp32_file_path)
        if config.verbose_mode:
            print("ops_fp32_list = ", ops_fp32_list)

        ops_any_list = get_list_from_file(any_file_path)
        if config.verbose_mode:
            print("ops_any_list = ", ops_any_list)

        low_precision_common = [i for i in ops_low_precision_list if i in ops_any_list]
        if config.verbose_mode:
            print("low_precision_common = ", low_precision_common)

        fp32_common = [i for i in ops_fp32_list if i in ops_any_list]
        if config.verbose_mode:
            print("fp32_common = ", fp32_common)

        x = [i for i in ops_any_list if i not in low_precision_common]
        x = [i for i in x if i not in fp32_common]
        ops_any_list = x
        if config.verbose_mode:
            print("ops_any_list = ", ops_any_list)

        # Make sure there is no OP which is there in more than
        # 1 list
        x = [i for i in ops_any_list if i in ops_low_precision_list]
        assert len(x) == 0, str(x) + " in both any & low precision list"
        x = [i for i in ops_any_list if i in ops_fp32_list]
        assert len(x) == 0, str(x) + " in both any & fp32 list"
        x = [i for i in ops_low_precision_list if i in ops_fp32_list]
        assert len(x) == 0, str(x) + " in both fp32 & low precision list"

        # Handle the multi input ops
        cast_ops_list(ops_any_list, ops_dict, cast_set = clp)

        # cast ops in the low precision list
        cast_ops_list(ops_low_precision_list, ops_dict, clp.to_lp())

        # cast ops in the fp32 list
        cast_ops_list(ops_fp32_list, ops_dict, clp.to_fp32())

    elif opt_level == "O2":
        ops_low_precision_list = [
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
            "bmm",
            "addmm",
            "mm",
            "mv",
            "dot",
        ]
        ops_fp32_list = []

        for key in ops_dict:
            if key not in ops_low_precision_list:
                ops_fp32_list.append(key)

        # cast ops in the low precision list
        cast_ops_list(ops_low_precision_list, ops_dict, clp.to_lp())

        # cast ops in the fp32 list
        cast_ops_list(ops_fp32_list, ops_dict, clp.to_fp32())
