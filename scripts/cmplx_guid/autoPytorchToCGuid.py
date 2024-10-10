import itertools
import math
import os
import random
import sys

import numpy as np

is_release = True

MAX_TENSOR_DIM = 8

# E.g. python main.py params.txt

path_to_params_file = sys.argv[1]

params_buf = open(path_to_params_file, "r")
lines = params_buf.readlines()

is_fwd = False
is_bwd = False
is_opbackend_present = True
make_changes_in_folders = ["pytorch-integration", "tpc_fuser", "tpc_kernels"]
make_changes_in_file = ""

for line in lines:
    line = line.replace(" ", "").replace("\n", "")
    if line == "":
        continue
    word1 = line.split("=")[0]
    word2 = line.split("=")[1]

    if word1[0] == "#":
        continue  # comment
    elif word1 == "npu_stack_path":
        path_to_npu_stack = word2  # Path to npu_stack - /home/tkanvar/trees/npu-stack
    elif word1 == "registration_declaration_path":
        path_to_reg_decl = word2  # Path to registration declarations file - /home/tkanvar/anaconda3/envs/py_venv_3.8/lib/python3.8/site-packages/torch/include/ATen/RegistrationDeclarations.h
    elif word1 == "hpu_op_yaml_entry":
        hpu_op = word2  # ops in hpu_op.yaml, comma separated if multiple ops have same op_backend
    elif word1 == "reg_decl_op":
        reg_decl_op = word2  # op name in registraion declaration file
    elif word1 == "op_name":
        op = word2  # frac
    elif word1 == "params":
        params = word2  # ns_QuadTree::ParamsSegmentsPerAxis, if no param then NoParam
    elif word1 == "trait":
        trait = word2  # BroadcastableElementwiseTrait
    elif word1 == "kernel_op":
        kernel_op = word2  # FracKernelOp
    elif word1 == "op_folder":
        op_folder = word2  # Elementwise
    elif word1 == "op_file_name":
        op_file_name = word2  # Frac, which will be made FracTest.cpp and FracTest.hpp
    elif word1 == "inputs_dim_max":
        inputs_dim_max = word2  # max_dim_of_inputs_in_order_of_op_defintion_including_optional_comma_separated -- If no limit then write MAX_TENSOR_DIM
    elif word1 == "inputs_dim_min":
        inputs_dim_min = word2  # min_dim_of_inputs_in_order_of_op_defintion_including_optional_comma_separated[p0]
    elif word1 == "outputs_dim_max":
        outputs_dim_max = word2
    elif word1 == "outputs_dim_min":
        outputs_dim_min = word2
    elif word1 == "torch_op":
        torch_op = word2  # torch::frac
    elif word1 == "pbtxt_file_path":
        pbtxt_file_path = word2  # /home/tkanvar/trees/npu-stack/tpc_kernels/doc/training/op_def/07_misc/frac.pbtxt
    elif word1 == "greco_pbtxt_file_path":
        greco_pbtxt_file_path = (
            word2  # /home/tkanvar/trees/npu-stack/tpc_kernels/doc/inference/op_def/elementwise/unary/frac.pbtxt
        )
    elif word1 == "is_fwd":
        if word2.lower() == "true":
            is_fwd = True  # if the op provided have fwd entry apart from normal entry, then append in preexisting cpp, hpp files.
    elif word1 == "is_bwd":
        if word2.lower() == "true":
            is_bwd = True  # if the op provided has bwd entry, then append in preexisting cpp, hpp files
    elif word1 == "is_opbackend_present":
        if word2.lower() == "false":
            is_opbackend_present = False
    elif word1 == "make_changes_in_folders":
        word2 = word2.replace(" ", "")
        make_changes_in_folders = word2.split(",")
    elif word1 == "make_changes_in_file":
        make_changes_in_file = word2

inputs_dim_max = inputs_dim_max.split(",")
inputs_dim_max = [eval(i) for i in inputs_dim_max]
inputs_dim_min = inputs_dim_min.split(",")
inputs_dim_min = [eval(i) for i in inputs_dim_min]
outputs_dim_max = outputs_dim_max.split(",")
outputs_dim_max = [eval(i) for i in outputs_dim_max]
outputs_dim_min = outputs_dim_min.split(",")
outputs_dim_min = [eval(i) for i in outputs_dim_min]

# Files read
reg_decl_file = path_to_reg_decl

# File write
td_file = path_to_npu_stack + "/tpc_fuser/mlir/include/tpc/fuser/Optimizer/Dialect/TPCKernelOps.td"
cmplx_guid_supp_list_file = path_to_npu_stack + "/tpc_fuser/mlir/lib/Driver/ComplexGuidSupportedList.inc"
tpckernel_ops_file = path_to_npu_stack + "/tpc_fuser/mlir/test/Optimizer/Dialect/tpckernel-ops.mlir"
complex_guid_file = path_to_npu_stack + "/tpc_fuser/mlir/test/Optimizer/Transforms/complex-guid.mlir"
op_file = (
    path_to_npu_stack
    + "/tpc_fuser/mlir/lib/Optimizer/Transforms/ComplexGuid/"
    + op_folder
    + "/"
    + op_file_name
    + ".cpp"
)
op_test_cpp_file = (
    path_to_npu_stack + "/tpc_fuser/mlir/pytenettests/ComplexGuidTests/" + op_folder + "/" + op_file_name + "Test.cpp"
)
op_test_hpp_file = (
    path_to_npu_stack + "/tpc_fuser/mlir/pytenettests/ComplexGuidTests/" + op_folder + "/" + op_file_name + "Test.hpp"
)
shape_inference_file = path_to_npu_stack + "/tpc_fuser/mlir/lib/Driver/ShapeInference.cpp"
cmakelist_file = path_to_npu_stack + "/tpc_fuser/mlir/lib/Optimizer/Transforms/ComplexGuid/CMakeLists.txt"

hpu_yaml_file = path_to_npu_stack + "/pytorch-integration/scripts/hpu_op.yaml"
gen_file = path_to_npu_stack + "/pytorch-integration/hpu_ops/" + reg_decl_op + "_gen.cpp"
greco_pbtxt_file = greco_pbtxt_file_path
pbtxt_file = pbtxt_file_path

######## Global Variables used and description
# op_info - Information of op
#           "outputs_var" - Output variables in complex guid mlir
#           "op" - op name in complex guid mlir
#           "inputs_var" - Input variables in complex guid mlir
#           "params" - Parameter in complex guid mlir for intermediate op, e.g. constantKernelOp has param
#           "is_optional" - Does the op have optional parameters also
#           "optional_inp" - Optional inputs - Number of elements are all permanent and optional inputs. Index of permanent inputs are marked as 0 and optional is marked as 0
#           "optional_inp_def_val" - Default values of optional parameters. Permanent parameters index values are empty
# optional_params_permutation - Is array of array. Its length is 0 if no optional parameter
#                               Truth table with value 'w' means we need to consider this index optional parameter and
#                               'w/o' means this index optional parameter is considered null
#                               Number of elements in 1st row is number of optional parameters allowed in an op
# features_map - map of features of op from Hpu_op.yaml file.
#                ["op_backend"] - name of gen file containing AddNode function
#                ["dtypes"] - string of supported dtypes and on which devices the dtypes are supported
#                             in the form - SupportedPrecisionTypes<[DeviceGaudi, DeviceGaudi2, DeviceGreco],\n[F32, BF16]
#                             in order to access them we need to split the string

op_info = {
    "outputs_var": [],
    "op": [],
    "inputs_var": [],
    "params": [],
    "is_optional": False,
    "optional_inp": [],
    "optional_inp_def_val": [],
}

features_map = {}

######## Populate .td file
if make_changes_in_folders.count("tpc_fuser") > 0 and (make_changes_in_file == "" or make_changes_in_file == td_file):
    print("Writing in file = ", td_file)

    # Read file to locate appropriate location to plcae it
    td_file_buf = open(td_file, "a")
    with open(td_file, "r") as f:
        contents = f.readlines()
        lineno = -1
        for line in contents:
            lineno += 1
            if line == "#endif // TPC_FUSER_TPCKERNELOPS_H":
                break

    # Prepare op_td
    def getInputs(op):
        num_inputs = 0
        isInputMatch = False
        inputs_any_match = []

        f = open(reg_decl_file, "r")
        lines = f.readlines()
        inputs = ""
        for line in lines:
            fn_params = line.split("(")
            fn = fn_params[0].split(" ")

            if len(fn) < 2:
                continue

            if fn[1] == op or (len(fn) > 2 and fn[2] == op):
                params_schema = fn_params[2].split(",")
                for p in params_schema:
                    num_inputs += 1

                    shape = "Shape"
                    if num_inputs > len(inputs_dim_max):
                        break

                    if inputs_dim_max[num_inputs - 1] == inputs_dim_min[num_inputs - 1]:
                        shape = "D" + str(inputs_dim_max[num_inputs - 1])

                    any_match = "Any"
                    if (
                        inputs_dim_max[num_inputs - 1] == outputs_dim_max[0]
                        and inputs_dim_min[num_inputs - 1] == outputs_dim_min[0]
                    ):
                        any_match = "Match"
                        isInputMatch = True
                    inputs_any_match.append(any_match)

                    if len(p.split("=")) > 1:
                        if inputs == "":
                            inputs += any_match + shape + "<Optional<PrecisionType>>"
                        else:
                            inputs += ", " + any_match + shape + "<Optional<PrecisionType>>"
                        op_info["is_optional"] = True
                        op_info["optional_inp"].append(1)
                        prm = p.split("=")[1]
                        if prm.find(")") != -1:
                            prm = prm.split(")")[0]
                        op_info["optional_inp_def_val"].append(prm)
                    else:
                        if inputs == "":
                            inputs += any_match + shape + "<PrecisionType>"
                        else:
                            inputs += ", " + any_match + shape + "<PrecisionType>"
                        op_info["optional_inp"].append(0)
                        op_info["optional_inp_def_val"].append("")

                    if p.find("->") != -1:
                        break

                break

        return inputs, num_inputs, isInputMatch, inputs_any_match

    def getHpuYamlOpInfo(op, features_map):
        hpu_ops = op.split(",")

        hpu_buf = open(hpu_yaml_file, "r")
        lines = hpu_buf.readlines()

        is_new_block_start = True
        is_first_line = True
        is_op_found = False
        line_indx = -1

        for line in lines:
            line_indx += 1

            if not is_first_line and line == "\n":
                if is_op_found:
                    break

                is_new_block_start = True
                continue

            if is_first_line:
                is_first_line = False

            if is_new_block_start and line.split(":")[0] == hpu_ops[0]:
                is_new_block_start = False
                is_op_found = True
                features_map["line_indx"] = line_indx
                continue

            if is_op_found:
                line = line.replace(" ", "")
                feature_line = line.split(":")

                if feature_line[0] == "dtypes":
                    features_map["dtypes"] = ""

                    # ToDo: look if there are multiple fields in dtype, else put all under one SupportedPrecisionTypes
                    features_map["dtypes"] += "SupportedPrecisionTypes<[DeviceGaudi, DeviceGaudi2, DeviceGreco],\n"
                    features_map["dtypes"] += "\t\t\t\t\t\t\t["

                    dtypes_hpu = feature_line[1].split("[")
                    dtypes_hpu = dtypes_hpu[1].split("]")
                    dtypes_hpu = dtypes_hpu[0].split(",")

                    is_first = True
                    for d in dtypes_hpu:
                        if not is_first:
                            features_map["dtypes"] += ", "
                        is_first = False
                        if d == "BFloat16":
                            features_map["dtypes"] += "BF16"
                        elif d == "Float":
                            features_map["dtypes"] += "F32"
                        elif d == "Int":
                            features_map["dtypes"] += "I32"

                    features_map["dtypes"] += "]>]\n"

                if feature_line[0] == "op_backend":
                    features_map["op_backend"] = feature_line[1].strip()

            if line == "\n":
                is_new_block = True
            else:
                is_new_block = False

        return features_map

    inputs_str, num_inputs, isInputMatch, inputs_any_match = getInputs(reg_decl_op)

    outputs_str = ""
    outputs_any_match = []
    if isInputMatch:
        outputs_str += "Match"
        outputs_any_match.append("Match")
    else:
        outputs_str += "Any"
        outputs_any_match.append("Any")
    if outputs_dim_max[0] == outputs_dim_min[0]:
        outputs_str += str(outputs_dim_max[0])
    else:
        outputs_str += "Shape"
    outputs_str += "<PrecisionType>"

    num_outputs = 1

    getHpuYamlOpInfo(hpu_op, features_map)

    op_td = 'defm "" : TPCKernel_Ops<["' + op
    if is_fwd:
        op_td += ", " + op + "_fwd"
    op_td += '"],\n'
    op_td += "    /* inputs=*/[" + inputs_str + "],\n"
    op_td += "    /*outputs=*/[" + outputs_str + "],\n"
    if params == "NoParams":
        op_td += "    " + params + ",\n"
    else:
        op_td += '    Params<"' + params + '">,\n'
    op_td += "    [" + trait + ",\n"
    op_td += "    " + features_map["dtypes"]
    op_td += ">;\n\n"

    # Write in td file
    contents.insert(lineno, op_td)

    if is_release:
        with open(td_file, "w") as f:
            contents = "".join(contents)
            f.write(contents)

######## Populate ComplexGuidSupportedList.inc
if make_changes_in_folders.count("tpc_fuser") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == cmplx_guid_supp_list_file
):
    print("Writing in file = ", cmplx_guid_supp_list_file)

    file_buf = open(cmplx_guid_supp_list_file, "r")
    lines = file_buf.readlines()
    indx = -1
    for line in lines:
        indx += 1
        if line.find("KernelOp") and line.find("ComplexGuid<") != -1:
            op_name = line.split("ComplexGuid<")[1]
            op_name = op_name.split(",")[0]

            if op_name > kernel_op:
                break

    line_to_insert = "ComplexGuid<" + kernel_op + ">,\n"
    if is_fwd:
        fwd_kernel_op = kernel_op.split("KernelOp")[0]
        fwd_kernel_op += "FwdKernelOp"
        line_to_insert += "ComplexGuid<" + fwd_kernel_op + ">,\n"
    lines.insert(indx, line_to_insert)

    if is_release:
        with open(cmplx_guid_supp_list_file, "w") as f:
            lines = "".join(lines)
            f.write(lines)

######## Populate op.cpp
if make_changes_in_folders.count("tpc_fuser") > 0 and (make_changes_in_file == "" or make_changes_in_file == op_file):
    print("Writing in file = ", op_file)

    op_file_buf = open(op_file, "w")

    def writeHeader(file_buf):
        file_buf.write(
            "//===- "
            + op_file_name
            + ".cpp -- "
            + op_file_name
            + " Op conversion ------------------------------===//\n"
        )
        file_buf.write("//\n")
        file_buf.write("// Copyright (C) 2022 HabanaLabs, Ltd.\n")
        file_buf.write("// All Rights Reserved.\n")
        file_buf.write("// Unauthorized copying of this file, via any medium is strictly prohibited.\n")
        file_buf.write("// Proprietary and confidential.\n")
        file_buf.write("//\n")
        file_buf.write("//===----------------------------------------------------------------------===//\n")
        file_buf.write("//\n")
        file_buf.write("// This file implements conversion patterns for " + op_file_name + " operation.\n")
        file_buf.write("//\n")
        file_buf.write("//===----------------------------------------------------------------------===//\n")
        file_buf.write("\n")
        file_buf.write('#include "HandleNDOperation.hpp"\n')
        file_buf.write('#include "../ExtractComplexGuidUtils.h"\n')
        file_buf.write("\n")
        file_buf.write("using namespace tpc::fuser::cguid;\n")
        file_buf.write("using namespace mlir;\n")
        file_buf.write("using namespace mlir::syn_rt;\n")
        file_buf.write("using namespace gcapi;\n")
        file_buf.write("\n")
        file_buf.write("namespace tpc {\n")
        file_buf.write("namespace fuser {\n")
        file_buf.write("\n")

    def writeFn(file_buf, params, num_inputs, num_outputs, features_map, op_info):
        file_buf.write("template <typename KernelOp>\n")
        if params == "NoParams":
            file_buf.write(
                "LogicalResult handleComplexGuid(KernelOp op, CommonNodeLoc loc, RewriterBase &rewriter) {\n"
            )
        else:
            file_buf.write(
                "LogicalResult handleComplexGuid(KernelOp op, "
                + params
                + "* params, CommonNodeLoc loc, RewriterBase &rewriter) {\n"
            )

        for i in range(num_inputs):
            file_buf.write("\tValue val" + str(i) + " = op.getIn" + str(i) + "();\n")
        for i in range(num_outputs):
            file_buf.write("\tValue out" + str(i) + " = op.getOut" + str(i) + "();\n")

        file_buf.write("\n")

        for i in range(num_inputs):
            file_buf.write("\t// auto val" + str(i) + "Type = val" + str(i) + ".getType().dyn_cast<SynTensorType>();\n")
            file_buf.write("\t// auto val" + str(i) + "Shape = val" + str(i) + "Type.getShape();\n")
            file_buf.write("\t// auto val" + str(i) + "Rank = val" + str(i) + "Type.getRank();\n")

        file_buf.write("\n")

        for i in range(num_outputs):
            file_buf.write("\tauto out" + str(i) + "Type = out" + str(i) + ".getType().dyn_cast<SynTensorType>();\n")
            file_buf.write("\tauto out" + str(i) + "Shape = out" + str(i) + "Type.getShape();\n")
            file_buf.write("\tauto out" + str(i) + "Rank = out" + str(i) + "Type.getRank();\n")

        file_buf.write("\n")
        file_buf.write("\tType precisionType = op.getPrecisionType();\n")

        file_buf.write("\n")
        for i in range(num_outputs):
            file_buf.write("\tValue result" + str(i) + ";\n")
            file_buf.write(
                "\tauto resultType"
                + str(i)
                + " = SynTensorType::get(out"
                + str(i)
                + "Type.getMinShape(), out"
                + str(i)
                + "Type.getMaxShape(), precisionType);\n"
            )

        gen_file_buf = open(gen_file, "r")
        gen_lines = gen_file_buf.readlines()

        file_buf.write("\n\t// ToDo: Write Code here\n")

        # Process the op_backend fn
        # ToDo: Use clang parser to parse C++ file - https://eli.thegreenplace.net/2011/07/03/parsing-c-in-python-with-clang
        build_node_found = False
        build_op_found = False
        cast_helper_found = False
        constant_helper_found = False
        reshape_helper_found = False
        broadcast_helper_found = False
        build_cast_found = False
        build_constant_found = False
        build_reshape_found = False
        build_broadcast_found = False
        param_iter = -1
        param_list = []

        addNode_kernelOp = ""
        addNode_inputs = []
        addNode_shape = ""
        addNode_var = ""
        addNode_new_var_iter = 0

        op_info["outputs_var"] = []
        op_info["op"] = []
        op_info["inputs_var"] = []
        op_info["params"] = []

        def assignVar(line, str_if_not_equal1, str_if_not_equal2, addNode_new_var_iter):
            if line.find("=") != -1:
                addNode_var = line.strip()
                addNode_var = addNode_var.split(" ")[1]
            elif (
                (str_if_not_equal2 == "" and line.find(str_if_not_equal1) != -1)
                or line.find(str_if_not_equal1) != -1
                or (str_if_not_equal2 != "" and line.find(str_if_not_equal2) != -1)
            ):
                addNode_var = "var" + str(addNode_new_var_iter)
                addNode_new_var_iter += 1
            return addNode_new_var_iter, addNode_var

        def get_kernelop(guid):
            kernelop = ""
            is_first_char_of_word = True
            for i in guid:
                if i == "_":
                    is_first_char_of_word = True
                    continue
                if is_first_char_of_word:
                    kernelop += i.upper()
                else:
                    kernelop += i
                is_first_char_of_word = False

            kernelop += "KernelOp"
            return kernelop

        if is_opbackend_present and not "op_backend" in features_map:
            print("ERROR: hpu_op.yaml doesnt have op_backend field for this op. Exiting")
            exit()

        for line in gen_lines:
            # Indentation is important in the project else clang_check wouldnt have passed and this file wouldnt have been in the project
            if is_opbackend_present and line.find(features_map["op_backend"] + "::AddNode") != -1:
                file_buf.write("\n\t// " + features_map["op_backend"] + "::AddNode in file = " + gen_file + "\n")

            elif line.find("BuildNode(") != -1:
                build_node_found = True
                addNode_new_var_iter, addNode_var = assignVar(line, "BuildNode(", "", addNode_new_var_iter)
            elif build_node_found:
                param_iter += 1
                if param_iter == 2:
                    addNode_guid = ""
                    if line.find("MULT_GUID") != -1:
                        addNode_guid = "mult_"
                    else:
                        addNode_guid = line.split('"')[1]
                    addNode_kernelOp = get_kernelop(addNode_guid)
                    op_info["op"].append(addNode_guid[:-1])
                    op_info["params"].append("")
                elif param_iter == 3:
                    inp = line.strip()
                    inp = inp.split("{")[1]
                    inp = inp.split("},")[0]
                    addNode_inputs = inp.split(",")
                elif param_iter == 4:
                    words = line.strip()
                    words = words.split("{{")[1]
                    addNode_shape = words.split(",")[0]

            elif line.find("BuildOp(") != -1:
                build_op_found = True
                addNode_new_var_iter, addNode_var = assignVar(line, "BuildOp(", "", addNode_new_var_iter)
            elif build_op_found:
                param_iter += 1
                if param_iter == 1:
                    addNode_guid = ""
                    if line.find("MULT_GUID") != -1:
                        addNode_guid = "mult_"
                    else:
                        addNode_guid = line.split('"')[1]
                    addNode_kernelOp = get_kernelop(addNode_guid)
                    op_info["op"].append(addNode_guid[:-1])
                    op_info["params"].append("")
                elif param_iter == 2:
                    inp = line.strip()
                    inp = inp.split("{")[1]
                    inp = inp.split("},")[0]
                    addNode_inputs = inp.split(",")
                elif param_iter == 3:
                    words = line.strip()
                    words = words.split("{{")[1]
                    addNode_shape = words.split(",")[0]

            elif line.find("BuildCast(") != -1:
                cast_helper_found = True
                print("WARNING - Cast helper needs to be coded")
                addNode_new_var_iter, addNode_var = assignVar(line, "BuildCast(", "", addNode_new_var_iter)
                op_info["op"].append("cast")
                op_info["params"].append("")

            elif (
                line.find("ConstantHelper(") != -1
                or line.find("BuildConstant(") != -1
                or constant_helper_found
                or build_constant_found
            ):
                if line.find("ConstantHelper(") != -1:
                    constant_helper_found = True
                elif line.find("BuildConstant(") != -1:
                    build_constant_found = True

                addNode_new_var_iter, addNode_var = assignVar(
                    line, "ConstantHelper(", "BuildConstant(", addNode_new_var_iter
                )
                op_info["op"].append("constant")

                words = line.strip()
                words = words.split("(")
                if len(words) > 1:
                    words = words[1]
                words = words.split(")")[0]
                words = line.split(",")
                param_list.extend(words)

                if line.find(";") != -1:
                    addNode_kernelOp = "ConstantKernelOp"
                    addNode_shape = 1
                    file_buf.write("\tns_ConstantKernel::Params constantParams" + addNode_var + ";\n")
                    file_buf.write("\tconstantParams" + addNode_var + ".constant.f = ")
                    const_val = words[2]
                    if constant_helper_found:
                        const_val = words[1]
                    op_info["params"].append("constant = " + const_val)
                    file_buf.write(const_val)
                    file_buf.write(";\n")
                    addNode_inputs = ["constantParams" + addNode_var]

            elif (
                line.find("ReshapeHelper(") != -1
                or line.find("BuildReshape(") != -1
                or reshape_helper_found
                or build_reshape_found
            ):
                if line.find("ReshapeHelper(") != -1:
                    reshape_helper_found = True
                elif line.find("BuildReshape(") != -1:
                    build_reshape_found = True

                addNode_new_var_iter, addNode_var = assignVar(
                    line, "ReshapeHelper(", "BuildReshape(", addNode_new_var_iter
                )

                words = line.strip()
                words = words.split("(")
                if len(words) > 1:
                    words = words[1]
                words = words.split(")")[0]
                words = line.split(",")
                param_list.extend(words)

                if line.find(";") != -1:
                    addNode_kernelOp = "ReshapeKernelOp"
                    if reshape_helper_found:
                        addNode_shape = words[2]
                    else:
                        addNode_shape = words[3]
                    if reshape_helper_found:
                        addNode_inputs = [words[1], "nullptr"]
                    else:
                        addNode_inputs = [words[2], "nullptr"]

                    op_info["params"].append("operand_segment_sizes = array<i32: 1, 0, 0>")

            elif (
                line.find("BroadcastHelper(") != -1
                or line.find("BuildBroadcast(") != -1
                or build_broadcast_found
                or broadcast_helper_found
            ):
                if line.find("BroadcastHelper(") != -1:
                    broadcast_helper_found = True
                elif line.find("BuildBroadcast(") != -1:
                    build_broadcast_found = True

                addNode_new_var_iter, addNode_var = assignVar(
                    line, "BroadcastHelper(", "BuildBroadcast(", addNode_new_var_iter
                )

                words = line.strip()
                words = words.split("(")
                if len(words) > 1:
                    words = words[1]
                words = words.split(")")[0]
                words = line.split(",")
                param_list.extend(words)

                if line.find(";") != -1:
                    addNode_kernelOp = "BroadcastKernelOp"
                    if broadcast_helper_found:
                        addNode_shape = words[2]
                    else:
                        addNode_shape = words[3]
                    if broadcast_helper_found:
                        addNode_inputs = [words[1], "nullptr"]
                    else:
                        addNode_inputs = [words[2], "nullptr"]
                    op_info["params"].append("")

            if line.find(";") != -1 and addNode_kernelOp != "":
                file_buf.write(
                    "\tauto newType_" + addNode_var + " = SynTensorType::get(" + addNode_shape + ", precisionType);\n"
                )
                file_buf.write(
                    "\tValue "
                    + addNode_var
                    + " = rewriter.create<"
                    + addNode_kernelOp
                    + ">(loc, newType_"
                    + addNode_var
                    + ", precisionType, "
                )

                inputs_temp = []
                is_first = True
                for i in addNode_inputs:
                    if not is_first:
                        file_buf.write(", ")
                    else:
                        is_first = False
                    inps = i.split("[")[0].split(".")[0]
                    inps = inps.strip()
                    file_buf.write(inps)
                    inputs_temp.append(inps)

                op_info["inputs_var"].append(inputs_temp)
                op_info["outputs_var"].append(addNode_var)

                file_buf.write(");\n")

                build_node_found = False
                build_op_found = False
                cast_helper_found = False
                constant_helper_found = False
                reshape_helper_found = False
                broadcast_helper_found = False
                build_cast_found = False
                build_constant_found = False
                build_reshape_found = False
                build_broadcast_found = False
                param_iter = -1
                param_list = []

                addNode_kernelOp = ""
                addNode_inputs = []
                addNode_shape = ""

        file_buf.write("\n")
        if num_outputs == 1:
            file_buf.write("\trewriter.replaceOp(op, {result0});\n")
        else:
            for i in range(num_outputs):
                file_buf.write("\trewriter.replaceOp(op" + str(i) + ", {result" + str(i) + "});\n")
        file_buf.write("\treturn success();\n")
        file_buf.write("}\n")
        file_buf.write("\n")

        if params == "NoParams":
            file_buf.write(
                "LogicalResult extractFunctionalComplexGuid("
                + kernel_op
                + " op, CommonNodeLoc loc, RewriterBase &rewriter) {\n"
            )
        else:
            file_buf.write(
                "LogicalResult extractFunctionalComplexGuid("
                + kernel_op
                + " op, "
                + params
                + "* params, CommonNodeLoc loc, RewriterBase &rewriter) {\n"
            )

        for i in range(num_outputs):
            file_buf.write("\tValue output = op.getOut0();\n")
            file_buf.write("\tuint32_t outRank = output.getType().template cast<SynTensorType>().getRank();\n\n")

            if outputs_dim_max[i] == MAX_TENSOR_DIM:
                file_buf.write("\tif (out" + str(i) + "Rank > MAX_TENSOR_DIM)\n")
                if params == "NoParams":
                    file_buf.write("\t\treturn performNDOperation<" + kernel_op + ">(op, nullptr, loc, rewriter);")
                else:
                    file_buf.write("\t\treturn performNDOperation<" + kernel_op + ">(op, params, loc, rewriter);")
                file_buf.write("\n")

        if params == "NoParams":
            file_buf.write("return handleComplexGuid<" + kernel_op + ">(op, loc, rewriter);\n")
        else:
            file_buf.write("\treturn handleComplexGuid<" + kernel_op + ">(op, params, loc, rewriter);\n")
        file_buf.write("}\n\n")

        if is_fwd:
            kernel_op_fwd = kernel_op.split("KernelOp")[0]
            kernel_op_fwd += "FwdKernelOp"

            if params == "NoParams":
                file_buf.write(
                    "LogicalResult extractFunctionalComplexGuid("
                    + kernel_op_fwd
                    + " op, CommonNodeLoc loc, RewriterBase &rewriter) {\n"
                )
            else:
                file_buf.write(
                    "LogicalResult extractFunctionalComplexGuid("
                    + kernel_op_fwd
                    + " op, "
                    + params
                    + "* params, CommonNodeLoc loc, RewriterBase &rewriter) {\n"
                )

            for i in range(num_outputs):
                file_buf.write("\tValue output = op.getOut0();\n")
                file_buf.write("\tuint32_t outRank = output.getType().template cast<SynTensorType>().getRank();\n\n")

                if outputs_dim_max[i] == MAX_TENSOR_DIM:
                    file_buf.write("\tif (out" + str(i) + "Rank > MAX_TENSOR_DIM)\n")
                    if params == "NoParams":
                        file_buf.write(
                            "\t\treturn performNDOperation<" + kernel_op_fwd + ">(op, nullptr, loc, rewriter);"
                        )
                    else:
                        file_buf.write(
                            "\t\treturn performNDOperation<" + kernel_op_fwd + ">(op, params, loc, rewriter);"
                        )
                    file_buf.write("\n")

            if params == "NoParams":
                file_buf.write("return handleComplexGuid<" + kernel_op_fwd + ">(op, loc, rewriter);\n")
            else:
                file_buf.write("return handleComplexGuid<" + kernel_op_fwd + ">(op, params, loc, rewriter);\n")
            file_buf.write("}\n\n")

        file_buf.write("} // end namespace fuser\n")
        file_buf.write("} // end namespace tpc\n")

    writeHeader(op_file_buf)
    writeFn(op_file_buf, params, num_inputs, num_outputs, features_map, op_info)

######## Populate tpckernel.mlir
if make_changes_in_folders.count("tpc_fuser") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == tpckernel_ops_file
):
    print("Writing in file = ", tpckernel_ops_file)

    op_dtype = features_map["dtypes"].split("\n")[1]
    op_dtype = op_dtype.split("[")[1]
    op_dtype = op_dtype.split("]")[0]
    op_dtype = op_dtype.split(",")[0].lower()
    op_test_name = op + "_" + op_dtype

    # Find the appropriate location
    file_buf = open(tpckernel_ops_file, "r")
    lines = file_buf.readlines()
    indx = -1
    for line in lines:
        indx += 1
        if line.find("// CHECK-LABEL: func.func @") != -1:
            op_name = line.split("// CHECK-LABEL: func.func @")[1]

            if op_name > op_test_name:
                break

    op_test = "// CHECK-LABEL: func.func @" + op_test_name + "\n"
    op_test += "func.func @" + op_test_name + "("

    match_dims = []
    inputs_tensor = []
    is_first = True
    for i in range(num_inputs):
        if not is_first:
            op_test += ", "
        is_first = False

        op_test += "%in" + str(i) + ": tensor<"
        is_first1 = True
        itensor = []
        for j in range(inputs_dim_min[i]):
            if not is_first1:
                op_test += "x"
            is_first1 = False

            val = ""
            if inputs_any_match[i] == "Match" and len(match_dims) != 0:
                val = match_dims[j]
            else:
                val = random.randint(10, 20)
            if inputs_any_match[i] == "Match":
                match_dims.append(val)

            itensor.append(val)
            inputs_tensor.append(itensor)
            op_test += str(val)
        op_test += "x" + op_dtype + ">"
    op_test += ") -> "

    outputs_tensor = []
    is_first = True
    for i in range(num_outputs):
        if not is_first:
            op_test += ", "
        is_first = False

        op_test += "tensor<"
        is_first1 = True
        otensor = []
        for j in range(outputs_dim_min[i]):
            if not is_first1:
                op_test += "x"
            is_first1 = False

            val = ""
            if outputs_any_match[i] == "Match":
                val = match_dims[j]
            else:
                val = random.randint(10, 20)

            otensor.append(val)
            outputs_tensor.append(otensor)
            op_test += str(val)
        op_test += "x" + op_dtype + ">"

    op_test += " {\n"
    op_test += "  %res = tpckernel." + op + " "

    is_first = True
    for i in range(num_inputs):
        if not is_first:
            op_test += ", "
        is_first = False

        op_test += "%in" + str(i)

    repeat_line = " : ("

    is_first = True
    for i in range(len(inputs_tensor)):
        if not is_first:
            repeat_line += ", "
        is_first = False

        repeat_line += "tensor<"
        is_first1 = True
        for j in range(len(inputs_tensor[i])):
            if not is_first1:
                repeat_line += "x"
            is_first1 = False

            repeat_line += str(inputs_tensor[i][j])
        repeat_line += "x" + op_dtype + ">"

    repeat_line += ") -> <" + op_dtype + "> -> ("

    is_first = True
    for i in range(len(outputs_tensor)):
        if not is_first:
            repeat_line += ","
        is_first = False

        repeat_line += "tensor<"
        is_first1 = True
        for j in range(len(outputs_tensor[i])):
            if not is_first1:
                repeat_line += "x"
            is_first1 = False

            repeat_line += str(outputs_tensor[i][j])
        repeat_line += "x" + op_dtype + ">"

    repeat_line += ")\n"
    op_test += repeat_line

    op_test += "  return %res : "
    is_first = True
    for i in range(len(outputs_tensor)):
        if not is_first:
            op_test += ", "
        is_first = False

        op_test += "tensor<"
        is_first1 = True
        for j in range(len(outputs_tensor[i])):
            if not is_first1:
                op_test += "x"
            is_first1 = False

            op_test += str(outputs_tensor[i][j])
        op_test += "x" + op_dtype + ">"
    op_test += "\n"

    op_test += "}"
    op_test += "\n"
    op_test += "\n"
    op_test += "// CHECK: tpckernel." + op + " "
    is_first = True
    for i in range(num_inputs):
        if not is_first:
            op_test += ", "
        is_first = False

        op_test += "%{{.*}}"
    op_test += repeat_line
    op_test += "\n"
    op_test += "// -----"
    op_test += "\n"
    op_test += "\n"

    lines.insert(indx, op_test)

    # Write in file
    if is_release:
        with open(tpckernel_ops_file, "w") as f:
            lines = "".join(lines)
            f.write(lines)

######## Populate complex-guid.mlir
if make_changes_in_folders.count("tpc_fuser") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == complex_guid_file
):
    print("Writing in file = ", complex_guid_file)

    op_dtype = features_map["dtypes"].split("\n")[1]
    op_dtype = op_dtype.split("[")[1]
    op_dtype = op_dtype.split("]")[0]
    op_dtype = op_dtype.split(",")[0].lower()

    # Find the appropriate location
    file_buf = open(complex_guid_file, "r")
    lines = file_buf.readlines()
    pvs_indx = -1
    prvs_line = ""
    for line in lines:
        if line.find("// CHECK-LABEL: syn_rt.graph") != -1:
            op_name = pvs_line.split(" ")[1]

            if op_name > kernel_op:
                break

        pvs_indx += 1
        pvs_line = line

    op_test = "// " + kernel_op + "\n"
    op_test += "// CHECK-LABEL: syn_rt.graph\nsyn_rt.graph"

    op_test += "("
    repeat_line = ""
    match_dims = []
    inputs_tensor = []
    is_first = True
    for i in range(num_inputs):
        if not is_first:
            op_test += ", "
            repeat_line += ", "
        is_first = False

        op_test += "%in" + str(i) + ": !syn_rt.tensor<"
        repeat_line += "%in" + str(i) + ": !syn_rt.tensor<"
        is_first1 = True
        itensor = []
        for j in range(inputs_dim_min[i]):
            if not is_first1:
                op_test += "x"
                repeat_line += "x"
            is_first1 = False

            val = ""
            if inputs_any_match[i] == "Match" and len(match_dims) != 0:
                val = match_dims[j]
            else:
                val = random.randint(10, 20)
            if inputs_any_match[i] == "Match":
                match_dims.append(val)

            itensor.append(val)
            inputs_tensor.append(itensor)
            op_test += str(val)
            repeat_line += str(val)
        op_test += "x" + op_dtype + " persistent> loc(unknown)"
        repeat_line += "x" + op_dtype + " persistent>"
    op_test += ") -> "
    repeat_line += ") -> "

    op_test += "("
    repeat_line += "("
    outputs_tensor = []
    is_first = True
    for i in range(num_outputs):
        if not is_first:
            op_test += ", "
            repeat_line += ", "
        is_first = False

        op_test += "!syn_rt.tensor<"
        repeat_line += "!syn_rt.tensor<"
        is_first1 = True
        otensor = []
        for j in range(outputs_dim_min[i]):
            if not is_first1:
                op_test += "x"
                repeat_line += "x"
            is_first1 = False

            val = ""
            if outputs_any_match[i] == "Match":
                val = match_dims[j]
            else:
                val = random.randint(10, 20)

            otensor.append(val)
            outputs_tensor.append(otensor)
            op_test += str(val)
            repeat_line += str(val)
        op_test += "x" + op_dtype + " persistent>"
        repeat_line += "x" + op_dtype + " persistent>"

    op_test += ") {\n"
    repeat_line += "){\n"
    op_test += "  %out = tpckernel." + op + " "

    is_first = True
    for i in range(num_inputs):
        if not is_first:
            op_test += ", "
        is_first = False

        op_test += "%in" + str(i)

    # Params of op
    if op_info["is_optional"]:
        op_test += "{operand_segment_sizes = array<i32: "
        for i in range(num_inputs):
            op_test += "1, "
        op_test += "0>}"

    op_test += " : ("

    is_first = True
    for i in range(len(inputs_tensor)):
        if not is_first:
            op_test += ", "
        is_first = False

        op_test += "!syn_rt.tensor<"
        is_first1 = True
        for j in range(len(inputs_tensor[i])):
            if not is_first1:
                op_test += "x"
            is_first1 = False

            op_test += str(inputs_tensor[i][j])
        op_test += "x" + op_dtype + " persistent>"

    op_test += ") -> <" + op_dtype + "> -> ("

    is_first = True
    for i in range(len(outputs_tensor)):
        if not is_first:
            op_test += ","
        is_first = False

        op_test += "!syn_rt.tensor<"
        is_first1 = True
        for j in range(len(outputs_tensor[i])):
            if not is_first1:
                op_test += "x"
            is_first1 = False

            op_test += str(outputs_tensor[i][j])
        op_test += "x" + op_dtype + " persistent>"

    op_test += ") loc(#loc0)\n"

    op_test += "  syn_rt.exit %out : "
    is_first = True
    for i in range(len(outputs_tensor)):
        if not is_first:
            op_test += ", "
        is_first = False

        op_test += "!syn_rt.tensor<"
        is_first1 = True
        for j in range(len(outputs_tensor[i])):
            if not is_first1:
                op_test += "x"
            is_first1 = False

            op_test += str(outputs_tensor[i][j])
        op_test += "x" + op_dtype + " persistent> loc(#loc1)"
    op_test += "\n"

    op_test += "} loc(#loc1)"
    op_test += "\n"
    op_test += "\n"
    op_test += '#loc0 = loc("fuser.node#K_ID_0":0:0)\n'
    op_test += "#loc1 = loc(unknown)\n"
    op_test += "\n"
    op_test += "// CHECK-SAME: (" + repeat_line

    op_info["actual_outputs_var"] = []
    for i in range(len(op_info["outputs_var"])):
        op_test += "// CHECK-NEXT: "

        out_var = "%[[rout0"
        if i > 0:
            out_var += "_" + str(i - 1)
        out_var += ":.*]]"
        op_info["actual_outputs_var"].append(out_var)
        print("out_var = ", out_var)

        op_test += out_var + " = tpckernel." + op_info["op"][i] + " "

        is_first = True
        for j in range(len(op_info["inputs_var"][i])):
            if not is_first:
                op_test += ", "
            is_first = False

            inpt = op_info["inputs_var"][i][j]
            if inpt in op_info["outputs_var"]:
                indx = op_info["outputs_var"].index(inpt)
                inpt = op_info["actual_outputs_var"][indx]
            op_test += inpt

        if op_info["params"][i] != "":
            op_test += " {" + op_info["params"][i] + "}"

        op_test += " : <ToDo: Please fill tensors accordingly>\n"

    op_test += "// CHECK-NEXT:  syn_rt.exit "
    if len(op_info["actual_outputs_var"]) > 0:
        op_test += (
            op_info["actual_outputs_var"][len(op_info["actual_outputs_var"]) - 1]
            + " : <ToDo: Please fill tensors accordingly>\n"
        )
    else:
        op_test += "<ToDo: Please fill tensors accordingly>\n"
    op_test += "\n"
    op_test += "\n"
    op_test += "// -----"
    op_test += "\n"

    lines.insert(indx, op_test)

    # Write in file
    if is_release:
        with open(complex_guid_file, "w") as f:
            lines = "".join(lines)
            f.write(lines)

######## Populate opTest.hpp
if make_changes_in_folders.count("tpc_fuser") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == op_test_hpp_file
):
    print("Writing in file = ", op_test_hpp_file)

    op_test_hpp_file_buf = open(op_test_hpp_file, "w")

    op_test_hpp_file_buf.write(
        "//===--------- " + op_file_name + "Test.hpp - " + op_file_name + " ComplexGuid Test ---------===//\n"
    )
    op_test_hpp_file_buf.write("//\n")
    op_test_hpp_file_buf.write("// Copyright (C) 2022 HabanaLabs, Ltd.\n")
    op_test_hpp_file_buf.write("// All Rights Reserved.\n")
    op_test_hpp_file_buf.write("// Unauthorized copying of this file, via any medium is strictly prohibited.\n")
    op_test_hpp_file_buf.write("// Proprietary and confidential.\n")
    op_test_hpp_file_buf.write("//\n")
    op_test_hpp_file_buf.write("//===----------------------------------------------------------------------===//\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("#ifndef MLIR_PYTENETTESTS_COMPLEXGUIDTESTS_" + op + "_HPP\n")
    op_test_hpp_file_buf.write("#define MLIR_PYTENETTESTS_COMPLEXGUIDTESTS_" + op + "_HPP\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write('#include "TestBase.hpp"\n')
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("namespace ComplexGuidTest {\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write(
        "enum dim {\n  oneDim = 1,\n  twoDim,\n  threeDim,\n  fourDim,\n  fiveDim,\n  sixDim,\n  sevenDim,\n  eightDim,\n  nineDim\n};\n"
    )
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("template <const synDataType inType = syn_type_float,\n")

    def getDim(num):
        if num == 1:
            return "oneDim"
        elif num == 2:
            return "twoDim"
        elif num == 3:
            return "threeDim"
        elif num == 4:
            return "fourDim"
        elif num == 5:
            return "fiveDim"
        elif num == 6:
            return "sixDim"
        elif num == 7:
            return "sevenDim"
        elif num == 8:
            return "eightDim"
        elif num == 9:
            return "nineDim"

    inTensorDim = 0
    for i in range(len(inputs_dim_min)):
        if inputs_dim_min[i] == inputs_dim_max[i] and inputs_dim_max[i] > inTensorDim:
            inTensorDim = inputs_dim_max[i]
        elif inputs_dim_max[i] == MAX_TENSOR_DIM:
            inTensorDim = MAX_TENSOR_DIM
            break

    if op_info["is_optional"]:
        inTensorDim += 1

    outTensorDim = 0
    for i in range(len(outputs_dim_min)):
        if outputs_dim_min[i] == outputs_dim_max[i] and outputs_dim_max[i] > outTensorDim:
            outTensorDim = outputs_dim_max[i]
        elif outputs_dim_max[i] == MAX_TENSOR_DIM:
            outTensorDim = MAX_TENSOR_DIM
            break

    op_test_hpp_file_buf.write("          const unsigned int inTensorDim = " + getDim(inTensorDim) + ",\n")
    op_test_hpp_file_buf.write("          const unsigned outTensorDim = " + getDim(outTensorDim) + ">\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("class " + op_file_name + "Test\n")
    op_test_hpp_file_buf.write("    : public TestBase,\n")
    op_test_hpp_file_buf.write("      public ::testing::WithParamInterface<testing::tuple<\n")
    op_test_hpp_file_buf.write("          synDeviceType, std::tuple<")

    is_first = True
    for i in range(inTensorDim):
        if not is_first:
            op_test_hpp_file_buf.write(", ")
        is_first = False
        op_test_hpp_file_buf.write("int")

    optional_params_permutation = []
    optional_inp_list = []
    op_test_hpp_file_buf.write(">>> {")
    if op_info["is_optional"]:
        op_test_hpp_file_buf.write(" // last int - \n")

        optional_inp = 0
        for i in range(len(op_info["optional_inp"])):
            if op_info["optional_inp"][i]:
                optional_inp += 1
                optional_inp_list.append("input" + str(i))
        optional_inp_combinations = list(itertools.product([0, 1], repeat=math.factorial(optional_inp)))
        for i in range(len(optional_inp_combinations)):
            op_test_hpp_file_buf.write("// " + str(i) + " (")
            tt = np.binary_repr(i, width=len(optional_inp_list))
            l = []
            is_first = True
            for j in range(len(optional_inp_list)):
                if not is_first:
                    op_test_hpp_file_buf.write(", ")
                is_first = False

                if tt[j] == "0":
                    op_test_hpp_file_buf.write("without " + optional_inp_list[j])
                    l.append("w/o")
                else:
                    op_test_hpp_file_buf.write("with " + optional_inp_list[j])
                    l.append("w")
            optional_params_permutation.append(l)
            op_test_hpp_file_buf.write(")\n")

    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("public:\n")
    op_test_hpp_file_buf.write("  struct GetName {\n")
    op_test_hpp_file_buf.write("    template <class ParamType>\n")
    op_test_hpp_file_buf.write("    std::string\n")
    op_test_hpp_file_buf.write("    operator()(const ::testing::TestParamInfo<ParamType> &info) const {\n")
    op_test_hpp_file_buf.write("      ::std::stringstream ss;\n")
    op_test_hpp_file_buf.write("      ss << GetDeviceName(std::get<0>(info.param));\n")
    op_test_hpp_file_buf.write('      ss << "ifm_" << ')

    is_first = True
    for i in range(inTensorDim):
        if not is_first:
            op_test_hpp_file_buf.write(' << "x" << ')
        is_first = False
        op_test_hpp_file_buf.write("std::get<" + str(i) + ">(ParamTuple)")

    op_test_hpp_file_buf.write(";\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("      std::string name = ss.str();\n")
    op_test_hpp_file_buf.write("      std::replace(name.begin(), name.end(), '-', '_');\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("      return name;\n")
    op_test_hpp_file_buf.write("    }\n")
    op_test_hpp_file_buf.write("  };\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("  " + op_file_name + "Test() {\n")
    op_test_hpp_file_buf.write("    // Set the device type for acquiring the same.\n")
    op_test_hpp_file_buf.write("    deviceType = ::testing::get<0>(GetParam());\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("    // Set the framework to be used for reference calculation.\n")
    op_test_hpp_file_buf.write("    referenceFwType = TORCH_REFERENCE_FW;\n")
    op_test_hpp_file_buf.write("  }\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("protected:\n")
    op_test_hpp_file_buf.write("  const synDataType outType = inType;\n")

    ifmInitializer_count = 0
    for i in range(len(op_info["optional_inp"])):
        # optional
        if op_info["optional_inp"][i]:
            ifmInitializer_count += 1
            op_test_hpp_file_buf.write("  unsigned int ifmInitializer" + str(i) + ";\n")
            break
        else:
            ifmInitializer_count += 1
            if op_info["is_optional"]:
                op_test_hpp_file_buf.write(
                    "  unsigned int ifmInitializer" + str(i) + "[" + getDim(inTensorDim - 1) + "];\n"
                )
            else:
                op_test_hpp_file_buf.write(
                    "  unsigned int ifmInitializer" + str(i) + "[" + getDim(inTensorDim) + "];\n"
                )

    if op_info["is_optional"]:
        op_test_hpp_file_buf.write("  unsigned int ofmInitializer[" + getDim(inTensorDim - 1) + "];\n")
    else:
        op_test_hpp_file_buf.write("  unsigned int ofmInitializer[" + getDim(inTensorDim) + "];\n")
    op_test_hpp_file_buf.write("\n")

    optional_ifm_var = []
    for i in range(len(op_info["optional_inp"])):
        if op_info["optional_inp"][i] == 0:
            op_test_hpp_file_buf.write("  unsigned int ifm" + str(i) + ";\n")
        else:
            op_test_hpp_file_buf.write("  unsigned int *ifm" + str(i) + ";\n")
            optional_ifm_var.append("ifm" + str(i))

    op_test_hpp_file_buf.write("  unsigned int ofm;\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("  virtual void SetFmInitializer() {\n")

    optional_ifmInitializer_var = "ifmInitializer"
    for j in range(len(op_info["optional_inp"])):
        if op_info["optional_inp"][j] == 0:
            if inputs_dim_max[j] == inputs_dim_min[j]:
                for i in range(inputs_dim_max[j]):
                    op_test_hpp_file_buf.write(
                        "    ifmInitializer"
                        + str(j)
                        + "["
                        + str(i)
                        + "] = ::testing::get<"
                        + str(i)
                        + ">(TestParam());\n"
                    )
            else:
                itd = inTensorDim
                if op_info["is_optional"]:
                    itd -= 1
                for i in range(itd):
                    op_test_hpp_file_buf.write(
                        "    ifmInitializer"
                        + str(j)
                        + "["
                        + str(i)
                        + "] = ::testing::get<"
                        + str(i)
                        + ">(TestParam());\n"
                    )
        else:
            op_test_hpp_file_buf.write(
                "    ifmInitializer" + str(j) + " = ::testing::get<" + str(inTensorDim - 1) + ">(TestParam());\n"
            )
            optional_ifmInitializer_var += str(j)
            break

    op_test_hpp_file_buf.write("\n")

    for i in range(outTensorDim):
        op_test_hpp_file_buf.write("    ofmInitializer[" + str(i) + "] = ifmInitializer0[" + str(i) + "];\n")

    op_test_hpp_file_buf.write("  }\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("  void RunTest() {\n")
    op_test_hpp_file_buf.write("    SetFmInitializer();\n")
    op_test_hpp_file_buf.write("\n")

    is_first = True
    for i in range(len(op_info["optional_inp"])):
        if op_info["optional_inp"][i] == 0:
            continue

        if is_first:
            op_test_hpp_file_buf.write("    // Optional parameters\n")
            op_test_hpp_file_buf.write("    const unsigned int dim = 1;\n")
        is_first = False

        op_test_hpp_file_buf.write("    unsigned int optional_inp" + str(i) + "[dim];\n")
        op_test_hpp_file_buf.write("    optional_inp" + str(i) + "[0] = " + op_info["optional_inp_def_val"][i] + ";\n")
        op_test_hpp_file_buf.write("    unsigned int ifm" + str(i) + "_1;\n")
        op_test_hpp_file_buf.write("\n")

    is_first = True
    for i in range(len(optional_params_permutation)):
        # While passing optional params to torch::op, we cannot pass later optional param values and not pass earlier param values.
        # e.g. torch::op(optional1 = 1.0, optional2 = 2.0)
        # We cannot do torch::op(optional2). We need to pass optional1 also.
        # But we can do torch::op(optional1). The order of optional1 and optional2 matter
        found_without = False
        need_to_contnue = False
        for j in range(len(optional_ifm_var)):
            if found_without and optional_params_permutation[i][j] == "w":
                need_to_contnue = True
                break
            if optional_params_permutation[i][j] == "w/o":
                found_without = True
        if need_to_contnue:
            continue

        # If ifm are in order then write in file
        if is_first:
            op_test_hpp_file_buf.write("    if")
            op_test_hpp_file_buf.write(" (" + optional_ifmInitializer_var + " == " + str(i) + ") {\n")
        elif not is_first and i < len(optional_params_permutation) - 1:
            op_test_hpp_file_buf.write("     } else if")
            op_test_hpp_file_buf.write(" (" + optional_ifmInitializer_var + " == " + str(i) + ") {\n")
        else:
            op_test_hpp_file_buf.write("     } else {\n")
        is_first = False

        for j in range(len(optional_ifm_var)):
            if optional_params_permutation[i][j] == "w/o":
                op_test_hpp_file_buf.write("      " + optional_ifm_var[j] + " = nullptr;\n")
            else:
                op_test_hpp_file_buf.write(
                    "      ifm"
                    + str(j)
                    + '_1 = createDeviceTensor("Optional'
                    + str(j)
                    + '", INPUT_TENSOR, MEM_INIT_RANDOM_POSITIVE, nullptr, scalarType, dim, optional_inp'
                    + str(j)
                    + ");\n"
                )
                op_test_hpp_file_buf.write("      ifm" + str(j) + " = &ifm" + str(j) + "_1;\n")

    if op_info["is_optional"]:
        op_test_hpp_file_buf.write("     }\n")
    op_test_hpp_file_buf.write("    // Create device input/output tensors.\n")

    for i in range(len(op_info["optional_inp"])):
        if op_info["optional_inp"][i] == 1:
            break

        itsrdim = MAX_TENSOR_DIM
        if inputs_dim_min[i] == inputs_dim_max[i]:
            itsrdim = inputs_dim_max[i]

        if (op_info["is_optional"] and itsrdim == inTensorDim - 1) or (
            not op_info["is_optional"] and itsrdim == inTensorDim
        ):
            op_test_hpp_file_buf.write(
                "    ifm"
                + str(i)
                + ' = createDeviceTensor("A'
                + str(i)
                + '", INPUT_TENSOR, MEM_INIT_ALL_ONE, nullptr, inType, inTensorDim, ifmInitializer'
                + str(i)
                + ");\n"
            )
        else:
            op_test_hpp_file_buf.write(
                "    ifm"
                + str(i)
                + ' = createDeviceTensor("A'
                + str(i)
                + '", INPUT_TENSOR, MEM_INIT_ALL_ONE, nullptr, inType, '
                + getDim(itsrdim)
                + ", ifmInitializer"
                + str(i)
                + ");\n"
            )

    op_test_hpp_file_buf.write(
        '    ofm = createDeviceTensor("'
        + op_file_name
        + '", OUTPUT_TENSOR, MEM_INIT_ALL_ZERO, nullptr, outType, outTensorDim, ofmInitializer);\n'
    )
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write('    std::string nodeName = "' + op + '";\n')
    op_test_hpp_file_buf.write('    std::string guidName = "' + op + '_";\n')

    if is_fwd:
        op_test_hpp_file_buf.write("    if (deviceType != synDeviceGreco) {\n")
        op_test_hpp_file_buf.write('        std::string nodeName = "' + op + '_fwd";\n')
        op_test_hpp_file_buf.write('        std::string guidName = "' + op + '_fwd_";\n')
        op_test_hpp_file_buf.write("    }\n")

    op_test_hpp_file_buf.write("\n")

    op_dtype = features_map["dtypes"].split("\n")[1]
    op_dtype = op_dtype.split("[")[1]
    op_dtype = op_dtype.split("]")[0]
    op_dtype = op_dtype.split(", ")
    is_first = True
    for i in op_dtype:
        if is_first:
            op_test_hpp_file_buf.write("    if ")
        else:
            op_test_hpp_file_buf.write("    } else if ")
        is_first = False

        if i == "F32":
            op_test_hpp_file_buf.write("(outType == syn_type_float) {\n")
            op_test_hpp_file_buf.write('      guidName += "f32";\n')
        elif i == "BF16":
            op_test_hpp_file_buf.write("(outType == syn_type_bf16) {\n")
            op_test_hpp_file_buf.write('      guidName += "bf16";\n')
        elif i == "F16":
            op_test_hpp_file_buf.write("(outType == syn_type_f16) {\n")
            op_test_hpp_file_buf.write('      guidName += "f16";\n')
        elif i == "I32":
            op_test_hpp_file_buf.write("(outType == syn_type_int) {\n")
            op_test_hpp_file_buf.write('      guidName += "i32";\n')
    op_test_hpp_file_buf.write("    }\n")

    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("    // Add node to graph.\n")

    op_test_hpp_file_buf.write("    TensorIndices inp = {")
    inp_list_non_optional = ""
    is_first = True
    for i in range(len(op_info["optional_inp"])):
        if op_info["optional_inp"][i] == 1:
            break
        if not is_first:
            op_test_hpp_file_buf.write(", ")
            inp_list_non_optional += ", "
        is_first = False
        op_test_hpp_file_buf.write("ifm" + str(i))
        inp_list_non_optional += "ifm" + str(i)
    op_test_hpp_file_buf.write("};\n")

    is_first = True
    for i in range(len(optional_params_permutation)):
        # While passing optional params to torch::op, we cannot pass later optional param values and not pass earlier param values.
        # e.g. torch::op(optional1 = 1.0, optional2 = 2.0)
        # We cannot do torch::op(optional2). We need to pass optional1 also.
        # But we can do torch::op(optional1). The order of optional1 and optional2 matter
        found_without = False
        need_to_contnue = False
        for j in range(len(optional_ifm_var)):
            if found_without and optional_params_permutation[i][j] == "w":
                need_to_contnue = True
                break
            if optional_params_permutation[i][j] == "w/o":
                found_without = True
        if need_to_contnue:
            continue

        # If order of optional params is maintained then call torch::op
        is_if_elif = True
        if is_first:
            op_test_hpp_file_buf.write("    if (")
        elif not is_first and i < len(optional_params_permutation) - 1:
            op_test_hpp_file_buf.write("    else if (")
        else:
            op_test_hpp_file_buf.write("    else")
            is_if_elif = False
        is_first = False

        inp_list = inp_list_non_optional
        is_first1 = True
        for j in range(len(optional_ifm_var)):
            if not is_first1:
                if is_if_elif:
                    op_test_hpp_file_buf.write(" && ")
            is_first1 = False

            if optional_params_permutation[i][j] == "w":
                if is_if_elif:
                    op_test_hpp_file_buf.write(optional_ifm_var[j] + " != nullptr")
                inp_list += ", " + optional_ifm_var[j]
            else:
                if is_if_elif:
                    op_test_hpp_file_buf.write(optional_ifm_var[j] + " == nullptr")

        if is_if_elif:
            op_test_hpp_file_buf.write(")")

        op_test_hpp_file_buf.write("\n")
        op_test_hpp_file_buf.write("      inp = {" + inp_list + "};\n")

    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("    addNodeToGraph(guidName.c_str(), inp, {ofm}, nullptr, 0, nodeName.c_str());\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("    // Run on device.\n")
    op_test_hpp_file_buf.write("    compileAndRun();\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("    // Create pytenet reference output tensors.\n")
    op_test_hpp_file_buf.write(
        '    unsigned int ofmRef = createReferenceTensor("'
        + op_file_name
        + 'Ref", outType, outTensorDim, ofmInitializer);\n'
    )
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("    // Run reference.\n")

    # non-optional params only
    is_first1 = True
    if not op_info["is_optional"]:
        op_test_hpp_file_buf.write("\n      pytenetTensors[ofmRef].torchTensor = " + torch_op + "(")
        for j in range(len(op_info["optional_inp"])):
            if not is_first1:
                op_test_hpp_file_buf.write(", ")
            is_first1 = False

            op_test_hpp_file_buf.write("\n          pytenetTensors[ifm" + str(j) + "].torchTensor")
        op_test_hpp_file_buf.write(");\n")

    # Optional and non-optional params
    is_first = True
    for i in range(len(optional_params_permutation)):
        # While passing optional params to torch::op, we cannot pass later optional param values and not pass earlier param values.
        # e.g. torch::op(optional1 = 1.0, optional2 = 2.0)
        # We cannot do torch::op(optional2). We need to pass optional1 also.
        # But we can do torch::op(optional1). The order of optional1 and optional2 matter
        found_without = False
        need_to_contnue = False
        for j in range(len(optional_ifm_var)):
            if found_without and optional_params_permutation[i][j] == "w":
                need_to_contnue = True
                break
            if optional_params_permutation[i][j] == "w/o":
                found_without = True
        if need_to_contnue:
            continue

        # If order of optional params is maintained then call torch::op
        if is_first:
            op_test_hpp_file_buf.write("    if")
        else:
            op_test_hpp_file_buf.write("    else if")
        is_first = False

        is_first1 = True
        op_test_hpp_file_buf.write("(")
        for j in range(len(optional_ifm_var)):
            if not is_first1:
                op_test_hpp_file_buf.write(" && ")
            is_first1 = False

            op_test_hpp_file_buf.write(optional_ifm_var[j])
            if optional_params_permutation[i][j] == "w/o":
                op_test_hpp_file_buf.write(" == nullptr")
            else:
                op_test_hpp_file_buf.write(" != nullptr")
        op_test_hpp_file_buf.write(") {")

        is_first1 = True
        num_non_optional_inputs = 0
        op_test_hpp_file_buf.write("\n      pytenetTensors[ofmRef].torchTensor = " + torch_op + "(")
        for j in range(len(op_info["optional_inp"])):
            if not is_first1 and (
                op_info["optional_inp"][j] == 0
                or (
                    op_info["optional_inp"][j] == 1
                    and optional_params_permutation[i][j - num_non_optional_inputs] == "w"
                )
            ):
                op_test_hpp_file_buf.write(", ")
            is_first1 = False

            if op_info["optional_inp"][j] == 0:
                op_test_hpp_file_buf.write("\n          pytenetTensors[ifm" + str(j) + "].torchTensor")
                num_non_optional_inputs += 1
            elif optional_params_permutation[i][j - num_non_optional_inputs] == "w":
                op_test_hpp_file_buf.write(
                    "\n          pytenetTensors[*ifm" + str(j) + "].torchTensor[0].item<double>()"
                )
        op_test_hpp_file_buf.write(");\n")

    op_test_hpp_file_buf.write("\n")

    op_test_hpp_file_buf.write("    pytenetDataType pdt;\n")
    op_dtype = features_map["dtypes"].split("\n")[1]
    op_dtype = op_dtype.split("[")[1]
    op_dtype = op_dtype.split("]")[0]
    op_dtype = op_dtype.split(", ")
    for i in op_dtype:
        if i == "F32":
            op_test_hpp_file_buf.write("    if (outType == syn_type_float) {\n")
            op_test_hpp_file_buf.write("      pdt = pytenet_type_float;\n")
        elif i == "BF16":
            op_test_hpp_file_buf.write("    if (outType == syn_type_bf16) {\n")
            op_test_hpp_file_buf.write("      pdt = pytenet_type_bf16;\n")
        elif i == "F16":
            op_test_hpp_file_buf.write("    if (outType == syn_type_f16) {\n")
            op_test_hpp_file_buf.write("      pdt = pytenet_type_f16;\n")
        elif i == "I32":
            op_test_hpp_file_buf.write("    if (outType == syn_type_int) {\n")
            op_test_hpp_file_buf.write("      pdt = pytenet_type_int;\n")
    op_test_hpp_file_buf.write("    }\n")

    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write(
        "    ASSERT_TRUE(validateResults(pytenetTensors[ofmRef].torchTensor.data_ptr(), hostBuffers[ofm], tensorDescs[ofmRef].numElements, pdt));\n"
    )
    op_test_hpp_file_buf.write("  }\n")
    op_test_hpp_file_buf.write("};\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("} // namespace ComplexGuidTest\n")
    op_test_hpp_file_buf.write("\n")
    op_test_hpp_file_buf.write("#endif // MLIR_PYTENETTESTS_COMPLEXGUIDTESTS_" + op + "_HPP\n")

######## Populate opTest.cpp
if make_changes_in_folders.count("tpc_fuser") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == op_test_cpp_file
):
    print("Writing in file = ", op_test_cpp_file)

    op_test_cpp_file_buf = open(op_test_cpp_file, "w")

    op_test_cpp_file_buf.write(
        "//===--------- " + op_file_name + "Test.cpp - " + op_file_name + " ComplexGuid Test ---------===//\n"
    )
    op_test_cpp_file_buf.write("//\n")
    op_test_cpp_file_buf.write("// Copyright (C) 2022 HabanaLabs, Ltd.\n")
    op_test_cpp_file_buf.write("// All Rights Reserved.\n")
    op_test_cpp_file_buf.write("// Unauthorized copying of this file, via any medium is strictly prohibited.\n")
    op_test_cpp_file_buf.write("// Proprietary and confidential.\n")
    op_test_cpp_file_buf.write("//\n")
    op_test_cpp_file_buf.write("//===----------------------------------------------------------------------===//\n")
    op_test_cpp_file_buf.write("\n")
    op_test_cpp_file_buf.write('#include "' + op_file_name + 'Test.hpp"\n')
    op_test_cpp_file_buf.write("\n")
    op_test_cpp_file_buf.write("using namespace ComplexGuidTest;\n")
    op_test_cpp_file_buf.write("\n")

    op_dtype = features_map["dtypes"].split("\n")[1]
    op_dtype = op_dtype.split("[")[1]
    op_dtype = op_dtype.split("]")[0]
    op_dtype = op_dtype.replace(" ", "")
    op_dtype = op_dtype.split(",")

    inTensorDim_test = [2, MAX_TENSOR_DIM]
    if inTensorDim < MAX_TENSOR_DIM:
        inTensorDim_test = [inputs_dim_min[0]]

    test_cases = {"class_name": [], "dtype": [], "dims": [], "dimTypeShort": [], "dimType": []}

    def getDtypeName(dt):
        dt = dt.lower()
        if dt == "f32":
            return "syn_type_float"
        elif dt == "bf16":
            return "syn_type_bf16"
        elif dt == "f16":
            return "syn_type_f16"
        elif dt == "i32":
            return "syn_type_int"

    for i in range(len(inTensorDim_test)):
        for j in range(len(op_dtype)):
            if not op_info["is_optional"]:  # No options
                test_cases["class_name"].append(op_file_name + op_dtype[j].upper() + str(inTensorDim_test[i]) + "Test")
                test_cases["dtype"].append(getDtypeName(op_dtype[j]))
                test_cases["dimType"].append(getDim(inTensorDim_test[i]))
                test_cases["dimTypeShort"].append(op_dtype[j].lower())

                d = []
                for l in range(inTensorDim_test[i]):
                    val = random.randint(1, 50)
                    d.append(val)
                test_cases["dims"].append(d)

            else:
                for k in range(len(optional_params_permutation)):
                    testname = ""
                    for l in range(len(optional_params_permutation[k])):
                        if optional_params_permutation[k][l] == "w":
                            testname += "WI" + str(l)
                        else:
                            testname += "WoI" + str(l)

                    d = []
                    for m in range(inTensorDim - 1):
                        if m >= inTensorDim_test[i]:
                            d.append(0)
                        else:
                            val = random.randint(1, 2000)
                            d.append(val)
                    d.append(k)
                    test_cases["dims"].append(d)

                    test_cases["class_name"].append(
                        op_file_name + op_dtype[j].upper() + str(inTensorDim_test[i]) + "Test" + testname
                    )
                    test_cases["dtype"].append(getDtypeName(op_dtype[j]))
                    test_cases["dimType"].append(getDim(inTensorDim_test[i]))
                    test_cases["dimTypeShort"].append(op_dtype[j].lower())

    for i in range(len(test_cases["class_name"])):
        op_test_cpp_file_buf.write(
            "class "
            + test_cases["class_name"][i]
            + " : public "
            + op_file_name
            + "Test<"
            + test_cases["dtype"][i]
            + ", "
            + test_cases["dimType"][i]
            + ", "
            + test_cases["dimType"][i]
            + "> {};\n"
        )
        op_test_cpp_file_buf.write(
            "TEST_P("
            + test_cases["class_name"][i]
            + ", "
            + op
            + "_"
            + test_cases["dimTypeShort"][i]
            + ") { RunTest(); }\n"
        )
        op_test_cpp_file_buf.write("INSTANTIATE_TEST_SUITE_P(\n")
        op_test_cpp_file_buf.write("    sanity, " + test_cases["class_name"][i] + ",\n")
        op_test_cpp_file_buf.write(
            "    ::testing::Combine(\n        ::testing::Values(synDeviceGaudi, synDeviceGaudi2, synDeviceGreco),\n"
        )
        op_test_cpp_file_buf.write("        ::testing::Values(std::make_tuple(")

        is_first = True
        for j in range(inTensorDim):
            if not is_first:
                op_test_cpp_file_buf.write(", ")
            is_first = False

            if j < len(test_cases["dims"][i]):
                op_test_cpp_file_buf.write(str(test_cases["dims"][i][j]))
            else:
                op_test_cpp_file_buf.write("0")

        op_test_cpp_file_buf.write("))),\n")
        op_test_cpp_file_buf.write("    " + test_cases["class_name"][i] + "::GetName());\n\n")

######## Populate CMakeList file
if make_changes_in_folders.count("tpc_fuser") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == cmakelist_file
):
    print("Writing in file = ", cmakelist_file)

    cmakelist_file_buf = open(cmakelist_file, "r")
    lines = cmakelist_file_buf.readlines()

    indx = -1
    for line in lines:
        indx += 1

        if line.split("/")[0].lower().replace(" ", "") > op_folder.lower() or (
            line.split("/")[0].lower().replace(" ", "") == op_folder.lower()
            and line.split("/")[1] > (op_file_name + ".cpp")
        ):
            break

    lines.insert(indx, "  " + op_folder + "/" + op_file_name + ".cpp\n")

    # Write in file
    if is_release:
        with open(cmakelist_file, "w") as f:
            lines = "".join(lines)
            f.write(lines)

######## Populate ShapeInference.cpp file
if make_changes_in_folders.count("tpc_fuser") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == shape_inference_file
):
    print("Writing in file = ", shape_inference_file)

    shape_inference_file_buf = open(shape_inference_file, "r")
    lines = shape_inference_file_buf.readlines()

    found1 = False
    found2 = False
    fn_found = False
    fn_line_indx = -1
    entry_found = False
    entry_line_indx = -1
    for line in lines:
        if not fn_found:
            fn_line_indx += 1
        if not entry_found:
            entry_line_indx += 1

        if found1 and not fn_found:
            fn_name = line.split("GetShapeInference")

            if len(fn_name) > 1:
                fn_name = fn_name[1].split("(")[0]
            else:
                found1 = False
                continue

            if fn_name > op_file_name:
                found1 = False
                fn_found = True
                fn_line_indx -= 1

        if found2 and not entry_found:
            fn_name = line.split("GetShapeInference")
            if len(fn_name) > 1:
                fn_name = fn_name[1].split("}")[0]
            elif line == "\n":
                found2 = False
                continue
            else:
                continue

            if fn_name > op_file_name:
                entry_found = True
                found2 = False

        if line.find("static ShapeInferenceReturn_t") != 1:
            found1 = True
        if line.find("static constexpr std") != -1:
            found2 = True

    # insert fn
    fn_lines_header = "static ShapeInferenceReturn_t\n"
    fn_lines_header += "GetShapeInference" + op_file_name + "(const ShapeInferenceParams *inputParams,\n"
    fn_lines_header += "                         ShapeInferenceOutput *outputData) {\n"

    fn_lines_fwd_header = ""
    if is_fwd:
        fn_lines_fwd_header = "static ShapeInferenceReturn_t\n"
        fn_lines_fwd_header += "GetShapeInference" + op_file_name + "Fwd(const ShapeInferenceParams *inputParams,\n"
        fn_lines_fwd_header += "                         ShapeInferenceOutput *outputData) {\n"

    fn_lines = "  if (!inputParams || !outputData) {\n"
    fn_lines += '    api_error() << "invalid argument SIF_NULL_PTR\\n";\n'
    fn_lines += "    return SIF_NULL_PTR;\n"
    fn_lines += "  }\n"
    fn_lines += "\n"

    if op_info["is_optional"]:
        min_num_inputs = 0
        max_num_inputs = 0
        for i in range(len(op_info["optional_inp"])):
            if op_info["optional_inp"][i] == 1:
                max_num_inputs += 1
                continue
            min_num_inputs += 1
        fn_lines += (
            "  if (inputParams->inputTensorsNr < "
            + str(min_num_inputs)
            + " || inputParams->inputTensorsNr > "
            + str(max_num_inputs)
            + ") {\n"
        )
    else:
        fn_lines += "  if (inputParams->inputTensorsNr != " + str(len(op_info["optional_inp"])) + ") {\n"

    fn_lines += '    api_error() << "invalid argument SIF_INCOMPATIBLE_INPUT_COUNT\\n";\n'
    fn_lines += "    return gcapi::SIF_INCOMPATIBLE_INPUT_COUNT;\n"
    fn_lines += "  }\n"
    fn_lines += "\n"
    fn_lines += "  if (inputParams->outputTensorsNr != 1) {\n"
    fn_lines += '    api_error() << "invalid argument SIF_INCOMPATIBLE_OUTPUT_COUNT\\n";\n'
    fn_lines += "    return gcapi::SIF_INCOMPATIBLE_OUTPUT_COUNT;\n"
    fn_lines += "  }\n"
    fn_lines += "\n"
    fn_lines += "  // ofm0 is same size of ifm0\n"
    fn_lines += "  outputData->outputTensors[0]->dims = inputParams->inputTensors[0]->dims;\n"
    fn_lines += "  memcpy(outputData->outputTensors[0]->sizes,\n"
    fn_lines += "         inputParams->inputTensors[0]->sizes,\n"
    fn_lines += "         inputParams->inputTensors[0]->dims * sizeof(unsigned));\n"
    fn_lines += "\n"
    fn_lines += "  return SIF_SUCCESS;\n"
    fn_lines += "}\n\n"

    final_fn_lines = fn_lines_header + fn_lines
    if is_fwd:
        final_fn_lines += fn_lines_fwd_header + fn_lines

    # insert entry
    entry_lines = '        {"' + op + '", GetShapeInference' + op_file_name + "},\n"
    if is_fwd:
        entry_lines += '        {"' + op + '_fwd", GetShapeInference' + op_file_name + "Fwd},\n"

    lines.insert(entry_line_indx, entry_lines)
    lines.insert(fn_line_indx, final_fn_lines)

    # Write in file
    if is_release:
        with open(shape_inference_file, "w") as f:
            lines = "".join(lines)
            f.write(lines)

######## Populate Hpu_Op.yaml file
if make_changes_in_folders.count("pytorch-integration") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == hpu_yaml_file
):
    print("Writing in file = ", hpu_yaml_file)

    hpu_yaml_file_buf = open(hpu_yaml_file, "r")
    lines = hpu_yaml_file_buf.readlines()

    is_new_block_start = True
    is_first_line = True
    is_op_found = False
    op_name_found = ""
    line_indx = -1
    lines_to_insert = (
        []
    )  # Array of arrays. 1st element in array of arrays - line #, 2nd elemet in array of arrays - "add" or "delete", 3rd element in array of arrays - line to add
    hpu_ops = hpu_op.split(",")

    for line in lines:
        line_indx += 1

        if not is_first_line and line == "\n":
            if is_op_found:
                is_op_found = False

                arr = []
                arr.append(line_indx)
                arr.append("add")
                arr.append("  guid: " + op_file_name + "\n")
                lines_to_insert.append(arr)

            is_new_block_start = True
            continue

        if is_first_line:
            is_first_line = False

        if is_new_block_start:
            hpu_op_found_in_line = False
            for k in range(len(hpu_ops)):
                if line.split(":")[0] == hpu_ops[k]:
                    hpu_op_found_in_line = True
                    break

            if hpu_op_found_in_line:
                is_new_block_start = False
                is_op_found = True
                continue

        if is_op_found:
            line = line.replace(" ", "")
            feature_line = line.split(":")

            if feature_line[0] == "op_backend" or feature_line[0] == "guid":
                arr = []
                arr.append(line_indx)
                arr.append("delete")
                lines_to_insert.append(arr)

        if line == "\n":
            is_new_block = True
        else:
            is_new_block = False

    processed_indx = []
    for i in range(len(lines_to_insert)):
        # Find the line with max line indx
        max_line = 0
        for j in range(len(lines_to_insert)):
            if max_line < lines_to_insert[j][0] and not bool(any(j in processed_indx for item in processed_indx)):
                max_line = j

        processed_indx.append(max_line)

        # Insert it in lines buffer
        if lines_to_insert[max_line][1] == "delete":
            lines.pop(lines_to_insert[max_line][0])
        else:
            lines.insert(lines_to_insert[max_line][0], lines_to_insert[max_line][2])

    # Write in file
    if is_release:
        with open(hpu_yaml_file, "w") as f:
            lines = "".join(lines)
            f.write(lines)

######## Populate op_gen.cpp file
if make_changes_in_folders.count("pytorch-integration") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == gen_file
):
    print("Writing in file = ", gen_file)

    gen_file_buf = open(gen_file, "r")
    lines = gen_file_buf.readlines()

    fn_found = False
    curly_bracket_stack = 0
    entered_fn = False
    for i in range(len(lines)):
        # Indentation is important in the project else clang_check wouldnt have passed and this file wouldnt have been in the project
        if is_opbackend_present and lines[i].find(features_map["op_backend"] + "::AddNode") != -1:
            fn_found = True

        if fn_found:
            lines[i] = "// " + lines[i]

        if fn_found and lines[i].find("{") != -1:
            curly_bracket_stack += 1
            entered_fn = True
        if fn_found and lines[i].find("}") != -1:
            curly_bracket_stack -= 1

        if fn_found and curly_bracket_stack == 0 and entered_fn:
            break

    # Write in file
    if is_release:
        with open(gen_file, "w") as f:
            lines = "".join(lines)
            f.write(lines)

######## Populate op_greco.pbtxt file
if make_changes_in_folders.count("tpc_kernels") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == greco_pbtxt_file_path
):
    print("Writing in file = ", greco_pbtxt_file_path)

    greco_pbtxt_file_path_buf = open(greco_pbtxt_file_path, "w")

    greco_pbtxt_file_path_buf.write("op {\n")
    greco_pbtxt_file_path_buf.write('  name: "' + op_file_name + '"\n')
    greco_pbtxt_file_path_buf.write("  type {\n")
    greco_pbtxt_file_path_buf.write('    name: "T"\n')

    greco_pbtxt_file_path_buf.write('    definition: "')
    op_dtype = features_map["dtypes"].split("\n")[1]
    op_dtype = op_dtype.split("[")[1]
    op_dtype = op_dtype.split("]")[0]
    op_dtype = op_dtype.split(", ")
    is_first = True
    for i in op_dtype:
        if not is_first:
            greco_pbtxt_file_path_buf.write(", ")
        is_first = False

        if i == "F32":
            greco_pbtxt_file_path_buf.write("tensor(float32)")
        elif i == "BF16":
            greco_pbtxt_file_path_buf.write("tensor(bfloat16)")
        elif i == "F16":
            greco_pbtxt_file_path_buf.write("tensor(float16)")
        elif i == "I32":
            greco_pbtxt_file_path_buf.write("tensor(int32)")
    greco_pbtxt_file_path_buf.write('"\n')

    greco_pbtxt_file_path_buf.write('    description: "<ToDo: Write description"\n')
    greco_pbtxt_file_path_buf.write("  }\n")

    num_inps = 0
    num_optional_inps = 0
    for i in range(len(op_info["optional_inp"])):
        greco_pbtxt_file_path_buf.write("  input {\n")
        greco_pbtxt_file_path_buf.write('    name: "I' + str(i) + '"\n')
        greco_pbtxt_file_path_buf.write('    type: "T"\n')

        if op_info["optional_inp"][i] == 0:
            greco_pbtxt_file_path_buf.write('    description: "The input tensor."\n')
            num_inps += 1
        else:
            greco_pbtxt_file_path_buf.write('    description: "The input optional tensor."\n')
            num_optional_inps += 1

        greco_pbtxt_file_path_buf.write("  }\n")

    greco_pbtxt_file_path_buf.write("  output {\n")
    greco_pbtxt_file_path_buf.write('    name: "O"\n')
    greco_pbtxt_file_path_buf.write('    type: "T"\n')
    greco_pbtxt_file_path_buf.write('    description: "<ToDo: Write description about its shape"\n')
    greco_pbtxt_file_path_buf.write("  }\n")
    greco_pbtxt_file_path_buf.write(
        '  summary: "The output is calculated as, O = <ToDo: Write equation of op using above input variables>"\n'
    )
    greco_pbtxt_file_path_buf.write('  description: "<ToDo: Write description\\n"\n')
    greco_pbtxt_file_path_buf.write(
        '  constraints: "Accepts '
        + str(num_inps)
        + " input tensors, with "
        + str(num_optional_inps)
        + ' optional, and one output tensor.\\n"\n'
    )
    greco_pbtxt_file_path_buf.write('  constraints: "<ToDo: Write shape of input and output tensors>\\n"\n')
    greco_pbtxt_file_path_buf.write('  constraints: "<ToDo: Write which inputs and output shape shoudl match>\\n"\n')

    greco_pbtxt_file_path_buf.write("  guid: [")
    is_first = True
    for i in op_dtype:
        if not is_first:
            greco_pbtxt_file_path_buf.write(", ")
        is_first = False

        if i == "F32":
            greco_pbtxt_file_path_buf.write(op + "_f32")
        elif i == "BF16":
            greco_pbtxt_file_path_buf.write(op + "_bf16")
        elif i == "F16":
            greco_pbtxt_file_path_buf.write(op + "_f16")
        elif i == "I32":
            greco_pbtxt_file_path_buf.write(op + "_i32")
    greco_pbtxt_file_path_buf.write("]\n")

    greco_pbtxt_file_path_buf.write("  supported_devices: [greco]\n")
    greco_pbtxt_file_path_buf.write("}")

######## Populate op.pbtxt file
if make_changes_in_folders.count("tpc_kernels") > 0 and (
    make_changes_in_file == "" or make_changes_in_file == pbtxt_file_path
):
    print("Writing in file = ", pbtxt_file_path)

    pbtxt_file_path_buf = open(pbtxt_file_path, "w")

    pbtxt_file_path_buf.write("op {\n")
    pbtxt_file_path_buf.write('  name: "' + op_file_name + '"\n')
    pbtxt_file_path_buf.write("  type {\n")
    pbtxt_file_path_buf.write('    name: "T"\n')

    pbtxt_file_path_buf.write('    definition: "')
    op_dtype = features_map["dtypes"].split("\n")[1]
    op_dtype = op_dtype.split("[")[1]
    op_dtype = op_dtype.split("]")[0]
    op_dtype = op_dtype.split(", ")
    is_first = True
    for i in op_dtype:
        if not is_first:
            pbtxt_file_path_buf.write(", ")
        is_first = False

        if i == "F32":
            pbtxt_file_path_buf.write("tensor(float32)")
        elif i == "BF16":
            pbtxt_file_path_buf.write("tensor(bfloat16)")
        elif i == "F16":
            pbtxt_file_path_buf.write("tensor(float16)")
        elif i == "I32":
            pbtxt_file_path_buf.write("tensor(int32)")
    pbtxt_file_path_buf.write('"\n')

    pbtxt_file_path_buf.write('    description: "<ToDo: Write description"\n')
    pbtxt_file_path_buf.write("  }\n")

    num_inps = 0
    num_optional_inps = 0
    for i in range(len(op_info["optional_inp"])):
        pbtxt_file_path_buf.write("  input {\n")
        pbtxt_file_path_buf.write('    name: "I' + str(i) + '"\n')
        pbtxt_file_path_buf.write('    type: "T"\n')

        if op_info["optional_inp"][i] == 0:
            pbtxt_file_path_buf.write('    description: "The input tensor."\n')
            num_inps += 1
        else:
            pbtxt_file_path_buf.write('    description: "The input optional tensor."\n')
            num_optional_inps += 1

        pbtxt_file_path_buf.write("    min_rank: " + str(inputs_dim_min[i]) + "\n")
        pbtxt_file_path_buf.write("    max_rank: " + str(inputs_dim_max[i]) + "\n")

        pbtxt_file_path_buf.write("  }\n")

    pbtxt_file_path_buf.write("  output {\n")
    pbtxt_file_path_buf.write('    name: "O"\n')
    pbtxt_file_path_buf.write('    type: "T"\n')
    pbtxt_file_path_buf.write("    min_rank: " + str(outputs_dim_min[0]) + "\n")
    pbtxt_file_path_buf.write("    max_rank: " + str(outputs_dim_max[0]) + "\n")
    pbtxt_file_path_buf.write('    description: "<ToDo: Write description about its shape"\n')
    pbtxt_file_path_buf.write("  }\n")
    pbtxt_file_path_buf.write(
        '  summary: "The output is calculated as, O = <ToDo: Write equation of op using above input variables>"\n'
    )
    pbtxt_file_path_buf.write('  description: "<ToDo: Write description\\n"\n')
    pbtxt_file_path_buf.write(
        '  constraints: "Accepts '
        + str(num_inps)
        + " input tensors, with "
        + str(num_optional_inps)
        + ' optional, and one output tensor.\\n"\n'
    )
    pbtxt_file_path_buf.write('  constraints: "<ToDo: Write shape of input and output tensors>\\n"\n')
    pbtxt_file_path_buf.write('  constraints: "<ToDo: Write which inputs and output shape shoudl match>\\n"\n')

    pbtxt_file_path_buf.write("  guid: [")
    is_first = True
    for i in op_dtype:
        if not is_first:
            pbtxt_file_path_buf.write(", ")
        is_first = False

        if i == "F32":
            pbtxt_file_path_buf.write(op + "_f32")
        elif i == "BF16":
            pbtxt_file_path_buf.write(op + "_bf16")
        elif i == "F16":
            pbtxt_file_path_buf.write(op + "_f16")
        elif i == "I32":
            pbtxt_file_path_buf.write(op + "_i32")
    pbtxt_file_path_buf.write("]\n")

    pbtxt_file_path_buf.write("  supported_devices: [gaudi, gaudi2]\n")
    pbtxt_file_path_buf.write("}")
