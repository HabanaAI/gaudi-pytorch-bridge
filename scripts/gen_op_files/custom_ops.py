###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################

import re

input_types_map = {
    "Device?": "c10::optional<Device>",
    "DeviceIndex": "DeviceIndex",
    "Dimname": "Dimname",
    "Dimname[]": "DimnameList",
    "Dimname[]?": "c10::optional<DimnameList>",
    "Generator?": "c10::optional<Generator>",
    "Layout": "Layout",
    "Layout?": "c10::optional<Layout>",
    "MemoryFormat": "MemoryFormat",
    "MemoryFormat?": "c10::optional<MemoryFormat>",
    "Scalar": "const Scalar &",
    "Scalar?": "const c10::optional<Scalar> &",
    "ScalarType": "ScalarType",
    "ScalarType?": "c10::optional<ScalarType>",
    "Scalar[]": "ArrayRef<Scalar>",
    "Storage": "Storage",
    "Stream": "Stream",
    "SymInt": "c10::SymInt",
    "SymInt?": "c10::optional<c10::SymInt>",
    "SymInt[]": "c10::SymIntArrayRef",
    "SymInt[]?": "OptionalSymIntArrayRef",
    "Tensor": "const Tensor &",
    "Tensor?": "const c10::optional<Tensor> &",
    "Tensor?[]": "const c10::List<c10::optional<Tensor>> &",
    "Tensor[]": "TensorList",
    "bool": "bool",
    "bool?": "c10::optional<bool>",
    "float": "double",
    "float?": "c10::optional<double>",
    "float[]?": "c10::optional<ArrayRef<double>>",
    "int": "int64_t",
    "int?": "c10::optional<int64_t>",
    "int[]": "IntArrayRef",
    "int[]?": "OptionalIntArrayRef",
    "str": "c10::string_view",
    "str?": "c10::optional<c10::string_view>",
}

output_types_map = {
    "()": "void",
    "QScheme": "QScheme",
    "Scalar": "Scalar",
    "ScalarType": "ScalarType",
    "SymInt": "c10::SymInt",
    "Tensor": "Tensor",
    "Tensor[]": "::std::vector<Tensor>",
    "bool": "bool",
    "float": "double",
    "int": "int64_t",
}


def input_type(dtype):
    if dtype in input_types_map:
        return input_types_map[dtype]
    if re.match(r"Tensor\(.*\)", dtype):
        return "TensorList" if dtype[-1] == "]" else "Tensor &"
    if re.match(r"bool\[(\d+)\]", dtype):
        ctype = re.match(r"bool\[(\d+)\]", dtype).groups()[0]
        return f"::std::array<bool,{ctype}>"
    assert True, f"Custom schema input dtype '{dtype}' is not yet implemented in gen_op.py. Feel free to add it."


def output_type(dtype):
    if dtype in output_types_map:
        return output_types_map[dtype]
    if re.match(r"Tensor\(.*\)", dtype):
        return "Tensor &"
    assert True, f"Custom schema output dtype '{dtype}' is not yet implemented in gen_op.py. Feel free to add it."


def cpp_from_schema(schema):
    ptrn = r'([^(]*)\((.*)\) -> ([^"]*)'
    m = re.match(ptrn, schema)
    assert m is not None, f"Custom schema {schema} didn't match pattern"

    op_name = m.groups()[0].split(".")[0].split("::")[-1]
    inputs = m.groups()[1].replace(", *", "").split(", ")
    outputs = m.groups()[2]
    if outputs[0] == "(":
        outputs = outputs[1:-1]
    outputs = outputs.split(", ")

    inputs_cpp = []
    for input in inputs:
        dtype, name = tuple(input.split(" "))
        name = name.split("=")[0]
        inputs_cpp.append(f"{input_type(dtype)} {name}")

    inputs_cpp = ", ".join(inputs_cpp)

    outputs_cpp = []
    for output in outputs:
        outputs_cpp.append(output_type(output))

    outputs_cpp = outputs_cpp[0] if len(outputs_cpp) == 1 else f"::std::tuple<{','.join(outputs_cpp)}>"

    return f"{outputs_cpp} {op_name}({inputs_cpp})"
