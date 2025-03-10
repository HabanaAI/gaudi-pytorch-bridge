###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
###############################################################################


import re

from gen_op_files.version_checker import is_pytorch_older_than

if is_pytorch_older_than("2.7.0"):
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
        "SymInt[2]": "c10::SymIntArrayRef",
        "SymInt[]?": "OptionalSymIntArrayRef",
        "Tensor": "const Tensor &",
        "Tensor?": "const c10::optional<Tensor> &",
        "Tensor?[]": "const c10::List<c10::optional<Tensor>> &",
        "Tensor[]": "TensorList",
        "bool": "bool",
        "bool?": "c10::optional<bool>",
        "float": "double",
        "float?": "c10::optional<double>",
        "float[]": "ArrayRef<double>",
        "float[]?": "c10::optional<ArrayRef<double>>",
        "int": "int64_t",
        "int?": "c10::optional<int64_t>",
        "int[]": "IntArrayRef",
        "int[]?": "OptionalIntArrayRef",
        "int[1]?": "OptionalIntArrayRef",
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
else:
    input_types_map = {
        "Device?": "::std::optional<at::Device>",
        "DeviceIndex": "at::DeviceIndex",
        "Dimname": "at::Dimname",
        "Dimname[]": "at::DimnameList",
        "Dimname[]?": "::std::optional<at::DimnameList>",
        "Generator?": "::std::optional<at::Generator>",
        "Layout": "at::Layout",
        "Layout?": "::std::optional<at::Layout>",
        "MemoryFormat": "at::MemoryFormat",
        "MemoryFormat?": "::std::optional<at::MemoryFormat>",
        "Scalar": "const at::Scalar &",
        "Scalar?": "const ::std::optional<at::Scalar> &",
        "ScalarType": "at::ScalarType",
        "ScalarType?": "::std::optional<at::ScalarType>",
        "Scalar[]": "ArrayRef<at::Scalar>",
        "Storage": "at::Storage",
        "Stream": "at::Stream",
        "SymInt": "c10::SymInt",
        "SymInt?": "::std::optional<c10::SymInt>",
        "SymInt[]": "c10::SymIntArrayRef",
        "SymInt[2]": "c10::SymIntArrayRef",
        "SymInt[]?": "at::OptionalSymIntArrayRef",
        "Tensor": "const at::Tensor &",
        "Tensor?": "const ::std::optional<at::Tensor> &",
        "Tensor?[]": "const c10::List<::std::optional<at::Tensor>> &",
        "Tensor[]": "at::TensorList",
        "bool": "bool",
        "bool?": "::std::optional<bool>",
        "float": "double",
        "float?": "::std::optional<double>",
        "float[]": "at::ArrayRef<double>",
        "float[]?": "::std::optional<at::ArrayRef<double>>",
        "int": "int64_t",
        "int?": "::std::optional<int64_t>",
        "int[]": "at::IntArrayRef",
        "int[]?": "at::OptionalIntArrayRef",
        "int[1]?": "at::OptionalIntArrayRef",
        "str": "c10::string_view",
        "str?": "::std::optional<c10::string_view>",
    }

    output_types_map = {
        "()": "void",
        "QScheme": "at::QScheme",
        "Scalar": "at::Scalar",
        "ScalarType": "at::ScalarType",
        "SymInt": "c10::SymInt",
        "Tensor": "at::Tensor",
        "Tensor[]": "::std::vector<at::Tensor>",
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
