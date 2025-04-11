#!/usr/bin/env python3
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


import json
import os
import pathlib
import shutil
from dataclasses import dataclass
from filecmp import dircmp

import gen_op.parser as parser
import pytest
import torch
from gen_op.code_generation import (
    check_valid_fields,
    cpp_from_schema,
    generate,
    generate_check_kernel_support,
    generate_op_backend_hclasses,
    generate_op_frontend_hclasses,
    get_op_group,
    is_acc_thread_supported,
    parse_params,
)
from gen_op.op import Op
from gen_op.version_checker import is_pytorch_exactly, is_pytorch_older_than

TORCH_PKG_PATH = torch.__path__[0]

profiles_path = os.path.join(os.getenv("PYTORCH_MODULES_ROOT_PATH"), ".devops/build_profiles/profiles.json")

with open(profiles_path, encoding="utf-8") as profiles_json:
    current_pytorch_version = json.load(profiles_json)["pt_versions"]["current"]["version"]

pytestmark = [
    pytest.mark.skipif(
        is_pytorch_older_than(current_pytorch_version), reason="Only newest PyTorch version should be validated"
    ),
    pytest.mark.xfail(
        not is_pytorch_exactly(current_pytorch_version),
        reason="It's time consuming for developers to set up pytorch-next just to update reference for future tests",
    ),
]


@dataclass
class Args:
    output_dir: str
    yaml: str
    check_kernel_support: str
    pt_signatures: str
    native_functions: str = os.path.join(TORCH_PKG_PATH, "../torchgen/packaged/ATen/native/native_functions.yaml")


def check_diffs_recursively(cmp, only_left, only_right, different):
    only_left += [f"{cmp.left}/{x}" for x in cmp.left_only]
    only_right += [f"{cmp.right}/{x}" for x in cmp.right_only]
    different += [f"{cmp.left}/{x}" for x in cmp.diff_files]

    for sub_cmp in cmp.subdirs.values():
        check_diffs_recursively(sub_cmp, only_left, only_right, different)


def test_ops_generation_e2e(monkeypatch):
    # without that generated files differ between CI and local test
    def mock_gen_op_file(args, **kwargs):
        return "run_gen_op.py"

    monkeypatch.setattr(os.path, "basename", mock_gen_op_file)

    ref_output_dir = ".".join(torch.__version__.split(".")[:2])
    test_path = pathlib.Path(__file__).parent.resolve()
    output_dir = os.path.join(test_path, "output")
    reference_dir = os.path.join(test_path, "files/ops_generation_e2e/reference_output", ref_output_dir)
    yaml_path = os.path.join(test_path, "files/ops_generation_e2e/hpu_op.yaml")
    pt_signatures = os.path.join(
        test_path,
        "files/ops_generation_e2e",
        (
            "FakeRegistrationDeclarations.h"
            if is_pytorch_older_than("2.7.0")
            else "FakeRegistrationDeclarationsFuture.h"
        ),
    )

    shutil.rmtree(output_dir, ignore_errors=True)

    args = Args(output_dir, yaml_path, False, pt_signatures)

    generate(args)
    args.check_kernel_support = True
    generate_check_kernel_support(args)

    cmp = dircmp(output_dir, reference_dir)
    unexpected_files = []
    missing_files = []
    different_files = []

    check_diffs_recursively(cmp, unexpected_files, missing_files, different_files)

    assert len(unexpected_files) == 0, f"These files and directories shouldn't be generated: {unexpected_files}"
    assert len(missing_files) == 0, f"These files and directories were not generated: {missing_files}"
    assert len(different_files) == 0, f"These files differ from reference: {different_files}"

    shutil.rmtree(output_dir)


if is_pytorch_older_than("2.7.0"):
    SCHEMA_CPP_LIST = [
        (
            "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
            "::std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const std::optional<Tensor> & weight, const std::optional<Tensor> & bias, const std::optional<Tensor> & running_mean, const std::optional<Tensor> & running_var, bool training, double momentum, double eps)",
        ),
        (
            "aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor",
            "Tensor index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, const Scalar & alpha)",
        ),
        (
            "hpu::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor",
            "Tensor cross_entropy_loss(const Tensor & self, const Tensor & target, const std::optional<Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, double label_smoothing)",
        ),
        (
            "aten::normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
            "Tensor normal(double mean, double std, c10::SymIntArrayRef size, std::optional<Generator> generator, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory)",
        ),
    ]
else:
    SCHEMA_CPP_LIST = [
        (
            "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
            "::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(const at::Tensor & input, const ::std::optional<at::Tensor> & weight, const ::std::optional<at::Tensor> & bias, const ::std::optional<at::Tensor> & running_mean, const ::std::optional<at::Tensor> & running_var, bool training, double momentum, double eps)",
        ),
        (
            "aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor",
            "at::Tensor index_add(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha)",
        ),
        (
            "hpu::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor",
            "at::Tensor cross_entropy_loss(const at::Tensor & self, const at::Tensor & target, const ::std::optional<at::Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, double label_smoothing)",
        ),
        (
            "aten::normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
            "at::Tensor normal(double mean, double std, c10::SymIntArrayRef size, ::std::optional<at::Generator> generator, ::std::optional<at::ScalarType> dtype, ::std::optional<at::Layout> layout, ::std::optional<at::Device> device, ::std::optional<bool> pin_memory)",
        ),
    ]


@pytest.mark.parametrize("schema, cpp", SCHEMA_CPP_LIST)
def test_cpp_from_schema(schema, cpp):
    result = cpp_from_schema(schema)
    assert result == cpp, f"Generated cpp signature differs from reference: {result} != {cpp}"


@pytest.mark.parametrize(
    "op, op_group",
    [
        ("abcd", "abcd"),
        ("test_op.variant", "test_op"),
        ("__dunder_op__.var", "__dunder_op__"),
        ("inplace_op_", "inplace_op"),
        ("_another_inplace_.Tensor", "_another_inplace"),
    ],
)
def test_get_op_group(op, op_group):
    result = get_op_group(op)
    assert result == op_group


@pytest.mark.parametrize(
    "frontend_class, override_fn, is_eager",
    [
        ("GeneratorToSeed", None, True),
        ("some_op", "some_override_lazy", False),
        ("LazyOp", None, True),
        ("LazyOp", "set_source_Tensor", True),
        ("LazyOp", "some_op_lazy", False),
    ],
)
def test_is_eager_op(frontend_class, override_fn, is_eager):
    ctxop = Op("test_op", {"op_frontend_class": frontend_class, "override_fn": override_fn})
    result = ctxop.is_eager_op()
    assert result == is_eager


@pytest.mark.parametrize(
    "override_fn, acc_thread, rtype, sig, is_acc",
    [
        (True, False, "", "", False),
        ("some_override", True, "", "", True),
        (None, True, "some_rtype", "some_sig", False),
        (None, False, "at::Tensor something", "some_sig", True),
        (None, True, "some_rtype", "at::TensorList something", True),
    ],
)
def test_is_acc_thread_supported(override_fn, acc_thread, rtype, sig, is_acc):
    ctxop = Op("test_op", {"override_fn": override_fn, "acc_thread": acc_thread})
    result = is_acc_thread_supported(ctxop, rtype, sig)
    assert result == is_acc


class FgenStub:
    def __init__(self, ctxop: Op):
        self.ctxop = ctxop


@pytest.mark.parametrize("is_backend", [True, False])
def test_generate_op_hclasses(is_backend):
    classes = {}
    header_file = "header.h"
    base_class = "ns::BaseClass"

    def generate_frontend_hclasses(*args):
        return generate_op_frontend_hclasses(*args, base_class)

    if is_backend:
        default_class = "OpBackend"
        generate_func = generate_op_backend_hclasses
        macro_suffix = "BACKEND("
        getter = "op_backend"
    else:
        default_class = "LazyOp"
        generate_func = generate_frontend_hclasses
        macro_suffix = f"FRONTEND({base_class}, "
        getter = "op_frontend"

    tested_classes = [default_class, "SomeTemplate", "SomeOp", "CustomClass", "SomeTemplateCustom", "SomeOp"]
    fgens = [FgenStub(Op("test_op", {getter: x})) for x in tested_classes]

    result = generate_func(fgens, classes, header_file)
    assert (
        result
        == f"HPU_OP_{macro_suffix}SomeOp)\nHPU_OP_{macro_suffix}CustomClass)\nHPU_OP_{macro_suffix}SomeTemplateCustom)\n"
    )
    assert classes == {"SomeOp": header_file, "CustomClass": header_file, "SomeTemplateCustom": header_file}


@pytest.mark.parametrize(
    "cpp_sig, out_indices, expected_results",
    [
        (
            "void _foreach_addcmul_(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Tensor & scalars)",
            [0],
            {
                "param_vars": ["self", "tensor1", "tensor2", "scalars"],
                "call_args": ["self"],
                "out_indices": [0],
                "fc_params": [],
            },
        ),
        (
            "::std::vector<at::Tensor> _foreach_addcmul(at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars)",
            None,
            {
                "param_vars": ["self", "tensor1", "tensor2", "scalars"],
                "call_args": [],
                "out_indices": [],
                "fc_params": [],
            },
        ),
    ],
)
def test_parse_params(cpp_sig, out_indices, expected_results):
    if is_pytorch_older_than("2.7.0"):
        cpp_sig = cpp_sig.replace("at::", "")
    tree = parser.parse(cpp_sig)
    rwxtree = parser.xparse(cpp_sig)
    params = parser.get_parameters(tree)
    rtype = parser.get_return_type_str(rwxtree, cpp_sig)
    funsig = parser.create_stdfunc_sig(rwxtree, cpp_sig)

    _, fname, _ = parser.get_function_signature(rwxtree, cpp_sig, lambda x: f"{x}")

    param_vars, call_args, out_indices, fc_params, _ = parse_params(params, fname, rtype, [], funsig, out_indices)

    assert param_vars == expected_results["param_vars"]
    assert call_args == expected_results["call_args"]
    assert out_indices == expected_results["out_indices"]
    assert fc_params == expected_results["fc_params"]


def test_check_valid_fields():
    op_name = "wrong_op"
    op_params = {"guid": "nop", "dtype": ["float"]}
    check_valid_fields_results = check_valid_fields(op_name, op_params)

    assert len(check_valid_fields_results) == 1
    assert check_valid_fields_results[0] == "Invalid field for wrong_op: dtype\n"
