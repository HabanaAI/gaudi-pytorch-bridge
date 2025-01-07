#!/usr/bin/env python3
###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
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


import os
import pathlib
import shutil
from dataclasses import dataclass
from filecmp import dircmp

import gen_op_files.parser as parser
import pytest
import torch
from gen_op import (
    _TYPE_NSMAP,
    check_op_params,
    cpp_from_schema,
    generate,
    generate_check_kernel_support,
    generate_op_backend_hclasses,
    generate_op_frontend_hclasses,
    get_op_group,
    is_acc_thread_supported,
    is_eager_op,
    parse_params,
)
from packaging.version import Version

TORCH_PKG_PATH = torch.__path__[0]

pytestmark = pytest.mark.skipif(
    Version(Version(torch.__version__).base_version) < Version("2.5.0"),
    reason="Only newest PyTorch version should be validated",
)


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
        return "gen_op.py"

    monkeypatch.setattr(os.path, "basename", mock_gen_op_file)

    ref_output_dir = ".".join(torch.__version__.split(".")[:2])
    test_path = pathlib.Path(__file__).parent.resolve()
    output_dir = os.path.join(test_path, "output")
    reference_dir = os.path.join(test_path, "files/ops_generation_e2e/reference_output", ref_output_dir)
    yaml_path = os.path.join(test_path, "files/ops_generation_e2e/hpu_op.yaml")
    pt_signatures = os.path.join(test_path, "files/ops_generation_e2e/RegistrationDeclarations.h")

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


SCHEMA_CPP_LIST = [
    (
        "aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)",
        "::std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const c10::optional<Tensor> & weight, const c10::optional<Tensor> & bias, const c10::optional<Tensor> & running_mean, const c10::optional<Tensor> & running_var, bool training, double momentum, double eps)",
    ),
    (
        "aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor",
        "Tensor index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source, const Scalar & alpha)",
    ),
    (
        "hpu::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, SymInt ignore_index=-100, float label_smoothing=0.0) -> Tensor",
        "Tensor cross_entropy_loss(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, c10::SymInt ignore_index, double label_smoothing)",
    ),
    (
        "aten::normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor",
        "Tensor normal(double mean, double std, c10::SymIntArrayRef size, c10::optional<Generator> generator, c10::optional<ScalarType> dtype, c10::optional<Layout> layout, c10::optional<Device> device, c10::optional<bool> pin_memory)",
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


class CtxOpStub:
    def __init__(self, props=dict()):
        self.properties = props

    def set_property(self, key, value):
        self.properties[key] = value

    def get_op_frontend_class(self):
        return self.properties.get("op_frontend_class", None)

    def get_op_backend_class(self):
        return self.properties.get("op_backend_class", None)

    def get_override_fn(self):
        return self.properties.get("override_fn", None)

    def get_acc_thread(self):
        return self.properties.get("acc_thread", None)


@pytest.mark.parametrize(
    "frontend_class, override_fn, is_eager",
    [
        ("GeneratorToSeed", None, True),
        ("some_op", "some_override", False),
        ("LazyOp", None, True),
        ("LazyOp", "set_source_Tensor", True),
        ("LazyOp", "some_op", False),
    ],
)
def test_is_eager_op(frontend_class, override_fn, is_eager):
    ctxop = CtxOpStub({"op_frontend_class": frontend_class, "override_fn": override_fn})
    result = is_eager_op(ctxop)
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
    ctxop = CtxOpStub({"override_fn": override_fn, "acc_thread": acc_thread})
    result = is_acc_thread_supported(ctxop, rtype, sig)
    assert result == is_acc


class FgenStub:
    def __init__(self, ctxop=CtxOpStub()):
        self.ctxop = ctxop


@pytest.mark.parametrize("is_backend", [True, False])
def test_generate_op_hclasses(is_backend):
    classes = dict()
    header_file = "header.h"
    base_class = "ns::BaseClass"

    if is_backend:
        default_class = "OpBackend"
        generate_func = generate_op_backend_hclasses
        macro_suffix = "BACKEND("
        getter = "op_backend_class"
    else:
        default_class = "LazyOp"
        generate_func = lambda *args: generate_op_frontend_hclasses(*args, base_class)
        macro_suffix = f"FRONTEND({base_class}, "
        getter = "op_frontend_class"

    tested_classes = [default_class, "SomeTemplate", "SomeOp", "CustomClass", "SomeTemplateCustom", "SomeOp"]
    fgens = [FgenStub(CtxOpStub({getter: x})) for x in tested_classes]

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
            "void _foreach_addcmul_(TensorList self, TensorList tensor1, TensorList tensor2, const Tensor & scalars)",
            [0],
            {
                "param_vars": ["self", "tensor1", "tensor2", "scalars"],
                "call_args": ["self"],
                "out_indices": [0],
                "fc_params": [],
            },
        ),
        (
            "::std::vector<Tensor> _foreach_addcmul(TensorList self, TensorList tensor1, TensorList tensor2, ArrayRef<Scalar> scalars)",
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
    tree = parser.parse(cpp_sig)
    rwsig = parser.rewrite_signature(cpp_sig, _TYPE_NSMAP)
    rwxtree = parser.xparse(rwsig)
    params = parser.get_parameters(tree)
    rtype = parser.get_return_type_str(rwxtree, rwsig)
    funsig = parser.create_stdfunc_sig(rwxtree, rwsig)

    _, fname, _ = parser.get_function_signature(rwxtree, rwsig, lambda x: "{}".format(x))

    param_vars, call_args, out_indices, fc_params, _ = parse_params(params, fname, rtype, [], funsig, out_indices)

    assert param_vars == expected_results["param_vars"]
    assert call_args == expected_results["call_args"]
    assert out_indices == expected_results["out_indices"]
    assert fc_params == expected_results["fc_params"]


def test_check_op_params_exception():
    op_name = "wrong_op"
    op_params = {"guid": "nop", "dtype": ["float"]}
    with pytest.raises(Exception, match="wrong_op.*dtype"):
        check_op_params(op_name, op_params)
