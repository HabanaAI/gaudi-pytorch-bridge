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
import random
import re

import habana_frameworks.torch.hpu as hthpu
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from habana_frameworks.torch.utils.experimental.detect_recompilation import (
    const_shape_dataloader,
    data_dynamicity,
    detect_recompilation_auto_model,
    get_shape,
)
from torch.utils.data import DataLoader, Dataset


class Relu(nn.Module):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        return torch.clip(x, min=0)


class ConvRelu(nn.Module):
    def __init__(self, inch, outch, relu=None):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(inch, outch, 3, 3)
        if relu is None:
            self.relu = Relu()
        else:
            self.relu = relu

    def forward(self, x):
        return self.relu(self.conv(x))


class InnerNet(nn.Module):
    def __init__(self, dyn_ops, reuse_relu, sz):
        super(InnerNet, self).__init__()
        if reuse_relu:
            relu = Relu()
        else:
            relu = None
        self.conv1 = ConvRelu(1, 32, relu)
        self.conv2 = ConvRelu(32, 16, relu)
        self.conv3 = nn.Conv2d(16, 8, 3, 3)
        self.conv4 = nn.Conv2d(8, 8, 1, 1)
        self.dyn_ops = dyn_ops
        self.idx = 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        if self.dyn_ops:
            # dynamic section
            mask = torch.tensor([True] * self.idx + [False] * (x.shape[0] - self.idx)).to(x.device)
            x1 = x[mask, :] * 3
            x2 = x[~mask, :] * 2
            x1 = torch.sum(x1, dim=[-1])
            x2 = torch.sum(x2, dim=[-1])
            self.idx += 1
            return torch.cat([x1, x2])
        else:
            x = torch.sum(x, dim=[-1])
            return x


class Net(nn.Module):
    def __init__(self, dyn_ops, reuse_relu, wrap_inner, sz):
        super(Net, self).__init__()
        innernet = InnerNet(dyn_ops, reuse_relu, sz)
        if wrap_inner:
            self.innernet = detect_recompilation_auto_model(innernet, mdlname="InnerNet", waittime=0.25)
        else:
            self.innernet = innernet

    def forward(self, x):
        return 2 * self.innernet(x)


def train(start_bs, dyn_inp, dyn_ops, reuse_relu=False, wrap_inner=False):

    random.seed(0)
    np.random.seed(0)
    device = "hpu"
    if device == "hpu":
        import habana_frameworks.torch.core as htcore

    net = Net(dyn_ops, reuse_relu, wrap_inner, start_bs)
    net = net.to(device)
    if not wrap_inner:
        net = detect_recompilation_auto_model(net, mdlname="Net", waittime=0.2)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    loss_func = torch.nn.MSELoss()
    # Batch sizes is changed to make the model dynamic and force recompilation
    if dyn_inp:
        bs_list = [start_bs] * 3 + [start_bs + 10] * 2
    else:
        bs_list = [start_bs] * 5

    for bs in bs_list:
        inp_size = 50
        inp_np = np.random.random((bs, 1, inp_size, inp_size)).astype(np.float32)
        inputs = torch.tensor(inp_np).to(device)
        outputs = torch.tensor(np.trace(inp_np, axis1=2, axis2=3)).squeeze().to(device)
        prediction = net(inputs)
        loss = loss_func(prediction, outputs)
        optimizer.zero_grad()
        loss.backward()
        if device == "hpu":
            htcore.mark_step()
        optimizer.step()
        if device == "hpu":
            htcore.mark_step()
        print(loss)
    print("Train loop done")

    if wrap_inner:
        return "\n".join(net.innernet.raw_logs()), net.innernet
    else:
        return "\n".join(net.raw_logs()), net


def strip_colors(ln):
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", ln)


def read_file_and_stripcols(flname):
    return [strip_colors(ln) for ln in open(flname).readlines()]


def change_module_name(modules):
    return ["/".join(filter(lambda x: len(x) > 0, ["InnerNet"] + k.split("innernet/")[1:])) for k in modules]


def match_fl1(fl1, dyn_inps, dyn_ops, reuse_relu, wrap_inner):
    curr_file = __file__.split("/")[-1]
    headers = "Step,Recompiling modules,New in,New out,Class,Location,Comment\n"
    modules = [
        "Net/innernet/conv1/conv",
        "Net/innernet/conv1/relu",
        "Net/innernet/conv1",
        "Net/innernet/conv2/conv",
        "Net/innernet/conv2/relu",
        "Net/innernet/conv2",
        "Net/innernet/conv3",
        "Net/innernet/conv4",
        "Net/innernet",
        "Net",
    ]
    classes = [
        "torch.nn.modules.conv.Conv2d",
        "Relu",
        "ConvRelu",
        "torch.nn.modules.conv.Conv2d",
        "Relu",
        "ConvRelu",
        "torch.nn.modules.conv.Conv2d",
        "torch.nn.modules.conv.Conv2d",
        "InnerNet",
        "Net",
    ]
    locs = [
        "torch/nn/modules/conv.py",
        curr_file,
        curr_file,
        "torch/nn/modules/conv.py",
        curr_file,
        curr_file,
        "torch/nn/modules/conv.py",
        "torch/nn/modules/conv.py",
        curr_file,
        curr_file,
    ]
    comment1 = "Recompiled due to new input shape"
    comment2 = "Already processed input shape still recompiled. Maybe dyn ops"
    comment3 = comment2 + ". Could be due to dynamic child"
    fl1lines = read_file_and_stripcols(fl1)
    if reuse_relu:
        modules[4] = "Net/innernet/conv1/relu"
    if wrap_inner:
        modules.remove("Net")
        modules = change_module_name(modules)
    passed = fl1lines[0] == headers

    def helper(lines, step_fn, module_nm_fn, inoutbool, class_fn, loc_fn, comment_fn):
        passed = True
        for idx, ln in enumerate(lines):
            step, modulenm, inp, out, cls, loc, comment = ln.strip().split(",")
            passed = passed and (int(step) == step_fn(idx))
            passed = passed and (modulenm == module_nm_fn(idx))
            passed = passed and (inp == inoutbool)
            passed = passed and (out == inoutbool)
            passed = passed and (class_fn(idx) in cls)
            passed = passed and (loc_fn(idx) in loc)
            passed = passed and (comment == comment_fn(cls))
        return passed

    def helper1(lines, expected_step):
        return helper(
            lines,
            lambda _: expected_step,
            lambda idx: modules[idx],
            "True",
            lambda idx: classes[idx],
            lambda idx: locs[idx],
            lambda _: comment1,
        )

    def module_nm_fn(idx):
        return "InnerNet" if wrap_inner else ("Net", "Net/innernet")[idx % 2 == 0]

    def class_fn(idx):
        return "InnerNet" if wrap_inner else ("Net", "InnerNet")[idx % 2 == 0]

    def helper2(lines, step_fn):
        return helper(
            lines,
            step_fn,
            module_nm_fn,
            "False",
            class_fn,
            lambda _: curr_file,
            lambda nm: (comment3, comment2)["InnerNet" in nm],
        )

    # in all cases, step0 will show all modules recompile
    passed = passed and helper1(fl1lines[1 : 1 + len(modules)], 0)
    if dyn_ops:
        if dyn_inps:
            passed = passed and (len(fl1lines) == 1 + len(modules) * 2 + 3 * (2, 1)[wrap_inner])
            st0 = 1 + len(modules)
            end0 = st0 + 2 * (2, 1)[wrap_inner]
            st1 = 1 + 2 * len(modules) + 2 * (2, 1)[wrap_inner]
            end1 = st1 + 1

            def fn(idx):
                return (
                    {0: 1, 1: 2}.get(idx, 4)
                    if wrap_inner
                    else {
                        0: 1,
                        1: 1,
                        2: 2,
                        3: 2,
                    }.get(idx, 4)
                )

            passed = passed and helper2(fl1lines[st0:end0] + fl1lines[st1:end1], fn)
            st0 = 1 + len(modules) + 2 * (2, 1)[wrap_inner]
            end0 = st0 + len(modules)
            passed = passed and helper1(fl1lines[st0:end0], 3)
        else:
            passed = passed and (len(fl1lines) == 1 + len(modules) + 4 * (2, 1)[wrap_inner])
            # step 1,2,3,4 will see only 2 modules (with dyn ops) recompile

            def fn(idx):
                return (idx + 1) if wrap_inner else (idx // 2 + 1)

            passed = passed and helper2(fl1lines[1 + len(modules) :], fn)
    else:
        if dyn_inps:
            passed = passed and len(fl1lines) == 1 + len(modules) * 2
            # step 3 will see all modules recompile because of dyn inps
            passed = passed and helper1(fl1lines[1 + len(modules) :], 3)
        else:
            pass
            # recompiles only on step 0, which have already been checked before
    return passed


def gen_expected2(dyn_inps, dyn_ops, reuse_relu, wrap_inner):
    lst = [
        "Net/innernet",
        "Net",
        "Net/innernet/conv1/conv",
        "Net/innernet/conv1/relu",
        "Net/innernet/conv1",
        "Net/innernet/conv2/conv",
        "Net/innernet/conv2/relu",
        "Net/innernet/conv2",
        "Net/innernet/conv3",
        "Net/innernet/conv4",
    ]
    if reuse_relu:
        lst.remove("Net/innernet/conv2/relu")
    if wrap_inner:
        lst.remove("Net")
        lst = change_module_name(lst)

    def mapper(x):
        if dyn_inps:
            if dyn_ops:
                return (5 if x == "InnerNet" else 2) if wrap_inner else (5 if x == "Net" or x == "Net/innernet" else 2)
            else:
                return 2
        else:
            if dyn_ops:
                return (5 if x == "InnerNet" else 1) if wrap_inner else (5 if x == "Net" or x == "Net/innernet" else 1)
            else:
                return 1

    num_recompiles = [mapper(k) for k in lst]
    if reuse_relu:
        num_recompiles[(3, 2)[wrap_inner]] = (2, 4)[dyn_inps]
    return lst, num_recompiles


def match_fl2(fl2, dyn_inps, dyn_ops, reuse_relu, wrap_inner):
    lst, num_recompiles = gen_expected2(dyn_inps, dyn_ops, reuse_relu, wrap_inner)
    fl2lines = read_file_and_stripcols(fl2)
    if fl2lines[0] != "Module name,Recompile count\n":
        return False
    assert len(lst) == len(num_recompiles)
    if len(lst) != len(fl2lines) - 1:
        return False
    d1 = {}
    for ln in fl2lines[1:]:
        lhs1, rhs1 = ln.split(",")
        assert lhs1 not in d1
        d1[lhs1] = int(rhs1)
    d2 = {k: v for k, v in zip(lst, num_recompiles)}
    return d1 == d2


def generator():
    # generate diff batch sizes,
    # so that each test forces a recompile atleast at the first step
    # and GC does not reuse already compiled recipe from previous test
    num = 2
    while True:
        yield num * 10
        num += 2


get_start_bs = generator()


def del_file(fl):
    if os.path.isfile(fl):
        os.remove(fl)


@pytest.mark.skip(reason="KeyError: 'InnerNet'")
@pytest.mark.parametrize("dyn_inps", [True, False])
@pytest.mark.parametrize("dyn_ops", [True, False])
@pytest.mark.parametrize("reuse_relu", [True, False])
@pytest.mark.parametrize("wrap_inner", [True, False])
def test_basic(dyn_inps, dyn_ops, reuse_relu, wrap_inner):
    hthpu.disable_dynamic_shape()
    start_bs = next(get_start_bs)
    print(dyn_inps, dyn_ops, reuse_relu, wrap_inner, start_bs)
    tag = "out"
    logs, mdl = train(start_bs, dyn_inps, dyn_ops, reuse_relu, wrap_inner)
    # fl0 = tag + '_raw.txt'
    fl1 = tag + "_1.csv"
    fl2 = tag + "_2.csv"
    # with open(fl0, 'w') as f:
    #    f.write(logs)
    # del_file(fl0)
    del_file(fl2)
    del_file(fl2)
    mdl.analyse_dynamicity()
    test_files_dumped = os.path.isfile(fl1) and os.path.isfile(fl2)
    assert test_files_dumped
    print("correct files dumped")
    matched_fl1 = match_fl1(fl1, dyn_inps, dyn_ops, reuse_relu, wrap_inner)
    matched_fl2 = match_fl2(fl2, dyn_inps, dyn_ops, reuse_relu, wrap_inner)
    test_res = matched_fl1 and matched_fl2
    assert test_res
    print("test passed")


# TODO: add a couple of model tests?


class SampleDataset(Dataset):
    """A toy dataset, that generates random inputs of different shapes"""

    def __init__(self, shapes, num, seed=42):
        self.shapes = shapes
        self.num = num
        torch.manual_seed(seed)
        random.seed(seed)

    def __len__(self):
        return self.num

    def _gen(self):
        return torch.rand(*random.sample(self.shapes, 1)[0])

    def __getitem__(self, idx):
        return self._gen()


def collate_batch(batch):
    max_shapes_per_dim = np.max(np.array([tuple(k.shape) for k in batch]), 0)
    rank = len(max_shapes_per_dim)
    final = []
    for item in batch:
        pad_params = []
        for k in range(rank):
            pad_params += [[0, max_shapes_per_dim[k] - item.shape[k]]]
        pad_params = pad_params[::-1]
        pad_params = sum(pad_params, [])
        final += [F.pad(item, pad_params, "constant", 0)]
    return torch.stack(final)


class SampleDatasetComplex(SampleDataset):
    """A toy dataset, that generates random inputs of different, complex shapes"""

    def __getitem__(self, idx):
        return (
            self._gen(),
            [self._gen(), self._gen(), {1: self._gen(), 2: (self._gen(),)}],
        )


@pytest.mark.skip
def test_dataloader_basic_fns():
    assert (2,) == get_shape(torch.tensor([1, 2]))
    assert ((2,), (3,)) == get_shape([torch.tensor([1, 2]), torch.tensor([1, 2, 3])])
    assert ((2,), ((1, 3),)) == get_shape([torch.tensor([1, 2]), {1: torch.tensor([1, 2, 3])}])


def test_dataloader_simple():
    dataset = SampleDataset([[3, 10, 10], [3, 20, 20], [3, 30, 30]], 1000)
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_batch, shuffle=True)
    report = data_dynamicity(dataloader)
    expected = {(4, 3, 30, 30): 194, (4, 3, 20, 20): 50, (4, 3, 10, 10): 6}
    assert report == expected

    const_shape_dl = const_shape_dataloader(dataloader, 2)
    assert len(const_shape_dl) == 2
    shape_sig = None
    for k in const_shape_dl:
        if shape_sig is None:
            shape_sig = get_shape(k)
            assert shape_sig in expected
        else:
            assert shape_sig == get_shape(k)
            print(get_shape(k))


def test_dataloader_complex():
    dataset2 = SampleDatasetComplex([[3, 10, 10], [3, 20, 20], [3, 30, 30]], 100)
    dataloader2 = DataLoader(dataset2, batch_size=1, collate_fn=None, shuffle=True)
    report2 = data_dynamicity(dataloader2)
    assert report2 == {
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 10, 10),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 10, 10),
                (1, 3, 10, 10),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 3,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 10, 10),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 2,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 10, 10),
                (1, 3, 20, 20),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 20, 20),))),
            ),
        ): 2,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 10, 10),
                (1, 3, 10, 10),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 3,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 20, 20),
                (1, 3, 10, 10),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 20, 20),
                (1, 3, 10, 10),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 20, 20),))),
            ),
        ): 2,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 20, 20),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 2,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 30, 30),))),
            ),
        ): 3,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 30, 30),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 20, 20),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 20, 20),
                (1, 3, 20, 20),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 20, 20),
                (1, 3, 20, 20),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 30, 30),
                (1, 3, 10, 10),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 30, 30),))),
            ),
        ): 2,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 10, 10),
                (1, 3, 20, 20),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 10, 10),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 20, 20),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 2,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 30, 30),
                (1, 3, 20, 20),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 20, 20),
                (1, 3, 20, 20),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 20, 20),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 20, 20),
                (1, 3, 10, 10),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 20, 20),
                (1, 3, 10, 10),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 30, 30),
                (1, 3, 10, 10),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 20, 20),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 10, 10),
                (1, 3, 20, 20),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 3,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 10, 10),
                (1, 3, 20, 20),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 2,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 20, 20),
                (1, 3, 10, 10),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 30, 30),
                (1, 3, 20, 20),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 20, 20),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 30, 30),
                (1, 3, 20, 20),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 2,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 20, 20),
                (1, 3, 30, 30),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 10, 10),
                (1, 3, 20, 20),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 2,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 20, 20),
                (1, 3, 20, 20),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 20, 20),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 30, 30),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 20, 20),
                (1, 3, 20, 20),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 20, 20),
                (1, 3, 20, 20),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 20, 20),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 30, 30),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 2,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 30, 30),
                (1, 3, 10, 10),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 20, 20),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 30, 30),
                (1, 3, 20, 20),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 30, 30),
                (1, 3, 20, 20),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 20, 20),))),
            ),
        ): 2,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 20, 20),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 30, 30),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 30, 30),
                (1, 3, 20, 20),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 30, 30),
                (1, 3, 10, 10),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 20, 20),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 20, 20),
                (1, 3, 20, 20),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 2,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 20, 20),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 30, 30),
                (1, 3, 10, 10),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 20, 20),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 20, 20),
            (
                (1, 3, 20, 20),
                (1, 3, 10, 10),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 10, 10),
                (1, 3, 30, 30),
                ((1, (1, 3, 10, 10)), (2, ((1, 3, 20, 20),))),
            ),
        ): 1,
        (
            (1, 3, 30, 30),
            (
                (1, 3, 20, 20),
                (1, 3, 30, 30),
                ((1, (1, 3, 30, 30)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 10, 10),
                (1, 3, 10, 10),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 10, 10),))),
            ),
        ): 1,
        (
            (1, 3, 10, 10),
            (
                (1, 3, 30, 30),
                (1, 3, 20, 20),
                ((1, (1, 3, 20, 20)), (2, ((1, 3, 30, 30),))),
            ),
        ): 1,
    }
