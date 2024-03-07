###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import habana_frameworks.torch.core as htcore
import pytest
import torch


def test_simple_sgd_convnet():
    class LeNet5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                torch.nn.BatchNorm2d(6),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fc = torch.nn.Linear(400, 120)
            self.relu = torch.nn.ReLU()
            self.fc1 = torch.nn.Linear(120, 84)
            self.relu1 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(84, 10)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            out = self.relu(out)
            out = self.fc1(out)
            out = self.relu1(out)
            out = self.fc2(out)
            return out

    model = LeNet5().to("hpu")
    model = torch.compile(model, backend="hpu_backend")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    input_tensor1 = torch.rand(8, 1, 32, 32).to("hpu")
    input_tensor2 = torch.randint(0, 9, (8,)).to("hpu")

    def iteration(x, y):
        result = model(x)
        loss = criterion(result, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    loss_compile1 = iteration(input_tensor1, input_tensor2)
    loss_compile2 = iteration(input_tensor1, input_tensor2)
    loss_compile3 = iteration(input_tensor1, input_tensor2)
    loss_compile4 = iteration(input_tensor1, input_tensor2)

    assert loss_compile4 < loss_compile3
    assert loss_compile3 < loss_compile2
    assert loss_compile2 < loss_compile1
