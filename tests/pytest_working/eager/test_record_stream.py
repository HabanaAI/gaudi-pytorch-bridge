###############################################################################
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

import os

import torch


class EnvironmentVariableSetter:
    """
    Allows temporary change of multiple environment variables.
    Requires a dictionary of environment variable names and their corresponding values.
    """

    def __init__(self, env_vars: dict):
        self._stored_keys = {}
        self._env_vars = env_vars

    def __enter__(self):
        for env_name, value in self._env_vars.items():
            self._stored_keys[env_name] = os.environ.get(env_name)
            os.environ[env_name] = str(value)

    def __exit__(self, *args):
        for env_name in self._env_vars.keys():
            if env_name in self._stored_keys:
                if self._stored_keys[env_name] is None:
                    del os.environ[env_name]
                else:
                    os.environ[env_name] = self._stored_keys[env_name]


def test_record_stream():
    with EnvironmentVariableSetter({"PT_HPU_ENABLE_RECORD_STREAM": 1, "PT_HPU_USE_LAUNCH_RECORD_STREAM": 1}):
        dev = torch.device("hpu")
        a = torch.randn((1024, 1024), device=dev)
        b = torch.randn((1024, 1024), device=dev)
        c = torch.randn((1024, 1024), device=dev)
        d = torch.randn((1024, 1024), device=dev)

        x = torch.relu(a)

        s1 = torch.hpu.Stream()
        with s1:
            # Since x is used in a different stream, so bridge will automatically
            # record the s1 stream to x's memory chunk.
            d.copy_(x, non_blocking=True)

        # Since x is recorded a stream s1, then it won't be really freed, until the
        # copy operation on s1 is done.
        del x

        # Do compute on default stream. At the same time, copy is running on s1
        # stream.
        y = torch.mm(b, c)

        # Wait for x to d copy done.
        s1.synchronize()

        # Use the copy reuslt on default stream.
        # Since the copy on s1 is done, the x memory chunk can be reused for z.
        z = torch.mm(y, d)

    print("success")


if __name__ == "__main__":
    test_record_stream()
