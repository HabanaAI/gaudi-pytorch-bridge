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

from tools import *


def main(args):
    dev_1 = args.device1
    dev_2 = args.device2
    path_1 = args.data_path1
    path_2 = args.data_path2
    rtol = args.rtol
    atol = args.atol
    topology = args.topology

    base_path = path_1
    file_pair_list = ca_make_file_pair_list(dev_1, dev_2, path_1, path_2)
    ca_compare_tensor_files(dev_1, dev_2, file_pair_list, base_path, rtol=rtol, atol=atol, topology=topology,skip_pattern=args.skip_pattern,rms_threshold=float(args.rms_threshold))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Convergence Analysis Tensor Comparison')

    parser.add_argument('--device1', default='cpu', help='first device')
    parser.add_argument('--device2', default='hpu', help='second device')
    parser.add_argument('--data-path1', help=' Tensor data path for dev1')
    parser.add_argument('--data-path2', help=' Tensor data path for dev2 if its data is in a different path')
    parser.add_argument('--rtol', default=1e-3, type=float, help='relative tolerance for maxabsdiff check')
    parser.add_argument('--atol', default=1e-3, type=float, help='absolute tolerance for maxabsdiff check')
    parser.add_argument('--topology', default='', help='Topology name. Give resnet for resnet50. Needed only if tensor comp needs topology specicif steps. Currently needed only for resnet')
    parser.add_argument("--skip-pattern",default='None', help='skip pattern for tensor comparison')
    parser.add_argument("--rms-threshold",default='1e-10', help='Threshold for tensor RMS to bypass cosine check')

    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
