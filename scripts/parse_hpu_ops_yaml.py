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

# Parses the ops yaml file (scripts/hpu_ops.yaml) and creates as csv file
# with a table of ops and the entries for  yaml fileds like

# op_name | acc_thread|broadcast|custom_fill_params|custom_output_shape|dtype| .......
# hardtanh|NA|NA|FillClampParams|NA|NA|['BFloat16', 'Float', 'Int']|......

# The outfile is : parsed_hpu_ops_yaml.csv

# usage: python parse_hpu_ops_yaml.py --pt_integ_path=<path to pytorch-integration git>

# if --pt_integ_path is not given, the path is taken from "PYTORCH_MODULES_ROOT_PATH"
# env variable


import argparse
import csv
import os

import yaml


def parse_yaml(f_yaml):
    keyset = set()
    with open(f_yaml, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            # print(yaml_dict)
            for k, v in yaml_dict.items():
                for vv in v.keys():
                    keyset.add(vv)
        except yaml.YAMLError as exc:
            print(exc)
    # print(keyset)
    op_dict = dict()
    with open(f_yaml, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            for k, v in yaml_dict.items():
                temp_dict = dict.fromkeys(keyset, "NA")
                op_entry_dict = {**temp_dict, **v}
                op_dict[k] = op_entry_dict
        except yaml.YAMLError as exc:
            print(exc)
    # print(op_dict)

    with open("parsed_hpu_ops_yaml.csv", "w", newline="") as op_csv:
        header = ["op_name"] + sorted(list(keyset))
        writer = csv.DictWriter(op_csv, fieldnames=header, delimiter="|")
        writer.writeheader()
        for k in op_dict.keys():
            v = op_dict.get(k)
            v["op_name"] = k
            writer.writerow(v)


def main(args):
    if args.pt_integ_path == None:
        pt_integ_path = os.environ["PYTORCH_MODULES_ROOT_PATH"]
    else:
        pt_integ_path = args.pt_integ_path

    yaml_file = os.path.join(pt_integ_path, "scripts/hpu_op.yaml")
    parse_yaml(yaml_file)


def parse_args_and_run_main(argv=None):
    # for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_integ_path", default=None, help="path of pytorch integration git")
    args = parser.parse_args(argv)
    main(args)


if __name__ == "__main__":
    parse_args_and_run_main()
