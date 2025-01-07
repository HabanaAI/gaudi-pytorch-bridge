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


import argparse
import glob
import urllib.request

import inflection

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", "-c", help="CPU ops file", required=True)
parser.add_argument("--gpu", "--cuda", "-g", help="GPU/CUDA ops file", required=True)
parser.add_argument("--hpu", "-hh", help="HPU ops file", required=True)
parser.add_argument("--filter", "-f", action="store_true")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--prioritize", "-prio", action="store_true")
parser.add_argument("--python_path", "-p")
parser.add_argument("--npu_stack", "-n")


def bool_to_yes(v):
    return "YES;" if v else "NO; "


def filter(op, filtered_out):
    dot_pos = op.find(".")
    if dot_pos >= 0:
        filtered_out.add(op[dot_pos:])
        op = op[:dot_pos]

    if len(op) > 1 and op[-1] == "_" and op[-2] != "_":
        op = op[:-1]

    return op, filtered_out


def read_ops(args, file, all_ops):
    filtered_out = set()
    ops = set()
    with open(file, "r") as read_obj:
        for line in read_obj:
            op = line.strip()

            if args.filter:
                op, filtered_out = filter(op, filtered_out)

            ops.add(op)
            all_ops.add(op)

    if args.verbose:
        for filt in sorted(filtered_out):
            print(filt)

    return sorted(ops), all_ops


def read_registration_declarations(args, python_path):
    filtered_out = set()
    reg_decl = {}
    with open(
        python_path + "/torch/include/ATen/RegistrationDeclarations.h",
        "r",
    ) as read_obj:
        for line in read_obj:
            pattern = '"schema": "'
            op_beg = line.find(pattern)
            if op_beg >= 0:
                op = line[op_beg + len(pattern) :]
                op_end = op.find("(")
                if op_end >= 0:
                    op = op[:op_end]

                    def check_key(key):
                        if '"' + key + '": "False"' in line:
                            return False
                        elif '"' + key + '": "True"' in line:
                            return True
                        else:
                            return None

                    dispatch = check_key("dispatch")
                    default = check_key("default")

                    dide = "D" + ("T" if dispatch else "F") + "_D" + ("T" if default else "F") + ";"
                    compound = not dispatch and default

                    if args.filter:
                        op, filtered_out = filter(op, filtered_out)

                    reg_decl[op] = {
                        "compound": bool_to_yes(compound),
                        "di_de": dide,
                    }

    if args.verbose:
        for filt in sorted(filtered_out):
            print(filt)

    return reg_decl


def read_http_operators_supported(args, url):
    f = urllib.request.urlopen(url)
    html_doc = f.read().decode("utf-8")
    pattern = "aten::"
    posb = html_doc.find(pattern)
    ops = set()
    while posb >= 0:
        pose = html_doc.find("(", posb)
        if pose >= 0:
            op = html_doc[posb:pose]
            while True:
                b = op.find("<")
                e = op.find(">")
                if b >= 0 and e >= 0 and e > b:
                    op = op[:b] + op[e + 1 :]
                else:
                    break

            if args.filter:
                op, filtered_out = filter(op, set())

            ops.add(op)
            posb = html_doc.find(pattern, pose)
        else:
            posb = -1

    ops = sorted(ops)
    if args.verbose:
        print(f"Ops read from: {url}")
        for op in ops:
            print(op)

    return ops


def read_preambler_data(args, prea_path):
    ops = {}
    for filename in glob.iglob(prea_path + "/**/*.csv", recursive=True):
        if args.verbose:
            print(f"Reading preambler data: {filename}")
        with open(filename, "r") as read_obj:
            for line in read_obj:
                pattern = ",Torch,"
                pos = line.find(pattern)
                if pos >= 0:
                    topo = line[:pos]
                    op = line[pos + len(pattern) :]
                    comma_pos = op.find(",")
                    if comma_pos >= 0:
                        op = op[:comma_pos]

                    if args.filter:
                        op, filtered_out = filter(op, set())
                        op = op.replace("_", "")

                    op = op.replace("2D", "2d")
                    op = op.replace("3D", "3d")
                    op = op.replace("ReLU", "relu")
                    op = inflection.underscore(op)
                    op = "aten::" + op

                    ops.setdefault(op, set()).add(topo)

    for key in ops.keys():
        ops[key] = sorted(ops[key])

    if args.verbose:
        max_len = 0
        for key in ops.keys():
            max_len = max(max_len, len(key))

        print("Preambler data:")
        for key in sorted(ops.keys()):
            str = key + " " * (max_len - len(key))

            sep = " : "
            for value in ops[key]:
                str = str + sep + value
                sep = ", "
            print(str)

    return ops


def prioritize(args, ops_summary, reg_decl):
    def int_from_dict(dict, key1, key2):
        if key1 in dict:
            if dict[key1][key2] == "DT_DF;":
                return 0
            elif dict[key1][key2] == "DT_DT;":
                return 1
            else:
                return 2
        else:
            return 3

    def key_fun(v):
        return (
            int(v["HPU"]),
            int_from_dict(reg_decl, v["name"], "di_de"),
            int(not v["missing"]),
            -len(v["topos_list"]),
            v["name"],
        )

    return sorted(ops_summary, key=key_fun)


def main():
    args = parser.parse_args()
    key_list = ["CPU", "GPU", "HPU"]
    ops = {}
    all_ops_set = set()
    for key in key_list:
        file = getattr(args, key.lower())
        ops[key], all_ops_set = read_ops(args, file, all_ops_set)
        if args.verbose:
            print(f"len(ops['{key}']) = {len(ops[key])} {len(all_ops_set) = }")

    if args.python_path:
        reg_decl = read_registration_declarations(args, args.python_path)
        all_ops_set.update(list(reg_decl.keys()))
        if args.verbose:
            print(f"{len(reg_decl) = } {len(all_ops_set) = }")
    else:
        reg_decl = {}

    all_ops = sorted(all_ops_set)

    max_len = 0
    for op in all_ops:
        max_len = max(max_len, len(op))

    if args.npu_stack:
        ops_in_topos = read_preambler_data(
            args,
            args.npu_stack + "/habanaqa/tests/graph_compiler/preambler/results/pytorch_sng",
        )
    else:
        ops_in_topos = {}

    tert_ops = read_http_operators_supported(args, "https://pytorch.org/TensorRT/indices/supported_ops.html")

    ops_summary = []
    for op in all_ops:
        entry = {"name": op}
        for key in key_list:
            entry[key] = op in ops[key]
        entry["missing"] = entry["GPU"] and not entry["HPU"]
        entry["tert"] = op in tert_ops

        op_key = op
        if not op_key in ops_in_topos:
            op_key = op_key.replace(".Tensor", "")

        entry["in_topos"] = op_key in ops_in_topos
        entry["topos_list"] = sorted(ops_in_topos[op_key]) if entry["in_topos"] else []
        ops_summary.append(entry)

    ops_decl_only = []
    ops_prea = []
    for prea_op, topos_list in ops_in_topos.items():
        if prea_op not in all_ops and prea_op + ".Tensor" not in all_ops:
            if prea_op in reg_decl:
                name = prea_op
            else:
                name = "preambler::" + prea_op.replace("aten::", "")
                max_len = max(max_len, len(name))

            entry = {
                "name": name,
                "missing": False,
                "tert": name in tert_ops,
                "in_topos": True,
                "topos_list": topos_list,
            }
            for key in key_list:
                entry[key] = False

            if prea_op in reg_decl:
                ops_decl_only.append(entry)
            else:
                ops_prea.append(entry)

    def sort_by_name(v):
        return v["name"]

    ops_decl_only.sort(key=sort_by_name)
    ops_prea.sort(key=sort_by_name)

    ops_summary.extend(ops_decl_only)
    ops_summary.extend(ops_prea)

    keys_str = ""
    for key in key_list:
        keys_str += " " + key + ";"

    def from_dict(dict, val, cat):
        if val in dict:
            return dict[val][cat]
        else:
            return "NA; "

    if args.prioritize:
        ops_summary = prioritize(args, ops_summary, reg_decl)

    print(f"{'OP;':<{max_len + 1}}{keys_str} COMPOUND; DI_DE; MISSING; TERT; IN_TOPOS; COUNT; TOPOS_LIST")
    for entry in ops_summary:
        op = entry["name"]

        keys_str = ""
        for key in key_list:
            keys_str += " " + bool_to_yes(entry[key])

        topos_str = ""
        sep = " "
        for topo in entry["topos_list"]:
            topos_str += sep + topo
            sep = ", "

        print(
            "{0:<{1}}{2} {3:<9} {4:<6} {5:<8} {6:<5} {7:<9} {8:>5};{9}".format(
                op + ";",
                max_len + 1,
                keys_str,
                from_dict(reg_decl, op, "compound"),
                from_dict(reg_decl, op, "di_de"),
                bool_to_yes(entry["missing"]),
                bool_to_yes(entry["tert"]),
                bool_to_yes(entry["in_topos"]),
                len(entry["topos_list"]),
                topos_str,
            )
        )


if __name__ == "__main__":
    main()
