# ******************************************************************************
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

import argparse
import os
from dataclasses import dataclass, field

parser = argparse.ArgumentParser()
parser.add_argument("--directory", "-d", help="pytorch-integration root directory", required=True)


def errormsg(file, ln, msg):
    return f"ERROR in {file}:{ln}\n{msg}"


def read_namespace(file, ln, line):
    b = line.find("(")
    e = line.find(",")
    if b >= 0 and e > b + 1:
        return line[b + 1 : e]
    else:
        err = errormsg(file, ln, f'Can\'t read namespace from line "{line}"')
        raise Exception(err)


@dataclass(frozen=True, order=True)
class MdefData:
    name: str
    schema: str
    file: str
    ln: int = field(compare=False)


mdefs = {}


def add_mdef(file, ln, ns, line):
    b = line.find('"')
    if b >= 0:
        m = line.find("(", b)
        e = line.find('"', m)
        if m > b + 1 and e > m + 1:
            mdef = MdefData(name=line[b + 1 : m], schema=line[b + 1 : e], file=file, ln=ln)
            if ns in mdefs and mdef in mdefs[ns]:
                mdefs_list = list(sorted(mdefs[ns]))
                prev_mdef = mdefs_list[mdefs_list.index(mdef)]
                err = errormsg(file, ln, f'mdef from line "{line}" repeated')
                err += f"\nPrevious definition was in {prev_mdef.file}:{prev_mdef.ln}"
                raise Exception(err)
            else:
                mdefs.setdefault(ns, set()).add(mdef)
                return

    err = errormsg(file, ln, f'Can\'t read mdef from line "{line}"')
    raise Exception(err)


def read_file(args, filename):
    if any([filename.endswith(e) for e in [".cpp", ".hpp", ".h"]]):
        with open(filename, "r") as fr:
            state = "outside"
            ln = 0
            for line in fr:
                ln += 1
                line = line.strip()
                if not line:
                    continue
                if state == "outside":
                    if any(line.startswith(pat) for pat in ["TORCH_LIBRARY_FRAGMENT(", "TORCH_LIBRARY("]):
                        namespace = read_namespace(filename, ln, line)
                        if namespace not in ["test_ops", "custom_op"]:
                            state = "mdefs"
                            combined_line = ""
                elif state == "mdefs":
                    if line == "}":
                        state = "outside"
                        continue
                    if line == "static_cast<void>(m);":
                        continue
                    if line.startswith("m.def("):
                        combined_line = line
                    else:
                        combined_line += line
                    if line.endswith(");"):
                        add_mdef(filename, ln, namespace, combined_line)
                        combined_line = ""


def read_directory(args, directory):
    for f_or_d in os.scandir(path=directory):
        fullname = directory + "/" + f_or_d.name
        if f_or_d.is_dir():
            read_directory(args, fullname)
        else:
            read_file(args, fullname)


def read_directory_with_check(args, directory):
    if os.path.isdir(directory):
        return read_directory(args, directory)

    raise Exception(f"Provided directory {directory} doesn't exist")


def main():
    args = parser.parse_args()
    try:
        custom_ops = read_directory_with_check(args, args.directory)
    except Exception as e:
        print(e)
        return

    maxlen = {}
    for ns, mdefset in mdefs.items():
        maxlen["ns"] = max(maxlen.setdefault("ns", 0), len(ns))
        for mdef in mdefset:
            for key in ["name", "schema"]:
                maxlen[key] = max(maxlen.setdefault(key, 0), len(getattr(mdef, key)))
            maxlen["file_with_ln"] = max(maxlen.setdefault("file_with_ln", 0), len(mdef.file) + 1 + len(str(mdef.ln)))

    for ns, mdefset in sorted(mdefs.items()):
        ns_str = f"{ns+';':{maxlen['ns']+1}}"
        for mdef in sorted(mdefset):
            file_with_ln = f"{mdef.file}:{mdef.ln}"
            print(
                f"{file_with_ln+';':{maxlen['file_with_ln']+1}} {ns_str} {mdef.name+';':{maxlen['name']+1}} {mdef.schema};"
            )


if __name__ == "__main__":
    main()
