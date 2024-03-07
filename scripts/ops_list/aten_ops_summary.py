# ******************************************************************************
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--hpu", "-hh", help="HPU ops file", required=True)
parser.add_argument("--verbose", "-v", action="store_true")


def bool_to_yes(v):
    return "YES" if v else "NO"


def read_ops(args, file):
    ops = set()
    with open(file, "r") as read_obj:
        for line in read_obj:
            op = line.strip()
            last_space = op.rfind(" ")
            if last_space >= 0:
                op = op[last_space + 1 :]
            ops.add(op)

    if args.verbose:
        for op in sorted(ops):
            print(op)

    return sorted(ops)


def read_http_aten_ops_supported(args, url):
    f = urllib.request.urlopen(url)
    html_doc = f.read().decode("utf-8")
    pattern = ">aten."
    posb = html_doc.find(pattern)
    ops = set()
    max_len = 0
    while posb >= 0:
        posb += len(pattern)
        pose = html_doc.find("<", posb)
        if pose >= 0:
            op = "aten::" + html_doc[posb:pose]
            ops.add(op)
            max_len = max(max_len, len(op))
            posb = html_doc.find(pattern, pose)
        else:
            posb = -1

    ops = sorted(ops)
    if args.verbose:
        print("Ops read from: {}".format(url))
        for op in ops:
            print(op)

    return max_len, ops


def main():
    args = parser.parse_args()
    ops = read_ops(args, args.hpu)

    max_len, aten_ops = read_http_aten_ops_supported(args, "https://pytorch.org/docs/master/ir.html")

    print("{0:<{1}} {2}".format("ATEN IR OP;", max_len + 1, "SUPPORTED"))
    for aten_op in aten_ops:
        supported = aten_op in ops
        print("{0:<{1}} {2}".format(aten_op + ";", max_len + 1, bool_to_yes(supported)))


if __name__ == "__main__":
    main()
