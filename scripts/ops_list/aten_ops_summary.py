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
parser.add_argument("--filter", "-f", action="store_true")
parser.add_argument("--verbose", "-v", action="store_true")


def bool_to_yes(v):
    return "YES" if v else "NO"


def filter(op, filtered_out):
    dot_pos = op.find(".")
    if dot_pos >= 0:
        filtered_out.add(op[dot_pos:])
        op = op[:dot_pos]

    return op, filtered_out


def read_ops(args, file):
    filtered_out = set()
    ops = set()
    with open(file, "r") as read_obj:
        for line in read_obj:
            op = line.strip().replace("aten::", "")

            if args.filter:
                op, filtered_out = filter(op, filtered_out)

            ops.add(op)

    if args.verbose:
        for filt in sorted(filtered_out):
            print(filt)

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
            op = html_doc[posb:pose]
            if args.filter:
                op, filtered_out = filter(op, set())

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

    max_len, aten_ops = read_http_aten_ops_supported(
        args, "https://pytorch.org/docs/master/ir.html"
    )

    print("{0:<{1}} {2}".format("ATEN IR OP;", max_len + 1, "SUPPORTED"))
    for aten_op in aten_ops:
        supported = aten_op in ops
        print(
            "{0:<{1}} {2}".format(
                aten_op + ";", max_len + 1, bool_to_yes(supported)
            )
        )


if __name__ == "__main__":
    main()
