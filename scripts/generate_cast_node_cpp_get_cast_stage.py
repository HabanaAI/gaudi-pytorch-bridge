# ******************************************************************************
# Copyright (C) 2021-2022 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
# ******************************************************************************

# Automatically generates body of the function:
#   CastStage get_cast_stage(CastTypes cast_types)
# in file:
#   pytorch-integration/pytorch_helpers/cast_sequence.cpp
#
# Input data comes from TPC kernels listed in:
#   tpc_kernels/src/kernel_factory_gaudi.cpp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--npu_stack_directory", "-d", type=str, required=True)

devices = ["gaudi", "gaudi2", "gaudi3"]
files_with_cast_kernels = {
    devices[0]: "/tpc_kernels/src/kernel_factory_gaudi.cpp",
    devices[1]: "/tpc_kernels/src/kernel_factory_gaudi2.cpp",
    devices[2]: "/tpc_kernels/src/kernel_factory_gaudi3.cpp",
}

sign_id = 0
exp_id = 1
mant_id = 2

# (sign, exponent, mantissa) bits
num_bits_gaudi = {
    "f32": (1, 8, 23),
    "bf16": (1, 8, 7),
    "i8": (1, 0, 7),
    "i16": (1, 0, 15),
    "i32": (1, 0, 31),
    "i64": (1, 0, 63),
    "u8": (0, 0, 8),
}

num_bits_gaudi2 = num_bits_gaudi.copy()
num_bits_gaudi2["f8"] = (1, 5, 2)
num_bits_gaudi2["hf8"] = (1, 4, 3)
num_bits_gaudi2["f16"] = (1, 5, 10)

num_bits_gaudi3 = num_bits_gaudi2.copy()


num_bits = {
    devices[0]: num_bits_gaudi,
    devices[1]: num_bits_gaudi2,
    devices[2]: num_bits_gaudi3,
}

cast_types = {}
max_len = {}
for device in devices:
    cast_types[device] = [key for key in num_bits[device].keys()]
    max_len[device] = max([len(t) for t in cast_types[device]])


def is_identity(src, dst):
    identities = []
    return (src, dst) in identities or (dst, src) in identities


def print_comment_table(casts, device):
    ident = "  "
    cast_types_local = cast_types[device]
    max_len_local = max_len[device]
    print("\n{}// cast".format(ident))

    label = "fr/to"
    label_len = len(label)
    first_col_len = max(max_len_local, label_len)

    first_row = "{0:<{1}}".format(label, first_col_len)
    for cast_type in cast_types_local:
        first_row += " " + cast_type
    print("{}// {}".format(ident, first_row))

    for src in cast_types_local:
        line = "{0:<{1}}".format(src, first_col_len)
        for dst in cast_types_local:
            c = "-"
            if src == dst:
                c = "*"
            elif is_identity(src, dst):
                c = "I"
            elif src in casts and dst in casts[src]:
                c = "X"
            line += "{0:>{1}}".format(c, len(dst) + 1)
        print("{}// {}".format(ident, line))


class CastOpWeight:
    def __init__(
        self,
        min_bits=None,
        num_steps=0,
        max_bits=None,
        min_int_mant=None,
        num_identities=0,
        max_mant_inc_after_sign_change=0,
        mant_bits_begin_end=None,
        num_bits=None,
    ):
        if num_bits:
            int_mant = num_bits[mant_id] if num_bits[exp_id] == 0 else 1000
            self.Init(
                num_bits,
                num_steps,
                num_bits,
                int_mant,
                num_identities,
                max_mant_inc_after_sign_change,
                mant_bits_begin_end,
            )
        else:
            self.Init(
                min_bits,
                num_steps,
                max_bits,
                min_int_mant,
                num_identities,
                max_mant_inc_after_sign_change,
                mant_bits_begin_end,
            )

    def Init(
        self,
        min_bits,
        num_steps,
        max_bits,
        min_int_mant,
        num_identities,
        max_mant_inc_after_sign_change,
        mant_bits_begin_end,
    ):
        self.min_bits = min_bits
        self.num_steps = num_steps
        self.max_bits = max_bits
        self.min_int_mant = min_int_mant
        self.num_identities = num_identities
        self.max_mant_inc_after_sign_change = max_mant_inc_after_sign_change
        self.mant_bits_begin_end = mant_bits_begin_end
        self.next = None

    def __add__(self, other):
        min_bits = tuple([min(b[0], b[1]) for b in zip(self.min_bits, other.min_bits)])
        num_steps = self.num_steps + other.num_steps
        max_bits = tuple([max(b[0], b[1]) for b in zip(self.max_bits, other.max_bits)])
        min_int_mant = min(self.min_int_mant, other.min_int_mant)
        num_identities = self.num_identities + other.num_identities
        max_mant_inc_after_sign_change = max(self.max_mant_inc_after_sign_change, other.max_mant_inc_after_sign_change)
        mant_bits_begin_end = [
            self.mant_bits_begin_end[0],
            other.mant_bits_begin_end[1],
        ]

        if self.min_bits[sign_id] != self.max_bits[sign_id]:
            max_mant_inc = max(mant_bits_begin_end[1] - mant_bits_begin_end[0], 0)
            max_mant_inc_after_sign_change = max(max_mant_inc_after_sign_change, max_mant_inc)

        return CastOpWeight(
            min_bits,
            num_steps,
            max_bits,
            min_int_mant,
            num_identities,
            max_mant_inc_after_sign_change,
            mant_bits_begin_end,
        )

    def __gt__(self, other):
        if self.min_bits[mant_id] != other.min_bits[mant_id]:
            return self.min_bits[mant_id] < other.min_bits[mant_id]  # less is deliberate
        elif self.min_bits[sign_id] != other.min_bits[sign_id]:
            return self.min_bits[sign_id] < other.min_bits[sign_id]  # less is deliberate
        elif self.min_bits[exp_id] != other.min_bits[exp_id]:
            return self.min_bits[exp_id] < other.min_bits[exp_id]  # less is deliberate
        elif self.min_int_mant != other.min_int_mant:
            # prefer bf16 -> i32 -> i16 (max_bits = 1,8,31, min_int_mant = 15)
            # over   bf16 ->  i8 -> i16 (max_bits = 1,8,15, min_int_mant = 7)
            # and vice versa
            return self.min_int_mant < other.min_int_mant  # less is deliberate
        elif self.max_mant_inc_after_sign_change != other.max_mant_inc_after_sign_change:
            # prefer i8 -> i32 -> u32, changing -1 to big positive number
            # over   i8 -> u8 -> u32,  changing -1 to 255
            # and similarly
            # even prefer i16 -> i32 -> i64 -> u64
            # over shorter i16 -> u32 -> u64
            return self.max_mant_inc_after_sign_change > other.max_mant_inc_after_sign_change
        elif self.num_steps != other.num_steps:
            return self.num_steps > other.num_steps
        elif self.max_bits[exp_id] != other.max_bits[exp_id]:
            return self.max_bits[exp_id] > other.max_bits[exp_id]
        elif self.max_bits[sign_id] != other.max_bits[sign_id]:
            return self.max_bits[sign_id] > other.max_bits[sign_id]
        elif self.max_bits[mant_id] != other.max_bits[mant_id]:
            # prefer i8 -> i32 -> i16 (max_bits = 1,0,31)
            # over   i8 -> i64 -> i16 (max_bits = 1,0,63)
            return self.max_bits[mant_id] > other.max_bits[mant_id]
        else:
            return self.num_identities > other.num_identities


def print_stages_list(weights, device):
    intermediates = set()
    max_len_interm = 0
    cast_types_local = cast_types[device]

    for src in cast_types_local:
        for dst in cast_types_local:
            w = weights[src][dst]
            if w.next and w.next != dst:
                intermediates.add(w.next)
                max_len_interm = max(max_len_interm, len(w.next))

    intermediates = list(intermediates)
    intermediates.sort()

    ident = "  "
    print("\n{}// clang-format off".format(ident))
    print("#define {0:<{1}} CastStage {{}}".format("OK", max_len_interm))
    for intermediate in intermediates:
        print(
            "#define {0:<{1}} CastStage {{ CastType::{2:<{3}} }}".format(
                intermediate.upper(), max_len_interm, intermediate, max_len_interm
            )
        )
    print("{}// clang-format on".format(ident))

    intermediates.reverse()
    return intermediates


def print_undefs(intermediates):
    print()
    for intermediate in intermediates:
        print("#undef {}".format(intermediate.upper()))
    print("#undef OK")


def print_mapping_table(weights, device):
    cast_types_local = cast_types[device]
    max_len_local = max_len[device]
    label_ok = "OK"
    columns_lens = {}
    for dst in cast_types_local:
        max_len_loc = len(dst)
        for weights_val in weights.values():
            w = weights_val[dst]
            label = w.next if w.next and w.next != dst else label_ok
            max_len_loc = max(max_len_loc, len(label))
            columns_lens[dst] = max_len_loc

    ident = "  "
    next_ident = ident + ident + ident

    print("\n{}// TODO: SW-35847 Remove indirect casting".format(ident))
    print("{}using LineT = EnumMappingTable<CastType, CastStage>;".format(ident))
    print("{}static const EnumMappingTable<CastType, LineT> cast_stage_matrix_{} = {{".format(ident, device))
    print("{}// clang-format off".format(next_ident))

    line = "{0}//         {1:<{2}} to:  ".format(next_ident, " ", max_len_local)
    for dst in cast_types_local:
        line += "  {0:>{1}}".format(dst, columns_lens[dst])
    print(line)

    for src in cast_types_local:
        line = "{0}/* from {1:>{2}} */ LineT{{".format(next_ident, src, max_len_local)
        prefix = ""
        for dst in cast_types_local:
            w = weights[src][dst]
            label = w.next if w.next and w.next != dst else label_ok
            line += "{0} {1:>{2}}".format(prefix, label.upper(), columns_lens[dst])
            prefix = ","
        line += " },"
        print(line)

    print("{}// clang-format on".format(next_ident))
    print("{}}};".format(ident))


def print_source(casts, weights, device):
    print(f"\n  // ======== {device} ========")
    print_comment_table(casts, device)
    intermediates = print_stages_list(weights, device)
    print_mapping_table(weights, device)
    print_undefs(intermediates)


def Floyd_Warshall(casts, device):
    weights = {}
    cast_types_local = cast_types[device]
    num_bits_local = num_bits[device]
    max_len_local = max_len[device]
    for src in cast_types_local:
        weights[src] = {}
        for dst in cast_types_local:
            if is_identity(src, dst):
                src_weight = CastOpWeight(
                    num_bits=num_bits_local[src],
                    num_identities=1,
                    mant_bits_begin_end=[num_bits_local[src][mant_id]] * 2,
                )
                dst_weight = CastOpWeight(
                    num_bits=num_bits_local[dst],
                    mant_bits_begin_end=[num_bits_local[dst][mant_id]] * 2,
                )
                weights[src][dst] = src_weight + dst_weight
                weights[src][dst].next = dst
            else:
                weights[src][dst] = CastOpWeight(
                    num_bits=(-1, -1, -1) if dst != src else (1000, 1000, 1000),
                    mant_bits_begin_end=[
                        num_bits_local[src][mant_id],
                        num_bits_local[dst][mant_id],
                    ],
                )
    for src, dst_cast_types in casts.items():
        for dst in dst_cast_types:
            if src == dst or is_identity(src, dst):
                print("Ignoring cast {}->{} as this is identity".format(src, dst))
            else:
                src_weight = CastOpWeight(
                    num_bits=num_bits_local[src],
                    num_steps=1,
                    mant_bits_begin_end=[num_bits_local[src][mant_id]] * 2,
                )
                dst_weight = CastOpWeight(
                    num_bits=num_bits_local[dst],
                    mant_bits_begin_end=[num_bits_local[dst][mant_id]] * 2,
                )
                weights[src][dst] = src_weight + dst_weight
                weights[src][dst].next = dst

    for u in cast_types_local:
        for v1 in cast_types_local:
            for v2 in cast_types_local:
                weight_test = weights[v1][u] + weights[u][v2]
                if weights[v1][v2] > weight_test:
                    weights[v1][v2] = weight_test
                    weights[v1][v2].next = weights[v1][u].next

    print("")
    for src in cast_types_local:
        for dst in cast_types_local:
            w = weights[src][dst]
            print(
                "{0:<{1}} {2:<{3}} min=({4:>4}, {5:>4}, {6:>4}) st={7} max=({8:>4}, {9:>4}, {10:>4}) mim={11:<4} ni={12} {13}".format(
                    src,
                    max_len_local,
                    dst,
                    max_len_local,
                    w.min_bits[sign_id],
                    w.min_bits[exp_id],
                    w.min_bits[mant_id],
                    w.num_steps,
                    w.max_bits[sign_id],
                    w.max_bits[exp_id],
                    w.max_bits[mant_id],
                    w.min_int_mant,
                    w.num_identities,
                    w.max_mant_inc_after_sign_change,
                    w.mant_bits_begin_end[0],
                    w.mant_bits_begin_end[1],
                    w.next,
                )
            )

    return weights


def generate_casts(file_with_cast_kernels, device):
    casts = {}
    cast_types_local = cast_types[device]
    with open(file_with_cast_kernels, "r") as read_obj:
        for line in read_obj:
            cast_pos = line.find("CastKernel::")
            if cast_pos >= 0:
                end_pos = line[cast_pos:].find(")")
                if end_pos < 0:
                    end_pos = 0
                else:
                    end_pos += cast_pos

                split = line[cast_pos:end_pos].split("::")
                words = split[1].split("_")
                if len(words) != 3:
                    print("Ommited non standard kernel {}".format(split[1]))
                    continue
                src = words[0]
                dst = words[2]
                if src in cast_types_local and dst in cast_types_local:
                    print("Read cast {} --> {}".format(src, dst))
                    casts.setdefault(src, set()).add(dst)
                else:
                    print("Omitted cast {} --> {}".format(src, dst))
    print("\nCasts summary:")
    for key, data in casts.items():
        print("{} -->".format(key))
        for dst in data:
            print("    {}".format(dst))

    weights = Floyd_Warshall(casts, device)
    return casts, weights


def main():
    args = parser.parse_args()
    casts = {}
    weights = {}

    for device, file in files_with_cast_kernels.items():
        print("\n======== {} ========\n".format(device))
        casts[device], weights[device] = generate_casts(args.npu_stack_directory + file, device)

    for device in devices:
        print_source(casts[device], weights[device], device)


if __name__ == "__main__":
    main()
