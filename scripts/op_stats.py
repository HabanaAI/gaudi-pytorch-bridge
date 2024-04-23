#!/usr/bin/python3
###############################################################################
# Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
# All Rights Reserved.
#
# Unauthorized copying of this file or any element(s) within it, via any medium
# is strictly prohibited.
# This file contains Habana Labs, Ltd. proprietary and confidential information
# and is subject to the confidentiality and license agreements under which it
# was provided.
#
###############################################################################
import argparse
import csv
import datetime
import glob
import json
import os

import pandas as pd
import yaml


def match_any(l, match):
    for m in match:
        if m in l:
            return True
    return False


def extract_signature_list(lst):
    sl = []
    for l in lst:
        sl.append(extract_signature(l))
    return sl


# checks if stats of a sublist of ops is to be written.
# if yes, returns the sublist of ops listed in the file and
# the path of the output file where the stats is written.
# the output file path and name are same as that of the input file
# just that the output file name has ".out.csv" at the end.
def need_op_sublist_stats():
    need = False
    s = os.getenv("PT_OP_STATS_SUBLIST", None)
    if s:
        if not os.path.isfile(s):
            print("Op stats requested for sublist, but path", s, "does not exist")
            assert 0
        else:
            need = True
    if not need:
        return None, None

    op_sl_file = open(s, "r")
    op_sl = op_sl_file.readlines()
    op_sl_array = []
    for line in op_sl:
        l = line.strip()
        if len(l) != 0:
            op_sl_array.append(line.strip())

    sublist_stats_output_fname = s + ".out.csv"

    return op_sl_array, sublist_stats_output_fname


def write_consolidated_op_list(name, op_d):
    Total = 0
    rv = 0
    nrv = 0
    rv_df_dt_t = 0
    rv_df_dt_impl = 0
    rv_df_dt_nimpl = 0
    rv_df_dt_impl_m = 0
    rv_df_dt_impl_a = 0

    rv_dt_dt_t = 0
    rv_dt_dt_impl = 0
    rv_dt_dt_nimpl = 0
    rv_dt_df_t = 0
    rv_dt_df_impl = 0
    rv_dt_df_nimpl = 0

    rv_dt_dt_impl_a = 0
    rv_dt_dt_impl_m = 0
    rv_dt_df_impl_a = 0
    rv_dt_df_impl_m = 0

    with open(name, "w", newline="") as op_csv:
        header = [
            "op_name",
            "relevant_to_hpu",
            "op_category",
            "op_type",
            "implemented",
            "implement_method",
            "op_signature",
        ]
        writer = csv.DictWriter(op_csv, fieldnames=header, delimiter="|")
        writer.writeheader()
        for k in sorted(op_d.keys()):
            v = op_d.get(k)
            row = {}
            row["op_name"] = k
            row["op_signature"] = v["op_with_sig"]
            row["relevant_to_hpu"] = v["relevant"]
            row["op_category"] = v["type"]
            row["op_type"] = v["type2"]
            row["implemented"] = v["impld"]
            row["implement_method"] = v["impl_method"]
            writer.writerow(row)

            Total += 1
            if row["relevant_to_hpu"] == "yes":
                rv = rv + 1
                if row["op_type"] == "df_dt":
                    rv_df_dt_t += 1
                    if row["implemented"] == "yes":
                        rv_df_dt_impl += 1
                        if row["implement_method"] == "auto":
                            rv_df_dt_impl_a += 1
                        elif row["implement_method"] == "manual":
                            rv_df_dt_impl_m += 1
                elif row["op_type"] == "dt_dt":
                    rv_dt_dt_t += 1
                    if row["implemented"] == "yes":
                        rv_dt_dt_impl += 1
                        if row["implement_method"] == "auto":
                            rv_dt_dt_impl_a += 1
                        elif row["implement_method"] == "manual":
                            rv_dt_dt_impl_m += 1
                elif row["op_type"] == "dt_df":
                    rv_dt_df_t += 1
                    if row["implemented"] == "yes":
                        rv_dt_df_impl += 1
                        if row["implement_method"] == "auto":
                            rv_dt_df_impl_a += 1
                        elif row["implement_method"] == "manual":
                            rv_dt_df_impl_m += 1
    nrv = Total - rv
    rv_df_dt_nimpl = rv_df_dt_t - rv_df_dt_impl
    rv_dt_dt_nimpl = rv_dt_dt_t - rv_dt_dt_impl
    rv_dt_df_nimpl = rv_dt_df_t - rv_dt_df_impl

    rv_non_mandat_t = rv_df_dt_t + rv_dt_dt_t

    with open("summary.csv", "w", newline="") as summary_csv:
        header = [
            "total_ops",
            "relevant_on_hpu",
            "not_relevant_on_hpu",
            "non_mandatroy_total",
            "df_dt_total",
            "df_dt_implemented",
            "df_dt_implemented_auto",
            "df_dt_implemented_manual",
            "df_dt_remaining",
            "dt_dt_total",
            "dt_dt_implemented",
            "dt_dt_implemented_auto",
            "dt_dt_implemented_manual",
            "dt_dt_remaining",
            "mandatory_total",
            "dt_df_implemented",
            "dt_df_implemented_auto",
            "dt_df_implemented_manual",
            "dt_df_remaining",
        ]
        writer = csv.DictWriter(summary_csv, fieldnames=header, delimiter="|")
        writer.writeheader()
        row = {}

        row["total_ops"] = Total
        row["relevant_on_hpu"] = rv
        row["not_relevant_on_hpu"] = nrv
        row["non_mandatroy_total"] = rv_non_mandat_t

        row["df_dt_total"] = rv_df_dt_t
        row["df_dt_implemented"] = rv_df_dt_impl
        row["df_dt_implemented_auto"] = rv_df_dt_impl_a
        row["df_dt_implemented_manual"] = rv_df_dt_impl_m
        row["df_dt_remaining"] = rv_df_dt_nimpl

        row["dt_dt_total"] = rv_dt_dt_t
        row["dt_dt_implemented"] = rv_dt_dt_impl
        row["dt_dt_implemented_auto"] = rv_dt_dt_impl_a
        row["dt_dt_implemented_manual"] = rv_dt_dt_impl_m
        row["dt_dt_remaining"] = rv_dt_dt_nimpl

        row["mandatory_total"] = rv_dt_df_t
        row["dt_df_implemented"] = rv_dt_df_impl
        row["dt_df_implemented_auto"] = rv_dt_df_impl_a
        row["dt_df_implemented_manual"] = rv_dt_df_impl_m
        row["dt_df_remaining"] = rv_dt_df_nimpl
        writer.writerow(row)

    # Write ops sublist stats
    op_sl_array, sublist_stats_output_fname = need_op_sublist_stats()

    if sublist_stats_output_fname is None:
        return

    with open(sublist_stats_output_fname, "w", newline="") as op_subl_csv:
        header = [
            "op_name",
            "relevant_to_hpu",
            "op_category",
            "op_type",
            "implemented",
            "implement_method",
            "op_signature",
        ]
        writer = csv.DictWriter(op_subl_csv, fieldnames=header, delimiter="|")
        writer.writeheader()
        for k in op_sl_array:
            v = op_d.get(k)
            print("\n ", k)
            row = {}
            row["op_name"] = k
            row["op_signature"] = v["op_with_sig"]
            row["relevant_to_hpu"] = v["relevant"]
            row["op_category"] = v["type"]
            row["op_type"] = v["type2"]
            row["implemented"] = v["impld"]
            row["implement_method"] = v["impl_method"]
            writer.writerow(row)


def write_unique_op_list_v1(name, op_d):
    with open(name, "w", newline="") as op_csv:
        header = [
            "unique_op_name",
            "relevant",
            "variants",
            "non_compound_not_impl",
            "non_compound_impl",
            "compound_impl",
            "compound_not_impl",
            "total_non_compound",
            "total_compound",
        ]
        writer = csv.DictWriter(op_csv, fieldnames=header, delimiter="|")
        writer.writeheader()
        for k in sorted(op_d.keys()):
            v = op_d.get(k)
            row = {}
            row["unique_op_name"] = k
            row["relevant"] = v["rlv"]
            row["variants"] = v["total_variants"]
            row["non_compound_not_impl"] = v["ncvni"]
            row["non_compound_impl"] = v["ncvi"]
            row["compound_impl"] = v["cvi"]
            row["compound_not_impl"] = v["cvni"]
            row["total_non_compound"] = v["tnc"]
            row["total_compound"] = v["tc"]
            writer.writerow(row)


def write_unique_op_list_v2(name, op_d):
    with open(name, "w") as op_csv:
        header = [
            "unique_op_name",
            "relevant",
            "variants",
            "df_df_not_impl",
            "df_df_impl",
            "df_dt_not_impl",
            "df_dt_impl",
            "dt_df_not_impl",
            "dt_df_impl",
            "dt_dt_not_impl",
            "dt_dt_impl",
            "df_df_total",
            "df_dt_total",
            "dt_df_total",
            "dt_dt_total",
        ]
        writer = csv.DictWriter(op_csv, fieldnames=header, delimiter="|")
        writer.writeheader()
        for k in sorted(op_d.keys()):
            v = op_d.get(k)
            row = {}
            row["unique_op_name"] = k
            row["relevant"] = v["rlv"]
            row["variants"] = v["total_variants"]
            row["df_df_not_impl"] = v["df_df_ni"]
            row["df_df_impl"] = v["df_df_i"]
            row["df_dt_not_impl"] = v["df_dt_ni"]
            row["df_dt_impl"] = v["df_dt_i"]
            row["dt_df_not_impl"] = v["dt_df_ni"]
            row["dt_df_impl"] = v["dt_df_i"]
            row["dt_dt_not_impl"] = v["dt_dt_ni"]
            row["dt_dt_impl"] = v["dt_dt_i"]
            row["df_df_total"] = v["t_df_df"]
            row["df_dt_total"] = v["t_df_dt"]
            row["dt_df_total"] = v["t_dt_df"]
            row["dt_dt_total"] = v["t_dt_dt"]
            writer.writerow(row)


def get_unique_op_name(n):
    s = n.split(".")[0]

    if s.endswith("_"):
        return s[:-1]
    else:
        return s


def unique_ops_stats_v1(unique_ops, pt_op_dict):
    unique_op_dict = {}
    for uop in unique_ops:
        for k, v in pt_op_dict.items():
            p_uop = get_unique_op_name(k)
            def_dict = {"total_variants": 0, "cvi": 0, "cvni": 0, "ncvi": 0, "ncvni": 0, "rlv": "no", "tc": 0, "tnc": 0}
            if uop == p_uop:
                unique_op_dict[uop] = unique_op_dict.get(uop, def_dict)
                unique_op_dict[uop]["total_variants"] = unique_op_dict[uop].get("total_variants", 0) + 1
                if pt_op_dict[k]["relevant"] == "yes":
                    unique_op_dict[uop]["rlv"] = "yes"
                else:
                    unique_op_dict[uop]["rlv"] = "no"

                if pt_op_dict[k]["type"] == "compound":
                    unique_op_dict[uop]["tc"] = unique_op_dict[uop].get("tc", 0) + 1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["cvi"] = unique_op_dict[uop].get("cvi", 0) + 1
                    else:
                        unique_op_dict[uop]["cvni"] = unique_op_dict[uop].get("cvni", 0) + 1
                else:
                    unique_op_dict[uop]["tnc"] = unique_op_dict[uop].get("tnc", 0) + 1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["ncvi"] = unique_op_dict[uop].get("ncvi", 0) + 1
                    else:
                        unique_op_dict[uop]["ncvni"] = unique_op_dict[uop].get("ncvni", 0) + 1

    return unique_op_dict


def unique_ops_stats_v2(unique_ops, pt_op_dict):
    unique_op_dict = {}
    for uop in unique_ops:
        for k, v in pt_op_dict.items():
            p_uop = get_unique_op_name(k)
            def_dict = {
                "total_variants": 0,
                "df_df_i": 0,
                "df_df_ni": 0,
                "df_dt_i": 0,
                "df_dt_ni": 0,
                "dt_df_i": 0,
                "dt_df_ni": 0,
                "dt_dt_i": 0,
                "dt_dt_ni": 0,
                "rlv": "no",
                "t_df_df": 0,
                "t_df_dt": 0,
                "t_dt_df": 0,
                "t_dt_dt": 0,
            }
            if uop == p_uop:
                unique_op_dict[uop] = unique_op_dict.get(uop, def_dict)
                unique_op_dict[uop]["total_variants"] = unique_op_dict[uop].get("total_variants", 0) + 1
                if pt_op_dict[k]["relevant"] == "yes":
                    unique_op_dict[uop]["rlv"] = "yes"
                else:
                    unique_op_dict[uop]["rlv"] = "no"

                if pt_op_dict[k]["type2"] == "df_df":
                    unique_op_dict[uop]["t_df_df"] = unique_op_dict[uop].get("t_df_df", 0) + 1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["df_df_i"] = unique_op_dict[uop].get("df_df_i", 0) + 1
                    else:
                        unique_op_dict[uop]["df_df_ni"] = unique_op_dict[uop].get("df_df_ni", 0) + 1
                elif pt_op_dict[k]["type2"] == "df_dt":
                    unique_op_dict[uop]["t_df_dt"] = unique_op_dict[uop].get("t_df_dt", 0) + 1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["df_dt_i"] = unique_op_dict[uop].get("df_dt_i", 0) + 1
                    else:
                        unique_op_dict[uop]["df_dt_ni"] = unique_op_dict[uop].get("df_dt_ni", 0) + 1
                elif pt_op_dict[k]["type2"] == "dt_df":
                    unique_op_dict[uop]["t_dt_df"] = unique_op_dict[uop].get("t_dt_df", 0) + 1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["dt_df_i"] = unique_op_dict[uop].get("dt_df_i", 0) + 1
                    else:
                        unique_op_dict[uop]["dt_df_ni"] = unique_op_dict[uop].get("dt_df_ni", 0) + 1
                elif pt_op_dict[k]["type2"] == "dt_dt":
                    unique_op_dict[uop]["t_dt_dt"] = unique_op_dict[uop].get("t_dt_dt", 0) + 1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["dt_dt_i"] = unique_op_dict[uop].get("dt_dt_i", 0) + 1
                    else:
                        unique_op_dict[uop]["dt_dt_ni"] = unique_op_dict[uop].get("dt_dt_ni", 0) + 1
                else:
                    print("got type2 as ", pt_op_dict[k]["type2"])
                    assert 0, "invalid type2"

    return unique_op_dict


def combine_auto_generated_files(p):
    # Auto generated files are split into multiple files like hpu_op0.cpp, hpu_op1.cpp ...
    # Combine them(read the lines in each file and return combined set of lines)
    px = p + "/hpu_op*[0-9].cpp"
    print(px)
    files = glob.glob(px)
    print(files)
    l_auto_ops_decl = []
    for f in files:
        f_auto_ops_decl = open(f, "r")
        l = f_auto_ops_decl.readlines()
        l_auto_ops_decl.extend(l)
        f_auto_ops_decl.close()
    return l_auto_ops_decl


def get_manual_ops_with_overrides_in_yaml(f_yaml):
    manual_ops_override = []
    with open(f_yaml, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            # print(yaml_dict)
            for k, v in yaml_dict.items():
                if "override_fn" in v.keys():
                    # print("kkkkkk" ,k)
                    manual_ops_override.append(k)
        except yaml.YAMLError as exc:
            print(exc)
    return manual_ops_override


def is_op_overridden_in_yaml(op_name, ops_override_list):
    for op in ops_override_list:
        if op_name == op:
            return True
    return False


def load_excluded_ops():
    import csv
    import pathlib

    script_dir = pathlib.Path(__file__).parent.resolve()
    exclude_ops_csv_path = os.path.join(script_dir, "ops_stats_data/exclude_ops.csv")

    excludes = []

    with open(exclude_ops_csv_path, "r") as excluded_ops_csv:
        excluded_ops_reader = csv.DictReader(excluded_ops_csv, delimiter=",")
        for row in excluded_ops_reader:
            excludes.append(row["OP"])

    return excludes


def main(args):
    f_op_decl = open(args.ops_decl, "r")
    l_op_decl = f_op_decl.readlines()
    f_op_decl.close()
    p1 = os.path.join(args.gen_files_path, "lazy/wrap_kernels_registrations.cpp")
    p2 = os.path.join(args.gen_files_path, "backend")
    f_manual_ops_decl = open(p1, "r")
    l_manual_ops_decl = f_manual_ops_decl.readlines()
    f_manual_ops_decl.close()
    l_auto_ops_decl = combine_auto_generated_files(p2)

    exclude = load_excluded_ops()

    valid_op_decl = []
    valid_op_decl_compound_op = []
    valid_op_decl_prop = []
    j = v = m = 0
    pt_op_dict = {}
    unique_ops = set()
    for line in l_op_decl:
        if "aten::" in line:
            op_name = line.split("aten::")[1].split("(")[0]
            op_with_sig = line.split(");")[0]
            prop = {}
            valid_op_decl.append(op_name)
            if not any([x in op_name for x in exclude]):
                # if not match_any(line, exclude):
                prop["relevant"] = "yes"
            else:
                prop["relevant"] = "no"

            if '"dispatch": "True"' in line:
                # valid_op_decl_compound_op.append('compound')
                prop["type"] = "non-compound"
                if '"default": "True"' in line:
                    prop["type2"] = "dt_dt"
                else:
                    prop["type2"] = "dt_df"

            else:
                # valid_op_decl_compound_op.append('non-compound')
                prop["type"] = "compound"
                # prop["type2"] = "df_xx"
                if '"default": "True"' in line:
                    prop["type2"] = "df_dt"
                else:
                    prop["type2"] = "df_df"
            prop["op_with_sig"] = op_with_sig + ")"
            valid_op_decl_prop.append(prop)
            pt_op_dict[op_name] = prop
            unique_ops.add(get_unique_op_name(op_name))

    print("len valid_op_decl = ", len(valid_op_decl))
    # print(valid_op_decl)

    valid_manual_ops_decl = []
    manual_op_dict = {}
    for line in l_manual_ops_decl:
        if "m.impl(" in line and "hpu_wrap" in line:
            op_name = line.split('m.impl("')[1].split('",')[0]
            valid_manual_ops_decl.append(op_name)
            manual_op_dict[op_name] = "manual"

    # Get and append Manual ops with override in yaml
    yaml_file = os.path.join(args.pt_integ_path, "scripts/hpu_op.yaml")
    manual_ops_override_list = get_manual_ops_with_overrides_in_yaml(yaml_file)
    print("\n\nManual ops with override in yaml = ", manual_ops_override_list)
    for op_name in manual_ops_override_list:
        valid_manual_ops_decl.append(op_name)
        manual_op_dict[op_name] = "manual"
    print("len valid_manual_ops_decl = ", len(valid_manual_ops_decl))
    print(valid_manual_ops_decl)

    valid_auto_ops_decl = []
    auto_op_dict = {}
    for line in l_auto_ops_decl:
        if "REGISTER_HPU_BACKEND" in line:
            op_name = line.split('.REGISTER_HPU_BACKEND("')[1].split('",')[0]
            op_name = op_name.split("::")[1] if "aten::" in op_name or "hpu::" in op_name else op_name
            # Skip adding to "auto" ops list if op is registered as part of auto code,
            #  but is actually manual op overridden in yaml
            if is_op_overridden_in_yaml(op_name, manual_ops_override_list):
                continue
            valid_auto_ops_decl.append(op_name)
            auto_op_dict[op_name] = "auto"
    print("len valid_auto_ops_decl = ", len(valid_auto_ops_decl))
    print(valid_auto_ops_decl)
    impl_ops_list = []
    impl_ops_list.extend(valid_manual_ops_decl)
    impl_ops_list.extend(valid_auto_ops_decl)
    impl_op_dict = {}
    impl_op_dict.update(manual_op_dict)
    impl_op_dict.update(auto_op_dict)
    print("len  impl_ops_decl = ", len(impl_ops_list))
    print(impl_ops_list)

    print(impl_op_dict)
    op_names = list(pt_op_dict.keys())
    # for pop in pt_op_dict.keys():
    for pop in op_names:
        if pop in impl_op_dict.keys():
            pt_op_dict[pop].update({"impld": "yes"})
            pt_op_dict[pop].update({"impl_method": impl_op_dict[pop]})
        else:
            pt_op_dict[pop].update({"impld": "no"})
            pt_op_dict[pop].update({"impl_method": "NA"})

    # print("\n\n ptopdict = ",pt_op_dict)
    write_consolidated_op_list("consolidate_ops_list.csv", pt_op_dict)

    print(impl_ops_list)
    print("********************************")
    print("len  unique ops = ", len(unique_ops))
    print("********************************")
    print(unique_ops)
    unique_op_dict_v1 = unique_ops_stats_v1(unique_ops, pt_op_dict)
    unique_op_dict_v2 = unique_ops_stats_v2(unique_ops, pt_op_dict)
    print(unique_op_dict_v1)
    print(unique_op_dict_v2)

    write_unique_op_list_v1("unique_ops_list.csv", unique_op_dict_v1)
    write_unique_op_list_v2("unique_ops_list2.csv", unique_op_dict_v2)
    print("len valid_op_decl = ", len(valid_op_decl))
    print("len  impl_ops_decl = ", len(impl_ops_list))
    print("len valid_manual_ops_decl = ", len(valid_manual_ops_decl))
    print("len valid_auto_ops_decl = ", len(valid_auto_ops_decl))
    print("\nOps with following strings in their names are considered not relevant on HPU", exclude)

    csv_json_files = ["summary", "consolidate_ops_list", "unique_ops_list", "unique_ops_list2"]

    for f in csv_json_files:
        fc = f + ".csv"
        fj = f + ".json"
        df = pd.read_csv(fc, sep="|")
        t = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        json_list = json.loads(df.to_json(orient="records"))
        for item in json_list:
            item["timestamp"] = t
        with open(fj, "w+") as f:
            json.dump(json_list, f)


def parse_args_and_run_main(argv=None):
    # for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ops_decl", default="", help="ops declarations file")
    parser.add_argument("--pt_integ_path", default="", help="path of pytorch integration git")
    parser.add_argument(
        "--gen_files_path", default="", help="path of auto generated op files and wrap declarations file for manual ops"
    )
    args = parser.parse_args(argv)
    main(args)


if __name__ == "__main__":
    parse_args_and_run_main()
