###############################################################################
#
#  Copyright (c) 2021-2025 Intel Corporation
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

# Refer Parser Tool Confluence Page: https://confluence.habana-labs.com/display/Frameworks/Dynamic+shapes+Compilation+Stats+Parser

import argparse
import csv
import json
import os


def compileInfo(path):
    recipe_dict = {}
    total_iter_count = 0
    total_static_comp = 0
    total_dyn_comp = 0
    total_static_hit = 0
    total_dyn_hit = 0

    print("\nGraph Info:")
    for file_name in os.listdir(path):
        if file_name.endswith(".json"):
            f = open(path + "/" + file_name)
            # data is loaded as list of dicts
            data_list = json.load(f)
            iter_count = 0
            static_comp = 0
            dyn_comp = 0
            static_hit = 0
            dyn_hit = 0
            for data in data_list:
                for iter in data:
                    iter_count += 1
                    for info in data[iter]:
                        if info == "compilations":
                            for i in data[iter][info]:
                                if i["scope"] == "STATIC":
                                    static_comp += 1
                                    recipe_dict[i["recipe"]] = "STATIC"
                                else:
                                    dyn_comp += 1
                                    recipe_dict[i["recipe"]] = "DYNAMIC"
                            break
                        if info == "selected recipe":
                            if recipe_dict[data[iter][info]] == "STATIC":
                                static_hit += 1
                            else:
                                dyn_hit += 1
            total_iter_count += iter_count
            total_static_comp += static_comp
            total_dyn_comp += dyn_comp
            total_static_hit += static_hit
            total_dyn_hit += dyn_hit
            print(
                file_name,
                " :: total_iterations = ",
                iter_count,
                " , static_compilations = ",
                static_comp,
                " , static_hits = ",
                static_hit,
                " , dynamic_compilations = ",
                dyn_comp,
                " , dynamic_hits = ",
                dyn_hit,
            )
    print("\nTotal Model Stats:")
    print(
        "total_iterations = ",
        total_iter_count,
        " , static_compilations = ",
        total_static_comp,
        " , static_hits = ",
        total_static_hit,
        " , dynamic_compilations = ",
        total_dyn_comp,
        " , dynamic_hits = ",
        total_dyn_hit,
    )


def printJit(path, file_name):
    if file_name == "":
        print("\nERROR: Please provide file_name with --name")
        return 0
    f = open(path + "/" + file_name)
    # data is loaded as list of dicts
    data_list = json.load(f)
    print("JIT_IR Begin :")
    Jit_ir = data_list[0]["000000000"]["compilations"][0]["jit ir graph"]
    for line in Jit_ir:
        print(line)
    print("JIT_IR End :")


graph_dict = {}


def statsParser(path, file_name):
    # print(file_name)
    f = open(path + "/" + file_name)
    # data is loaded as list of dicts
    data_list = json.load(f)
    print_freq = 200
    miss_percent = []
    miss_cnt = 0
    miss_freq = 0
    total_cnt = 0
    data_dict = {}
    for data in data_list:
        for k1 in data:
            total_cnt += 1.0
            if total_cnt % print_freq == 0:
                miss_percent.append(100 * miss_freq / print_freq)
                miss_freq = 0
            cache_hit = True
            bucket_id = 0
            for k2 in data[k1]:
                if k2 == "compilations":
                    cache_hit = False
                    miss_cnt += 1.0
                    miss_freq += 1.0
                if k2 == "selected bucket":
                    for k3 in data[k1][k2]:
                        if k3 == "id":
                            bucket_id = data[k1][k2]["id"]
            data_dict[k1] = [cache_hit, bucket_id]
    if miss_freq != 0:
        miss_percent.append(100 * miss_freq / (total_cnt % print_freq))
    graph_dict[file_name] = [data_dict, total_cnt, 100 * miss_cnt / total_cnt, miss_percent]


def printStats(file_name, miss_threshold, max_graphs):
    if graph_dict[file_name][2] > miss_threshold and graph_dict[file_name][1] > max_graphs:
        print(
            file_name,
            " ; total_cnt ; ",
            graph_dict[file_name][1],
            " ; miss percentage ; ",
            graph_dict[file_name][2],
            " ; Trend ; ",
            graph_dict[file_name][3],
        )


def parseAllJson(path, miss_threshold, max_graphs):
    for file in os.listdir(path):
        if file.endswith(".json"):
            statsParser(path, file)
            printStats(file, miss_threshold, max_graphs)
    return graph_dict


def parse1Json(file_name):
    statsParser(file_name)
    return graph_dict


def strToList(val):
    val = val.strip()
    if val == "[]":
        return 0
    if val[-1] == "r":
        val = val[0:-13]
    val = val[1:-1]
    val = val.split(",")
    for j in range(len(val)):
        val[j] = int(val[j])
    return val


def check_list_min(list1, list2):
    any(list1[i] < list2[i] for i in range(len(list1)))


def check_list_max(list1, list2):
    any(list1[i] > list2[i] for i in range(len(list1)))


def reasonMismatch(range_dict, shapes, bucket, compile):
    Hit = False
    for r in range(1, bucket):
        match = True
        for keys in range_dict[r]:
            min = range_dict[r][keys][0]
            max = range_dict[r][keys][1]
            val = shapes[keys]

            # code below can be enabled if we need to skip some keys
            # from bucket match decision
            # if keys in [0, 1, 11, 12, 5, 6, 8, 9]:
            #  continue

            if check_list_min(val, min):
                print(f"Bucket {r} Failed in MinShape {keys} - Input={val} -> Range-[ min={min} - max={max} ]")
                match = False
            if check_list_max(val, max):
                print(f"Bucket {r} Failed in MaxShape {keys} - Input={val} -> Range-[ min={min} - max={max} ]")
                match = False
        if match:
            print(f"xxxxxxxxxxxxxxxxxxxx    Bucket found = {r} xxxxxxxxxxxxxxxxxxx")
            print("xxxxxxxxxxxxxxxxxxxx    Cache Hit     xxxxxxxxxxxxxxxxxxx")
            Hit = True
            assert compile is not True
            break

    if Hit is False:
        print("xxxxxxxxxxxxxxxxxxxx     Cache Miss     xxxxxxxxxxxxxxxxxxx")


def analyzeBucket(path, file_name, bucket_analyze):
    f = open(path + "/" + file_name)
    # data is loaded as list of dicts
    data_list = json.load(f)
    range_dict = {}
    bucket = 0
    # process each dicts
    for data in data_list:
        # key of the data dict 0000,0001...
        for k1 in data:
            # key of operation compilation, selected_bucket ...
            compile = False
            for k2 in data[k1]:
                if k2 == "compilations":
                    ranges = data[k1][k2][0]["ranges"]
                    compile = True
                    if ranges:
                        for k3 in ranges:
                            min, max = ranges[k3].split("-")
                            min = strToList(min)
                            max = strToList(max)
                            my_list = []
                            my_list.append(min)
                            my_list.append(max)
                            ranges[k3] = my_list
                        range_dict[bucket] = ranges
                        bucket = bucket + 1
                if k2 == "shapes":
                    shapes = data[k1][k2]
                    for k3 in shapes:
                        shapes[k3] = strToList(shapes[k3])
                    if compile and bucket == bucket_analyze:
                        reasonMismatch(range_dict, shapes, bucket - 1, compile)


def analyzeBucketCall(path, file_name):
    if file_name == "":
        print("\nERROR: Please provide file_name with --name")
        return 0
    f = open(path + "/" + file_name)
    # data is loaded as list of dicts
    data_list = json.load(f)
    recipe_bucket_map = {}
    bucket_hit_count = {}
    # process each dicts
    for data in data_list:
        for iter in data:
            compile_flag = False
            for info in data[iter]:
                if info == "compilations":
                    compile_flag = True
                    recipe_temp = data[iter][info][0]["recipe"]
                if info == "selected bucket" and compile_flag:
                    recipe_bucket_map[recipe_temp] = data[iter][info]["id"]
                    bucket_hit_count[data[iter][info]["id"]] = 0
                    compile_flag = False
                if info == "selected recipe":
                    bucket_hit_count[recipe_bucket_map[data[iter][info]]] += 1
    print("\nBucket-Iteration Calls:")
    for k, v in bucket_hit_count.items():
        print(k, " : ", v)


def dumpShapes(path, file_name):
    if file_name == "":
        print("\nERROR: Please provide file_name with --name")
        return 0
    f = open(path + "/" + file_name)
    # data is loaded as list of dicts
    data_list = json.load(f)
    shape_dict = {}

    for data in data_list:
        for iter in data:
            for info in data[iter]:
                if info == "shapes":
                    shapes = data[iter][info]
                    for sid in shapes:
                        if sid not in shape_dict:
                            shape_dict[sid] = []
                        shape_dict[sid].append(strToList(shapes[sid]))
    header = []
    for k, v in shape_dict.items():
        if not isinstance(v[0], int):
            for i in range(len(v[0])):  # Taking only first element to see dim
                header.append(k + "_DIM_" + str(i))
        else:  # Empty Tensors
            header.append(k + "_DIM_0")
    max_range = max([len(i) for i in shape_dict.values()])

    with open(file_name.split(".")[0] + ".csv", "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        for i in range(max_range):
            temp = []
            for sid, shape_dict_sid in shape_dict.items():
                if i < len(shape_dict_sid) and not isinstance(shape_dict_sid[i], int):
                    for item in shape_dict_sid[i]:
                        temp.append(item)
                elif i < len(shape_dict_sid) and isinstance(shape_dict_sid[i], int):
                    temp.append("")
                elif i >= len(shape_dict_sid):
                    if isinstance(shape_dict_sid[0], int):
                        temp.append("")
                    else:
                        for _ in shape_dict_sid[0]:
                            temp.append("")
                else:
                    print("Exception: ", sid)  # DEBUG
            writer.writerow(temp)
    print("\nCSV of shape dump saved: {}/{}.csv".format(os.getcwd(), file_name.split(".")[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse the Json Compilation Stats and print Cache Hit/Miss Information for Graph/Bucket.Please refer the PARSER TOOL CONFLUENCE PAGE for more info on how to use: https://confluence.habana-labs.com/display/Frameworks/Dynamic+shapes+Compilation+Stats+Parser"
    )
    parser.add_argument(
        "--path",
        dest="path",
        type=str,
        help="Path of the folder containing json dumps to analyse all static and dynamic cache hits and recompiles",
    )
    parser.add_argument(
        "--name", dest="name", default="", type=str, help="Name of the json to analyze buckets cache hit/miss"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Analyze all graphs in --path folder and print overall cache hit/miss along with miss trend per 100 iterations",
    )
    parser.add_argument("--jit", action="store_true", help="Print Jit for a particular Graph. Use with --name option")
    parser.add_argument(
        "--dump_shape",
        action="store_true",
        help="Dump shapes for a particular graph which can be imported in xls. use --name flag to specify the graph",
    )
    parser.add_argument(
        "--thresh",
        dest="miss_threshold",
        type=float,
        default=0.0,
        help="[Dev Util] Threshold of the miss above which to print",
    )
    parser.add_argument(
        "--graphs",
        dest="max_graphs",
        type=float,
        default=0.0,
        help="[Dev Util] Min number of graph to comnsider for analysis",
    )
    parser.add_argument(
        "--bucket",
        dest="bucket",
        type=int,
        default=-1,
        help="[Dev Util] Bucket to analyze for cache miss. Use along with --name flag",
    )

    args = parser.parse_args()
    if args.dump_shape:
        dumpShapes(args.path, args.name)
    elif args.bucket >= 0:
        analyzeBucket(args.path, args.name, args.bucket)
    elif args.jit:
        printJit(args.path, args.name)
    elif args.stats:
        out = parseAllJson(args.path, args.miss_threshold, args.max_graphs)
    elif args.name and args.path:
        analyzeBucketCall(args.path, args.name)
    elif args.path:
        compileInfo(args.path)
    else:
        print("Undefined flag combination. Please check --help")
