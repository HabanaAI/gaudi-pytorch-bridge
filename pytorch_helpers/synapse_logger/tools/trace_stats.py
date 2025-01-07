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
import json
import os
import re
import sys


def get_fw_time(path, tid, start_time, end_time):
    try:
        file1 = open(path, "r")
        host_data = json.load(file1)
    except ValueError:
        print("json parse has failed may be due to missing ']' ")
        exit()
    t_start = host_data[0]["ts"]
    fw_time = 0
    event_name = ""
    i = 0
    t_end = 0
    time_stamp = t_start * 1000 + start_time
    start_time = int(time_stamp / 1000)
    time_stamp = t_start * 1000 + end_time
    end_time = int(time_stamp / 1000)
    start_index = 0
    end_index = 0
    # get the start of the entry
    for i in range(len(host_data)):
        event = host_data[i]
        if start_index == 0 and event["tid"] == tid and event["ts"] >= start_time:
            event_name = event["name"]
            print("\nStart node: ", event_name)
            start_index = i

        if end_index == 0 and event["tid"] == tid and event["ts"] >= end_time:
            print("End node: ", event["name"], "\n")
            end_index = i
            break

    if start_index == 0:
        print("\ntid, start_time or end_time could be out of range")
        return

    t_end = start_time
    i = start_index
    ops_count = 0
    while i < end_index:
        ops_count += 1
        while True:
            if host_data[i]["name"] == event_name and ("ph" in host_data[i] and host_data[i]["ph"] == "E"):
                break
            i += 1
        t_end = host_data[i]["ts"]
        i += 1
        event_name = host_data[i]["name"]
        t_begin = host_data[i]["ts"]
        assert t_begin >= t_end, "ts_begin should be greater than ts_end"
        fw_time += t_begin - t_end

    total_time = end_time - start_time
    print("Total ops: ", ops_count)
    if fw_time > 1000:
        print("Total Time: ", total_time / 1000, "ms")
        print("FW Time: ", fw_time / 1000, "ms")
    else:
        print("Total Time: ", total_time, "us")
        print("FW Time: ", fw_time, "us")

    print("FW time / Total time: ", fw_time / (end_time - start_time))


def get_time(time_sec):
    time_list = time_sec.split()
    time_ = 0
    for ts in time_list:
        match = re.match(r"([0-9]+)([a-z]+)", ts, re.I)
        items = match.groups()
        t = int(items[0])
        if items[1] == "m":
            # minute
            time_ += t * 60 * 1000000000
        elif items[1] == "s":
            # second
            time_ += t * 1000000000
        elif items[1] == "ms":
            # milsecond
            time_ += t * 1000000
        elif items[1] == "us":
            # microsecond
            time_ += t * 1000
        elif items[1] == "ns":
            # nanosecond
            time_ += t
    return time_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate the framework host time.")
    parser.add_argument("--path", dest="file_path", type=str, help="file path")
    parser.add_argument("--tid", dest="tid", type=int, help="tid")
    parser.add_argument("--start-time", dest="start_time", type=str, help="ts begin")
    parser.add_argument("--end-time", dest="end_time", type=str, help="ts end")

    args = parser.parse_args()
    if args.file_path is None or args.tid is None or args.start_time is None or args.end_time is None:
        parser.print_help()
        exit(1)

    start_time = get_time(args.start_time)
    end_time = get_time(args.end_time)

    print("===================================================")
    print("||      Summary of FW and Bridge Host Time       ||")
    print("===================================================")
    get_fw_time(args.file_path, args.tid, start_time, end_time)
