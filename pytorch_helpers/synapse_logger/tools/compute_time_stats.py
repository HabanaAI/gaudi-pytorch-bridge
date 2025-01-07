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
from collections import namedtuple

import matplotlib.pyplot as plt
from gson_parsing import gson_iterator


class EventEntry:
    def __init__(self):
        self.begin_event = None
        self.end_event = None

    def elapsed_time(self):
        assert self.begin_event, "No begin event"
        assert self.end_event, "No end event"
        return self.end_event["ts"] - self.begin_event["ts"]

    def valid(self):
        return self.begin_event is not None and self.end_event is not None


def collect_func_times(input_file, func_names):
    func_times = {func_name: [] for func_name in func_names}
    try:
        for _, entry in gson_iterator(input_file):
            if entry["name"] in func_times:
                if entry["ph"] == "B":
                    ee = EventEntry()
                    ee.begin_event = entry
                    func_times[entry["name"]].append(ee)
                else:
                    assert entry["ph"] == "E", "unexpected value"  # sanity check
                    found_matching_event = False
                    func_times_for_name = func_times[entry["name"]]
                    for i in range(len(func_times_for_name) - 1, -1, -1):
                        if func_times_for_name[i].end_event:
                            continue  # recursively called func - look back to least recent entry w/o end event
                        if (
                            func_times_for_name[i].begin_event["tid"] == entry["tid"]
                            and func_times_for_name[i].begin_event["pid"] == entry["pid"]
                        ):
                            found_matching_event = True
                            func_times_for_name[i].end_event = entry
                            break

                    def parse(elem):
                        return elem.begin_event, (elem.end_event if elem.end_event else "None")

                    if not found_matching_event:
                        print(f"WARNING: Did not find matching begin event for {entry}.")
    except:
        pass
    return func_times


ResultStat = namedtuple("ResultStat", ["samples", "min", "max", "mean", "stddev", "median"])


def stat(population):
    import statistics

    s_min = min(population)
    s_max = max(population)
    s_mean = statistics.mean(population)
    s_median = statistics.median(population)
    s_stddev = statistics.stdev(population) if len(population) > 1 else 0

    return ResultStat(samples=population, min=s_min, max=s_max, mean=s_mean, stddev=s_stddev, median=s_median)


def show_us(x):
    if x is None:
        return "N/A"
    if type(x) in [float, int]:
        return f"{x:0.2f} us"
    return x


g_fname_offset = 35
g_other_offset = 15


def print_stats_header():
    print(
        f'{"Func name":<{g_fname_offset}}{"Samples":<{g_other_offset}}\
{"Mean":<{g_other_offset}}{"Min":<{g_other_offset}}{"Max":<{g_other_offset}}\
{"Stddev":<{g_other_offset}}{"Median":<{g_other_offset}}'
    )
    print("-" * (g_fname_offset + 6 * g_other_offset))


def print_stats_for_func(func_name, func_stats):
    print(
        f"{func_name:<{g_fname_offset}}{len(func_stats.samples):<{g_other_offset}}\
{show_us(func_stats.mean):<{g_other_offset}}{show_us(func_stats.min):<{g_other_offset}}\
{show_us(func_stats.max):<{g_other_offset}}{show_us(func_stats.stddev):<{g_other_offset}}\
{show_us(func_stats.median):<{g_other_offset}}"
    )


def generate_func_times(input_file, func_names):
    func_names = func_names.split(",")
    func_times = collect_func_times(input_file, func_names)
    func_stats = {
        func_name: stat([f.elapsed_time() for f in func_times[func_name] if f.valid()]) for func_name in func_names
    }
    return func_stats


def print_stats(func_stats, plot_hist=False, hist_bins=None):
    if plot_hist:
        fig, axs = plt.subplots(len(func_stats), 1)

    print("=" * (g_fname_offset + 6 * g_other_offset))
    print_stats_header()
    n = 0
    for key in func_stats:
        print_stats_for_func(key, func_stats[key])
        if plot_hist:
            ax_obj = axs[n] if len(func_stats) > 1 else axs
            ax_obj.hist(
                func_stats[key].samples,
                bins=hist_bins,
                edgecolor="k",
                range=(func_stats[key].min, func_stats[key].mean + func_stats[key].stddev),
            )
            ax_obj.set_title(key)
            n = n + 1
    print("=" * (g_fname_offset + 6 * g_other_offset))
    if plot_hist:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Statistics computation tool for SynapseLogger output."
        "Example command: python3 compute_time_stats.py --input_file logger_file.json --func_names empty_strided_hpu_lazy"
    )
    parser.add_argument(
        "--input_file",
        default=".local.synapse_log.json",
        help="input SynapseLogger file. May be either raw json or gzipped file.",
    )
    parser.add_argument(
        "--func_names",
        required=True,
        help="Comma-separated list of functions to get stats for.",
    )
    parser.add_argument(
        "--hist",
        action="store_true",
        help="Generate time execution histograms for given functions.",
    )
    parser.add_argument(
        "--bins",
        action="store",
        type=int,
        default=30,
        help="Specify number of bins for histograms generation.",
    )
    args = parser.parse_args()
    func_times = generate_func_times(args.input_file, args.func_names)
    print_stats(func_times, args.hist, args.bins)
