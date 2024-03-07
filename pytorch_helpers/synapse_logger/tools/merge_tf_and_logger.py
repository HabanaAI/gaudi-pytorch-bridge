#!/usr/bin/env python
# ******************************************************************************
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
# ******************************************************************************
import argparse
import collections
import json
import logging
import sys

import gson_parsing

HOROVOD_PID_OFFSET = 10000
SYNAPSE_PID = 1


def merge_profiler_logs(gson_file, hvd_file, tf_files, map_tid=True):
    events, time_range = load_events_for_merge(gson_file, hvd_file, tf_files)
    if tf_files and map_tid:
        tid_mapping = logs_figure_tid_mapping(events)
        print("tid mapping", tid_mapping)
    beautify(events)
    if tf_files and map_tid:
        apply_mapping(events, tid_mapping)
    print("after mapping", len(events), "events")
    write_events(events)


def beautify(events):
    for e in events:
        if "cat" in e and "args" in e and "name" in e["args"] and "name" in e and e["name"] == "unknown":
            n = e["args"]["name"][e["args"]["name"].find(":") + 1 :]
            e["name"] = n
            e["args"]["op"] = n


def filter_time(events, min_time, max_time):
    dropped = 0
    for e in events:
        starte = e.get("ts", min_time)
        ende = e.get("ts", max_time)  # + e.get("dur", 1)
        if starte >= min_time and ende <= max_time:
            yield e
        else:
            assert "name" in e, f"no name {e}"
            if e["name"][:3] in ("syn", "obj"):
                pass
                # print(".", end="")
            else:
                pass
                # print("rm", e['name'], starte - min_time, ende - max_time, end = " ")
                # sys.exit()
            dropped += 1
    print(f"filter time dropped {dropped} events")


def write_events(events, out_file_name="merged.json.gz"):
    if out_file_name.endswith(".gz"):
        import gzip

        opener = gzip.open, "wt"
    else:
        opener = open, "w"
    with opener[0](out_file_name, opener[1]) as out_fd:
        print('{"traceEvents": ', file=out_fd, end="")
        json.dump(events, out_fd, indent=1)
        print("}", file=out_fd)


def load_events_for_merge(gson_file, hvd_file, tf_files):
    events = list()
    events.extend(load_tf_events(*tf_files) if tf_files else [])
    events.extend(load_hvd_events(hvd_file) if hvd_file else [])
    # Get range from both HVD and TF
    event_time_range = find_events_time_span(events)
    events.extend(load_synapse_logger_events(gson_file, event_time_range) if gson_file else [])
    events.sort(key=lambda x: x.get("ts", 0))  # + x.get("dur", 0))
    return events, event_time_range


def fix_gson(gson_lines):
    for e in gson_lines:
        try:
            if e["name"] == "object":
                continue
            r = {k: e[k] for k in ("ts", "ph", "tid", "name")}
            if "dur" in e:
                r["dur"] = e["dur"]
            r["args"] = e.get("args", {})
            if "func" in e:
                r["name"] = e["func"].name
            r["pid"] = SYNAPSE_PID
            yield r
        except:
            print(f"error when processing entry {e}")


def load_tf_events(*tf_files):
    events = list()
    tf_start = 10e100
    tf_stop = 0

    for tf_file in tf_files:
        tf = json.load(open(tf_file))["traceEvents"]
        time_start, time_stop = find_events_time_span(tf)
        tf_start, tf_stop = min(tf_start, time_start), max(tf_stop, time_stop)
        events += tf

        print(f"added {len(tf)} ranging {time_start}..{time_stop} from {tf_file}, events range {tf_start}..{tf_stop}.")
    return events


def load_hvd_events(*hvd_files):
    events = list()
    hvd_start = 10e100
    hvd_stop = 0

    for hvd_file in hvd_files:
        lines = open(hvd_file).readlines()
        lines = list(map(lambda x: x.strip(), lines))
        # Horovod does not close list at the and of timeline file
        # as chrome:\\tracing does not require it. Python json
        # parser however does.
        # We need to replace comma in last line to closing bracket.
        if lines[-1][-1] == ",":
            lines[-1] = lines[-1][:-1] + "]"
        hvd_events = json.loads("".join(lines))
        for event in hvd_events:
            if "pid" in event:
                event["pid"] = event["pid"] + HOROVOD_PID_OFFSET
        time_start, time_stop = find_events_time_span(hvd_events)
        hvd_start, hvd_stop = min(hvd_start, time_start), max(hvd_stop, time_stop)
        events += hvd_events

        print(
            f"added {len(hvd_events)} ranging {time_start}..{time_stop} from {hvd_file}, events range {hvd_start}..{hvd_stop}."
        )
    return events


def logs_figure_tid_mapping(events):

    tf_tids, syn_tids = set(), dict()
    pomap = dict()
    TF, SYN = "tf", "syn"
    threads = dict()  # keep last processed operation on each thread

    def add_tid(new_tid):
        # when new syn tid comes it can be mapped to any tf thread that is not yet mapped and vice versa
        if new_tid < 20:
            if not new_tid in tf_tids:
                print("new tid: " + str(new_tid))
                pomap.update(
                    {(stid, new_tid): True for stid in syn_tids.keys()}
                )  # mapping of any known syn tid to this new tf tid is possible
                tf_tids.add(new_tid)
                threads[new_tid] = {"ts": 0, "dur": 0}
            return TF
        else:
            if not new_tid in syn_tids:
                print("new stid: " + str(new_tid))
                pomap.update({(new_tid, ttid): True for ttid in tf_tids})
                syn_tids[new_tid] = 0
                threads[new_tid] = {"ts": 0, "dur": 0}
            return SYN

    def update_thread(tid, e, who):
        if "dur" in e and "name" in e:
            if who == SYN and e["name"] != "synEventSynchronize":
                syn_tids[tid] += 1
            threads[tid] = e

    for e in events:
        if "tid" not in e:
            continue
        tid = e["tid"]

        who = add_tid(tid)

        if who == TF and e["pid"] == 1:
            # disallow mapping to any synapse thread that is currently in the middle of a call
            for stid in syn_tids.keys():
                s = threads[stid]

                s_end = s["ts"] + s["dur"]
                e_end = e["ts"] + e.get("dur", 0)
                begin_overlaps = s["ts"] < e["ts"] < s_end
                end_overlaps = s["ts"] < e_end < s_end

                if begin_overlaps or end_overlaps:
                    if pomap[(stid, tid)]:
                        if 0:
                            print(
                                "Disallow ",
                                tid,
                                stid,
                                f"because tf begin {begin_overlaps} end {end_overlaps} \n     ",
                                e,
                                "\n  overlaps\n    ",
                                s["ts"],
                                s["dur"],
                                s_end,
                                s["name"],
                            )
                        pomap[(stid, tid)] = False

        update_thread(tid, e, who)
    for ttid in tf_tids:
        print("mappins for ", ttid, " are ", list(stid for stid in syn_tids.keys() if pomap[(stid, ttid)]))
    unmapped = tf_tids.copy()
    mapping = dict()
    while unmapped:
        # find stid with least options
        opts = list()
        for ttid in unmapped:
            sopts = [(stid, worth) for stid, worth in syn_tids.items() if pomap[(stid, ttid)]]
            opts.append((ttid, sopts))
        opts.sort(key=lambda x: (len(x[1]), min([obj[1] for obj in x[1]], default=0)))
        ms_tid, ms_opts = opts[0]
        if len(ms_opts) < 5 and len(ms_opts) > 0:
            ms_opts.sort(key=lambda x: x[1], reverse=True)
            mt_tid = ms_opts[0][0]
            print(f"tf thread with least options is {ms_tid} ({ms_opts} options), try mapping with {mt_tid}")
            for ttid in tf_tids:
                pomap[(mt_tid, ttid)] = False
            mapping[ms_tid] = mt_tid
        else:
            print(f"tf thread with least options is {ms_tid} ({ms_opts} options), so not mapping to anything")
            mapping[ms_tid] = ms_tid
        unmapped.remove(ms_tid)
    return mapping


def apply_mapping(events, mapping):
    for e in events:
        if not "tid" in e:
            continue
        new_tid = mapping.get(e["tid"], None)
        if new_tid is not None:
            e["tid"] = new_tid
            e["cat"] = "synapse"


def find_events_time_span(events):
    result = min(x.get("ts", 10e100) for x in events), max(y.get("ts", 0) + y.get("dur", 0) for y in events)
    return result


def load_synapse_logger_events(gson_file, event_time_range):
    # Name the synapse log
    events = [({"name": "process_name", "ph": "M", "pid": SYNAPSE_PID, "args": {"name": "Synapse logger"}})]
    gson_lines = gson_parsing.gson_iterator(gson_file)
    gson_lines = gson_parsing.replace_with_dur(gson_lines)
    gson_lines = fix_gson(gson_lines)
    gson_lines = list(gson_lines)
    gson_lines = [
        event
        for event in gson_lines
        if (
            event["ts"] >= event_time_range[0]
            and ((event["ts"] + event["dur"] if ("dur" in event) else event["ts"]) <= event_time_range[1])
        )
    ]
    gson_start, gson_stop = find_events_time_span(gson_lines)
    print(f"added {len(gson_lines)} ranging {gson_start}..{gson_stop} from {gson_file}.")
    events.extend(gson_lines)
    return events


if __name__ == "__main__":

    def main():
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("-s", "--synapse_file", default=None, type=str, help="Synapse log file to merge.")
        parser.add_argument("-d", "--hvd_file", default=None, type=str, help="Horovod timeline file to merge.")
        parser.add_argument(
            "-t", "--tf_files", default=None, type=str, nargs="*", help="Tensorflow timeline files to merge."
        )
        parser.add_argument("-v", "--verbose", action="store_true", help="Produce diagnostic output")
        args = parser.parse_args()
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARN)
        merge_profiler_logs(args.synapse_file, args.hvd_file, args.tf_files)

    main()
