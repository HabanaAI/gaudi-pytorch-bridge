#!/usr/bin/env python

import argparse
import json


def list_ops_with_send_size(json_path, json_out_path):
    with open(json_path) as json_file:
        timeline = json.load(json_file)["traceEvents"]
        send_and_recv = [obj for obj in timeline if (obj["name"] == "_Recv" or obj["name"] == "_Send")]
        mem_cp = [
            obj
            for obj in timeline
            if obj["name"] == "synMemCopyAsync" and obj["ts"] > min([x["ts"] for x in send_and_recv])
        ]
        mem_per_obj = [
            {
                "name": next(
                    (
                        x
                        for x in send_and_recv
                        if x["ts"] < memEvent["ts"] < (x["ts"] + x["dur"])
                        and memEvent["pid"] == x["pid"]
                        and memEvent["tid"] == x["tid"]
                    ),
                    None,
                )["args"]["name"],
                "size": memEvent["args"]["size"],
                "direction": memEvent["args"]["direction"],
            }
            for memEvent in mem_cp
        ]

        with open(json_out_path, "w") as json_out_file:
            json.dump(mem_per_obj, json_out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate memory size")
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    list_ops_with_send_size(args.input, args.output)
