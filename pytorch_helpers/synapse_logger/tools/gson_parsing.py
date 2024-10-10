# ******************************************************************************
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
# ******************************************************************************

"""
utilities for parsing and transforming synapse logger json trace file.

Synapse logger uses json Trace Event Format that is compatible with chrome:tracing.
See https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#

gson_iterator offers basic iterator that can be further chained through
transformation functions.  As a convention each iterator produces 2-tuples
containing line number and dict entry from trace.
"""

import json
import logging
from collections import OrderedDict, namedtuple
from enum import Enum

log = logging.getLogger("synapse_logger.gson_parsing")


class syn_type:
    syn_type_na = 0  # // invalid
    syn_type_fixed = 2**0  # // 8-bit integer
    syn_type_int8 = syn_type_fixed  # // alias to syn_type_fixed
    syn_type_bf16 = 2**1  # // 16-bit float- 8 bits exponent 7 bits mantisa 1 bit sign
    syn_type_single = 2**2  # // 32-bit floating point
    syn_type_float = syn_type_single  # // alias to syn_type_single
    syn_type_int16 = 2**3  # // 16-bit integer
    syn_type_int32 = 2**4  # // 32-bit integer
    syn_type_uint8 = 2**5  # // 8-bit unsigned integer
    syn_type_int4 = 2**6  # // 4-bit signed integer
    syn_type_uint4 = 2**7  # // 4-bit unsigned integer
    syn_type_fp16 = 2**8  # // 16-bit floating point
    syn_type_uint16 = 2**9  # // 16-bit unsigned integer
    syn_type_uint32 = 2**10  # // 32-bit unsigned integer


syn_types = {
    syn_type.syn_type_na: ("syn_type_na    ", "na", 1),
    syn_type.syn_type_fixed: ("syn_type_fixed ", "int8_t", 1),
    syn_type.syn_type_int8: ("syn_type_int8  ", "int8_t", 1),
    syn_type.syn_type_bf16: ("syn_type_bf16  ", "bf16", 2),
    syn_type.syn_type_single: ("syn_type_single", "float", 4),
    syn_type.syn_type_float: ("syn_type_float ", "float", 4),
    syn_type.syn_type_int16: ("syn_type_int16 ", "int16_t", 2),
    syn_type.syn_type_int32: ("syn_type_int32 ", "int32_t", 4),
    syn_type.syn_type_uint8: ("syn_type_uint8 ", "uint8_t", 1),
    syn_type.syn_type_int4: ("syn_type_int4  ", "int4_t", 0.5),
    syn_type.syn_type_uint4: ("syn_type_uint4 ", "uint4_t", 0.5),
    syn_type.syn_type_fp16: ("syn_type_fp16  ", "fp16_t", 2),
    syn_type.syn_type_uint16: ("syn_type_uint16", "uint16_t", 2),
    syn_type.syn_type_uint32: ("syn_type_uint32", "uint32_t", 4),
}

hcl_collective_ops = {
    0: "eHCLReduce",
    1: "eHCLAllReduce",
    2: "eHCLReduceScatter",
    3: "eHCLAll2All",
    4: "eHCLBroadcast",
    5: "eHCLAllGather",
    6: "eHCLAll2AllV",
}


class synDeviceType(Enum):
    synDeviceGoya = 0
    synDeviceGoya2 = synDeviceGreco = 1
    synDeviceGaudi = 2
    synDeviceGaudi2 = 4
    synDeviceGaudi3 = 5
    synDeviceEmulator = 6
    synDeviceTypeInvalid = 7
    synDeviceTypeSize = 8


hcl_ops = {0: "eHCLOpNone", 1: "eHCLSum", 2: "eHCLProd"}

FuncDef = namedtuple("FuncDef", ["name", "args", "return_type"])


def func_def_from_pretty_function(pfunction, args):
    type_and_name, args_types = pfunction.split("(")
    return_type, name = type_and_name.split(" ")
    args_types = OrderedDict(zip(args, args_types[:-1].split(", ")))
    return FuncDef(name, args_types, return_type)


def replace_with_dur(matched_sequence):
    for num, e in matched_sequence:
        if "ph" in e:
            if e["ph"] == "E":
                continue
            if e["ph"] == "B":
                if not "end_ts" in e:
                    continue
                e["ph"] = "X"
                e["dur"] = e["end_ts"] - e["ts"]
        yield e


def filter_nested(entries):
    tid_name = dict()
    no = 0
    for _, entry in entries:
        if entry["name"][:4] == "call":
            if entry["ph"] == "B":
                if entry["tid"] in tid_name:
                    continue
                else:
                    tid_name[entry["tid"]] = entry["name"]

            if entry["ph"] == "E":
                if entry["tid"] in tid_name and tid_name[entry["tid"]] == entry["name"]:
                    del tid_name[entry["tid"]]
                else:
                    continue

        no = no + 1
        yield (no, entry)


def match_call_results(entries):
    """
    matches begin/end traces of call:<some_func>.

    Since in the input sequence the beginnings and ends of calls may be
    separated by arbitrary amount of other entries from many threads, this
    generator keeps an internal stack of yet unmatched entries.

    in the output sequence begins of calls have additional keys 'result' and 'end_ts'. Such sequence can be easily transformed into

    Parameters:
        entries (iterable): sequence of log entries
    """
    stack = []
    backlog = []
    for no, entry in entries:
        try:
            if entry["name"][:4] == "call" and entry["ph"] == "E":
                func_name = entry["name"][5:]
                for sno, s in reversed(stack):
                    if s["name"][:4] == "call" and s["ph"] == "B" and s["tid"] == entry["tid"] and "result" not in s:
                        assert (
                            s["func"].name == func_name
                        ), f"at line {no}: function call begin/end mismatch - expected {s['func'].name} got {func_name}"
                        log.debug(f"at line {no} matched result of call to {s['func'].name} from line {sno}")
                        s["result"] = entry.get("args", None)
                        s["end_ts"] = entry["ts"]
                        entry["name"] = func_name  # "call"
                        break
                else:
                    assert False, f"at line {no}: got result of a function that wasn't called"
            log.debug(f"stack appending line {no}")
            stack.append((no, entry))
            while stack and (stack[0][1]["name"][:4] != "call" or stack[0][1]["ph"] == "E" or "result" in stack[0][1]):
                # pop stack until we reach incomplete call (i.e. without result matched)
                result = stack.pop(0)
                log.debug(
                    f"matcher yielding {result[1]['name'][:4] != 'call' or 'result' in result[1]}, {bool(stack)}, {result}"
                )
                yield result
        except Exception as e:
            log.error("call-result matching stack during exception:\n" + "\n".join(f"\t{no} : {s}" for no, s in stack))
            log.error(e)
            raise
    while stack:
        no, p = stack.pop(0)
        if p["name"][:4] == "call" and p["ph"] == "B":
            log.warn(f"line {no} incomplete call to {p['func'].name}")
            p["result"] = {"status": "incomplete"}
        yield no, p


def process_call(entry):
    if entry["name"][:4] == "call":
        func_name = entry["name"][5:]
        if entry["ph"] == "B":
            entry["name"] = "call"
            entry["func"] = func_def_from_pretty_function(entry["func"], entry["args"].keys())
    return entry


def printer(entry):
    print(entry)
    return entry


def gson_iterator(gson_file_name, end_on_error=True):
    """
    Basic synapse logger json iterator offering line-by-line loading ang support of gzipped traces.
    """
    if gson_file_name[-3:] == ".gz":
        import gzip

        source = gzip.open  # (gson_file_name, mode="r")
    else:
        source = open  # (gson_file_name)
    with source(gson_file_name, mode="r") as gson_file:

        text_lines = (text_line for text_line in enumerate(gson_file))
        next(text_lines)  # skip '[\n'

        def parse_json(iterator):
            for no, text_line in iterator:
                try:
                    json_line = json.loads(text_line[:-2])
                    yield no, json_line
                except json.decoder.JSONDecodeError as e:
                    log.warn(f"Json decode error on line {no}: {e}, end_on_error is {end_on_error}")
                    log.warn(f"offending line '{text_line[:-2]}'")
                    log.warn(f"near '{text_line[e.pos-10:e.pos+10]}'")
                    if end_on_error:
                        return
                    else:
                        raise

        json_lines = parse_json(text_lines)

        json_lines = filter_nested(json_lines)
        json_lines = ((no, process_call(entry)) for no, entry in json_lines)
        json_lines = match_call_results(json_lines)

        for no, entry in json_lines:
            try:
                args = entry.get("args", {})
                if entry["name"][:4] == "call":
                    if entry["ph"] == "B":
                        if "deviceId" in args:
                            args["deviceId"] = "device_id"
                        if "deviceType" in args:
                            args["deviceType"] = f'synDeviceType::{synDeviceType(int(args["deviceType"])).name}'

                        for arg, arg_type in entry["func"].args.items():
                            if arg_type in ("const char*", "const char *") and args[arg] != "nullptr":
                                args[arg] = f'"{args[arg]}"'
                            if arg_type in ("const synRecipeInfo*", "const synRecipeInfo *"):
                                args[arg] = tuple(args[arg])
                            if arg == "size":
                                val = args[arg]
                                if type(val) is list:
                                    args[arg] = list(map(lambda x: int(x, 16), val))
                                else:
                                    # if isinstance(args[arg], str) and args[arg][:2]=='0x':
                                    args[arg] = int(val, 16)
                yield no, entry
            except Exception as e:
                log.error(f"Error when processing entry line {no} (or next?)\n{entry}")
                if end_on_error:
                    log.warn(str(e))
                    return
                raise


def zip_launch_info(entry):
    launch = entry["args"]
    tensors = ((name, addr) for name, addr in zip(launch["launchTensorsInfo"][::2], launch["launchTensorsInfo"][1::2]))
    return tensors


def descriptor_byte_size(descriptor):
    size = 1
    for dno in range(descriptor["fields"]["m_dims"]):
        size *= descriptor["fields"]["m_sizes"][dno]
    return size * syn_types[descriptor["fields"]["m_dataType"]][2]
