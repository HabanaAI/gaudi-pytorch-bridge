#!/usr/bin/env python
# ******************************************************************************
# Copyright (C) 2020 HabanaLabs, Ltd.
# All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
#
# ******************************************************************************
import json
from collections import OrderedDict, defaultdict, Mapping
import os
import sys
import logging
from enum import Enum
from shutil import copy2
import argparse
from gson_parsing import func_def_from_pretty_function, gson_iterator, syn_types, hcl_collective_ops, hcl_ops
from io import StringIO

log = logging.getLogger("synapse_logger.gson2test")


class synDmaDir(Enum):
    HOST_TO_DRAM = 0
    DRAM_TO_HOST = 1
    DRAM_TO_DRAM = 2


synMemFlags = {0x0: 0, 0x1: "synMemHost", 0x2: "synMemDevice"}


class TransposePermutationDim:
    # // Assume basic permutation is BDHWC
    TBD_4DimSize = 4
    TBD_5DimSize = 5

    TPD_Batch = 4
    TPD_4Dim_Batch = 3
    TPD_Depth = 3
    TPD_Height = 2
    TPD_Width = 1
    TPD_Channel = 0

    # // Assume basic permutation is QRSCK
    TPD_Weights_Q = 4
    TPD_Weights_R = 3
    TPD_Weights_S = 2
    TPD_Weights_C = 1
    TPD_Weights_K = 0

    @staticmethod
    def from_int(v):
        f = {v: k for k, v in TransposePermutationDim.__dict__.items() if k.startswith("T")}
        return f[v]


def descriptor_byte_size(descriptor):
    size = 1
    for dno in range(descriptor["fields"]["m_dims"]):
        size *= descriptor["fields"]["m_sizes"][dno]
    return size * syn_types[descriptor["fields"]["m_dataType"]][2]


class Flow:
    def __init__(self, input_iterator=".local.synapse_log.json"):
        self.objs = dict()
        self.devmem_objs = defaultdict(list)
        self.functions = defaultdict()
        self.bin_file_size = 0
        if isinstance(input_iterator, str):
            input_iterator = gson_iterator(input_iterator)
        self.log = list(input_iterator)
        log.info("loaded log lenght %i", len(self.log))
        try:
            for line, entry in self.log:
                args = entry.get("args", {})
                if entry["name"][:4] == "call":
                    pass
                elif entry["name"] == "object":
                    self.objs[args["at"]] = entry
                    new_max_offset = args["data_offset"] + args["byte_size"] if "data_offset" in args else 0
                    self.bin_file_size = max(self.bin_file_size, new_max_offset)
                elif entry["name"] in ("reference", "event"):
                    pass
        except Exception as e:
            log.error(f"Error when processing entry line {line}\n{entry}")
            raise
        self.var_idx = 0
        log.info("done preprocessing")

    @staticmethod
    def find(where, what, skip=0, reverse=False, debug_func=""):
        """ Produces collection of log entries that match some parameters.
        This is used to search the log for messages related to the one being processed.
        An example would be looking for a synGraphCompile of a graph that is
        beeing launched based on the graph handle, or for a previous memcopy to a
        given destination address.

        Parameters
            where : iterable with traces
            what:  iterable of 2-tuples (<property_path>, <value>).
                property_path is a string of dot-separated identifiers and <value>
                is the expected value.
                Eg what=(("func.name", "synFunc"),) searches for element['func'].name == "synFunc".
            skip: number of log entries to be ignored
            reverse (bool): whether to search the log in reverse order
            debug_func: a helper to enter debug mode when looking for a certain function
                When set to a function name this enables detailed log on relevant queries
                that may reveal why some query doesn't hit a match.

        """
        rng = range(skip - 1, 0, -1) if reverse else range(skip, len(where), 1)

        def null_log(*_):
            pass

        debug_log = null_log
        for no in rng:
            _, entry = where[no]
            for pno, (prop, value) in enumerate(what):
                debug_log(f"looking for {value} in {prop}")
                e = entry
                for p in prop:

                    if isinstance(e, Mapping):
                        debug_log(f"going to '{p}' among {e.keys()}")
                        sub = e.get(p, None)
                    elif isinstance(e, tuple) and isinstance(p, int):
                        debug_log(f"going to index '{p}' of list lenght {len(e)}")
                        sub = e[p]
                    else:
                        debug_log(f"going to atribute '{p}' of object >>>{e}<<< ")
                        sub = getattr(e, p, None)
                    if sub != None:
                        e = sub
                    else:
                        break
                if e != value:
                    debug_log(f"miss because {e} != {value}")
                    e = None
                    debug_log = null_log
                    break
                if prop == ("func", "name") and value == debug_func:
                    debug_log = log.info
            if e != None:
                yield no, entry

    @staticmethod
    def find_first(where, what, skip=0, reverse=False):
        try:
            lookup = Flow.find(where, what, skip, reverse)
            return next(lookup)
        except StopIteration:
            log.error(f"failed to find {what} starting at {skip} in {'reverse' if reverse else 'normal'} order")
            raise

    def find_first_call(self, func_name, what, skip=0, reverse=False):
        return Flow.find_first(
            self.log,
            ((("name",), "call"), (("ph",), "B"), (("func", "name"), func_name)) + what,
            skip=skip,
            reverse=reverse,
        )

    @staticmethod
    def arg_type(entry, arg):
        return entry["func"].args[entry["args"].index(arg)]

    @staticmethod
    def call(entry, args, wrap="ASSERT_EQ", expect_status="synSuccess"):
        """
        Parameters
        ----------
        expect_status : None or synStatus as either string or int
            When set to None then emitting assert that a call returns status that was actually captured during log-capture session.
            When set to a status, the assert will expect given value.
        """
        result = entry["func"].name + "("
        rargs = [f"/*{n}=*/" + str(args.get(n, a)) for n, a in entry["args"].items()]
        result += ", ".join(rargs) + ")"

        if expect_status is None:
            expect_status = entry["result"]["status"]

        if entry["func"].return_type == "synStatus" and wrap:
            status = "synSuccess" if expect_status == 0 else f"synStatus({expect_status})"
            result = f"{wrap}({status}, {result});"
        else:
            result = result + ";"
        return result

    class SpacesMap:
        # maps objects from the log (pointers at time of log collection) to names of variables in C code
        def __init__(self, objs, renderer):
            self.memory = defaultdict(str)
            self.memory["0"] = "nullptr", "nullptr_t*"
            self.device_allocations = dict()
            self.objs = objs
            self.renderer = renderer
            # measure maximum accessed addr to possibly run test in environment
            # with less memory (e.g. capture on asic and reproduce in simulator)
            self.max_used_device_address = 0

        def get(self, key, value_type):
            replacement, replacement_type = self.memory.get(key, (f"MAP_FAIL({key})", "void*"))
            cast = ""
            repl_type_is_ptr, arg_type_is_ptr = (replacement_type.find("*") > 0, value_type.find("*") > 0)
            if arg_type_is_ptr and not repl_type_is_ptr:
                cast = "&"
            elif replacement_type != value_type:
                cast = f"({value_type})"
            return f"{cast}{replacement}"

        def get_args(self, entry, args):
            return {arg: self.get(entry["args"][arg], entry["func"].args[arg]) for arg in args}

        def add(self, key, var_type, var_name, size="", initializer="", local=False, extra_comment=""):
            if key in self.memory.keys() and not local:  # local overrides current contents of map
                var = self.memory[key]
            else:
                var = var_name, var_type
                extra_comment = extra_comment if extra_comment else f"replacement of {key}"
                self.renderer.var(f"{var_type} {var_name}{size}")
                self.renderer.out(f"{var_name}{size}{initializer};  // {extra_comment}")

            self.memory[key] = var
            return var[0]

        def map_host_data(self, ptr, size, name):
            entry = self.objs.get(ptr, dict())
            if not entry:
                self.renderer.out(f"std::vector<uint8_t> {name}_buffer({size}, 0);")
                initializer = f" = {name}_buffer.data()"
            else:
                initializer = f" = data_adr + {entry['args']['data_offset']}" if entry else "= {0}"
            ret = self.add(ptr, "uint8_t*", name, "", initializer)
            return ret

        def map_device_suballocation(self, no, device_address, size=0):
            int_device_address = int(device_address[2:], 16)
            for str_alloc_addr, alloc_size in self.device_allocations.items():
                alloc_addr = int(str_alloc_addr[2:], 16)
                if int_device_address >= alloc_addr and int_device_address < alloc_addr + alloc_size:
                    base = self.get(str_alloc_addr, "uint64_t")
                    offset = int_device_address - alloc_addr
                    v = self.add(
                        device_address,
                        "uint64_t",
                        f"devmem_{no}",
                        initializer=f"={base} + 0x{offset:X}",
                        extra_comment=f"allocation {str_alloc_addr}+0x{offset:X}, pool end 0x{alloc_size:X}",
                    )
                    self.max_used_device_address = max(self.max_used_device_address, int(device_address[2:], 16) + size)
                    return v
            log.error(f"Cannot map device suballocation at address {device_address}, seems it wasn't allocated")
            return f"MAP_FAIL({device_address})"

        def add_device_pool(self, no, device_ptr: str, size):
            self.device_allocations[device_ptr] = size
            alloc_end = int(device_ptr, 16) + size
            v = self.add(
                device_ptr,
                "uint64_t",
                f"devmem_{no}",
                extra_comment=f"allocation {device_ptr}:0x{alloc_end:x} ({size} bytes)",
            )

    class MultiThreadedRenderer:
        def __init__(self):
            self.threads = {}
            self.vars = ""
            self.bin_file_size = "x"

        def set_tid(self, tid):
            self.tid = tid
            if not tid in self.threads:
                self.threads[tid] = StringIO()

        def out(self, *args, **kwargs):
            print(" ", *args, **kwargs, file=self.threads[self.tid])

        def sync(self, no):
            self.out(f"x.sync({no});")

        def var(self, var_def, comment="", initializer="{}"):
            self.vars += f"{var_def}{initializer};  // {comment}\n  "

        def _render_threads(self, out):
            for tid, body in self.threads.items():
                out(f"void thread_proc_{tid}() {{")
                out(body.getvalue())
                out(f'  std::clog << "worker thread " << get_current_tid() << " (orig {tid}) completed" << std::endl;')
                out(f"}} // thread_proc{tid}")

        def _render_main(self, out):
            out("TEST(sample_test_case, sample_test) {")
            out("  {\n    std::unique_ptr<logger_test> t{new logger_test()};\n  t->run();\n  }")
            out('  std::clog << "Finished\\n";')
            out("}\n")

        def render(self, out):
            out(
                "#include <gtest/gtest.h>\n"
                "#include <iostream>\n"
                "#include <vector>\n"
                "#include <sys/mman.h>\n"
                "#include <fcntl.h>\n"
                "#include <sys/stat.h>\n"
                "#include <unistd.h>\n"
                "#include <hcl_api.h>\n"
                "#include <synapse.h>\n"
                "#include <synapse_api.h>\n"
                "#include <synapse_api_types.h>\n"
                '#include "../syncrotron.h"\n'
                '#include "../compare.h"\n'
                "#include <perf_lib_layer_params.h>\n"
            )

            out(
                "struct test_base {"
                "int data_fd;\n"
                "unsigned char* data_adr;\n"
                "size_t data_file_size{};\n"
                "uint32_t device_id;\n"
            )
            out(self.vars)
            if self.bin_file_size:
                out(
                    "test_base() {\n"
                    "  initialize();\n"
                    "}\n"
                    "void initialize() {\n"
                    f"  size_t data_file_size={self.bin_file_size};\n"
                    "  struct stat data_file_stat;\n"
                    '  const char* bin_file_name=".local.synapse_log.data";\n'
                    "  data_fd = open(bin_file_name, O_RDWR, 0);\n"
                    "  int mmap_flags = MAP_PRIVATE | MAP_POPULATE;\n"
                    "  if (data_fd != -1) {\n"
                    "      ASSERT_EQ(fstat(data_fd, &data_file_stat), 0);\n"
                    "      data_file_size = data_file_stat.st_size;\n"
                    "  } else if (errno == ENOENT) {\n"
                    '      printf("WARNING: cannot open `%s` so I am using zeros. This may affect test behavior.\\n", bin_file_name);\n'
                    "      mmap_flags |= MAP_ANONYMOUS;\n"
                    "  } else\n"
                    '      ASSERT_EQ(errno, 0) << "failed to open binary file";\n'
                    "  data_adr = (unsigned char*) mmap(NULL, data_file_size, PROT_READ | PROT_WRITE, mmap_flags, data_fd, 0);\n"
                    '  printf("mmap\'ed 0x%zx bytes data file at %p\\n", data_file_size, data_adr);\n'
                    '  ASSERT_NE(data_adr, MAP_FAILED) << "mmapping of bin file failed" << errno;'
                    "} // constructor\n"
                )

            out("};\n\n")
            out("struct logger_test:public test_base {")
            out("syncrotron x;")
            out("void run() {")
            for tid in self.threads:
                out(f"  x.add_proc(&logger_test::thread_proc_{tid}, this);")

            out("  x.start();\n}//run\n")
            if self.bin_file_size:
                out("~logger_test() {")
                out("  munmap(data_adr, data_file_size);")
                out("  close(data_fd);\n}\n")
            self._render_threads(out)
            out("};")
            self._render_main(out)

    class SingleThreadedRenderer:
        def __init__(self):
            self.bin_file_size = "x"
            self.out_file = StringIO()

        def set_tid(self, tid):
            pass

        def sync(self, no):
            pass

        def out(self, *args, **kwargs):
            print(" ", *args, **kwargs, file=self.out_file)

        def var(self, var_def, comment="", initializer="{}"):
            self.out(f"{var_def}{initializer};  // {comment}\n  ")

        def render(self, out):
            out(
                "#include <gtest/gtest.h>\n"
                "#include <iostream>\n"
                "#include <vector>\n"
                "#include <sys/mman.h>\n"
                "#include <fcntl.h>\n"
                "#include <sys/stat.h>\n"
                "#include <unistd.h>\n"
                "#include <hcl_api.h>\n"
                "#include <synapse.h>\n"
                "#include <synapse_api.h>\n"
                "#include <synapse_api_types.h>\n"
                '#include "../compare.h"\n'
                "#include <perf_lib_layer_params.h>\n"
            )

            out("TEST(sample_test_case, sample_test) {")
            if self.bin_file_size:
                out(f"size_t data_file_size={self.bin_file_size};")
                out("struct stat data_file_stat;")
                out('const char* bin_file_name=".local.synapse_log.data";')
                out(
                    "int data_fd = open(bin_file_name, O_RDWR, 0);\n"
                    "int mmap_flags = MAP_PRIVATE | MAP_POPULATE;\n"
                    "if (data_fd != -1) {\n"
                    "    ASSERT_EQ(fstat(data_fd, &data_file_stat), 0);\n"
                    "    data_file_size = data_file_stat.st_size;\n"
                    "} else if (errno == ENOENT) {\n"
                    '    printf("WARNING: cannot open `%s` so I am using zeros. This may affect test behavior.\\n", bin_file_name);'
                    "    mmap_flags |= MAP_ANONYMOUS;\n"
                    "} else\n"
                    '    ASSERT_EQ(errno, 0) << "failed to open binary file";\n'
                    "unsigned char* data_adr = (unsigned char*) mmap(NULL, data_file_size, PROT_READ | PROT_WRITE, mmap_flags, data_fd, 0);"
                    'printf("mmap\'ed data file at %p\\n", data_adr);\n'
                )
                out('ASSERT_NE(data_adr, MAP_FAILED) << "mappping of bin file failed" << errno;')
            out("uint32_t device_id;")
            out(self.out_file.getvalue())
            out('  std::clog << "Finished\\n";')
            if self.bin_file_size:
                out("ASSERT_EQ(munmap(data_adr, data_file_size), 0);")
                out("close(data_fd);")
            out("}\n")

    def dump_c(self, renderer, device_address_limit=None):
        class MSPACE:
            WORKSPACE_SIZE = "workspace size"
            RECIPE_SIZE = "recipe size"

        space = Flow.SpacesMap(self.objs, renderer)

        for no, entry in self.log:
            try:
                renderer.set_tid(entry["tid"])
                out = renderer.out
                renderer.sync(no)
                args = entry.get("args", {})
                if entry["name"] == "reference":  # training api

                    v = f"ref{no}"
                    out(f"float* {v} = (float*)(data_adr + {args['data_offset']});\n")

                    recipe_name, tensor_name = args["to"].split(
                        ":"
                    )  # e.g. ".graph_dumps/habana_cluster_0_1-recipe_0:tensor3"
                    compile_no, compile_node = self.find_first_call(
                        "synGraphCompile", ((("args", "pRecipeName"), f'"{recipe_name}"'),), skip=no, reverse=True
                    )
                    upload_no, upload_node = self.find_first_call(
                        "synLaunch",
                        ((("args", "pRecipehandle"), compile_node["result"]["pRecipeHandle"]),),
                        skip=no,
                        reverse=True,
                    )
                    out_patching = upload_node["args"]["launchTensorsInfo"]
                    out_patching = zip(out_patching[::2], out_patching[1::2])
                    dev_mem = next(
                        dev_addr for enq_tensor_name, dev_addr in out_patching if enq_tensor_name == tensor_name
                    )

                    memcpy = self.find_first_call(
                        "synMemCopyAsync", ((("args", "src"), dev_mem),), skip=no, reverse=True
                    )
                    log.info(
                        f"comparing against destination of a mamcpy {memcpy[1]['args']['src']} to {memcpy[1]['args']['dst']}"
                    )
                    data_ptr = space.memory[memcpy[1]["args"]["dst"]][0]
                    out(f"ASSERT_TRUE(compare({v}, ({args['data_cast']}*){data_ptr}, {args['length']}));")
                elif entry["name"] == "object":
                    if args["type"] == "std::vector<TransposePermutationDim>":
                        renderer.var(
                            f"{args['type']} params_{no}",
                            "",
                            initializer="{"
                            + ",".join((TransposePermutationDim.from_int(p) for p in args["fields"]))
                            + "}",
                        )
                        renderer.var(f"{args['type']}* object_{no}", "", initializer=f"{{&params_{no}}}")
                        space.memory[args["at"]] = (f"object_{no}", args["type"] + "*")

                    if args["type"] == "synTensorDescriptor":
                        descriptor = args["fields"]

                        out(f"unsigned dims{no}[5] = {{" + ",".join((str(dim) for dim in descriptor["m_sizes"])) + "};")

                        fields = args["fields"]
                        fields["m_dataType"] = syn_types[fields["m_dataType"]][0]
                        fields["m_name"] = f'"{fields["m_name"]}"'
                        fields["m_sizes"] = f"dims{no}"
                        if "const" in args:
                            if "data_offset" in args:
                                fields["m_ptr"] = f"data_adr + " + str(args["data_offset"])
                            else:
                                out(f"std::vector<uint8_t> tensor_descriptor_{no}_buffer({args['byte_size']}, 0);")
                                out(f"uint8_t* tensor_descriptor_{no}_data = tensor_descriptor_{no}_buffer.data();")
                                fields["m_ptr"] = f"tensor_descriptor_{no}_data"
                        else:
                            fields["m_ptr"] = f"nullptr"
                        fields = ", ".join(f"/*.{k}*/{v}" for k, v in fields.items())
                        v = space.add(
                            args["at"],
                            args["type"],
                            f"tensor_descriptor_{no}",
                            initializer=f"={{{fields}}}",
                            local=True,
                        )
                    else:
                        v = f"object_{no}"
                        if "::Params" in args["type"]:
                            v = f"params{no}"
                        if "Attrib" in args["type"]:
                            v = f"attrib{no}"

                        if "value" in args:
                            out(f"uint8_t {v}_data[] = {args['value']};")
                            v = space.add(
                                args["at"],
                                args["type"] + "*",
                                v,
                                initializer=f"=({args['type']}*) {v}_data",
                                local=True,
                            )
                        v = space.add(args["at"], args["type"], v)

                elif entry["name"] == "call" and entry["ph"] == "B":
                    func_def = entry["func"]
                    if func_def.name == "synDeviceAcquireByDeviceType":
                        out(Flow.call(entry, {"pDeviceId": "&device_id", "deviceType": "synDeviceGaudi"}))
                    elif func_def.name == "synDeviceGetMemoryInfo":
                        v = space.add(entry["result"]["free"], "uint64_t", f"device_free_memory{no}", local=True)
                        args["free"] = entry["result"]["free"]
                        v = space.add(entry["result"]["total"], "uint64_t", f"device_total_memory{no}", local=True)
                        args["total"] = entry["result"]["total"]
                        out(Flow.call(entry, space.get_args(entry, ("free", "total"))))
                    elif func_def.name == "synDeviceMalloc":
                        out("\n")
                        if device_address_limit:
                            limited_size = min(
                                args["size"], device_address_limit - int(entry["result"]["buffer"][2:], 16)
                            )
                            log.info(f"limiting malloc from 0x{args['size']:x} to 0x{limited_size:x}")
                            args["size"] = limited_size

                        space.add_device_pool(no, entry["result"]["buffer"], args["size"])

                        args["buffer"] = entry["result"]["buffer"]
                        replacements = space.get_args(entry, ("buffer",))
                        replacements["flags"] = synMemFlags[args["flags"]]
                        out(Flow.call(entry, replacements, wrap="EXPECT_EQ"))
                    elif func_def.name == "synHostMap":
                        space.map_host_data(args["buffer"], args["size"], f"host_data_{no}")
                        out(Flow.call(entry, space.get_args(entry, ("buffer",))))
                    elif func_def.name in ("synHostUnmap", "synDeviceFree"):
                        out(Flow.call(entry, space.get_args(entry, ("buffer",))))
                    elif func_def.name == "synCreateGenericNodeEx":  # old API
                        replacements = dict()
                        mapped_inputs = ", ".join([space.memory[tensor][0] for tensor in args["inputs"]])
                        if mapped_inputs:
                            out(f"synTensor node{no}_inputs[] = {{{mapped_inputs}}};")
                            replacements["inputs"] = f"node{no}_inputs"
                        else:
                            replacements["inputs"] = "nullptr"

                        mapped_outputs = ", ".join([space.memory[tensor][0] for tensor in args["outputs"]])
                        if mapped_outputs:
                            out(f"synTensor node{no}_outputs[] = {{{mapped_outputs}}};\n")
                            replacements["outputs"] = f"node{no}_outputs"
                        else:
                            replacements["outputs"] = "nullptr"

                        if args["inputLayouts"]:
                            input_layouts = '"' + '", "'.join(args["inputLayouts"]) + '", ""'
                            out(f"const char* node{no}_input_layouts[] = {{{input_layouts}}};")
                            replacements["inputLayouts"] = f"node{no}_input_layouts"
                        else:
                            replacements["inputLayouts"] = "nullptr"

                        if args["outputLayouts"]:
                            output_layouts = '"' + '", "'.join(args["outputLayouts"]) + '", ""'
                            out(f"const char* node{no}_output_layouts[] = {{{output_layouts}}};")
                            replacements["outputLayouts"] = f"node{no}_output_layouts"
                        else:
                            replacements["outputLayouts"] = "nullptr"

                        replacements.update(space.get_args(entry, ("userParams",)))
                        out(Flow.call(entry, replacements))
                    elif func_def.name == "synNodeCreate":
                        replacements = dict()
                        mapped_inputs = ", ".join(
                            [space.get(tensor, "synTensor") for tensor in args["pInputsTensorList"]]
                        )
                        if mapped_inputs:
                            out(f"synTensor node{no}_inputs[] = {{{mapped_inputs}}};")
                            replacements["pInputsTensorList"] = f"node{no}_inputs"
                        else:
                            replacements["pInputsTensorList"] = "nullptr"

                        mapped_outputs = ", ".join(
                            [space.get(tensor, "synTensor") for tensor in args["pOutputsTensorList"]]
                        )
                        if mapped_outputs:
                            out(f"synTensor node{no}_outputs[] = {{{mapped_outputs}}};\n")
                            replacements["pOutputsTensorList"] = f"node{no}_outputs"
                        else:
                            replacements["pOutputsTensorList"] = "nullptr"

                        if args["inputLayouts"]:
                            input_layouts = '"' + '", "'.join(args["inputLayouts"]) + '", ""'
                            out(f"const char* node{no}_input_layouts[] = {{{input_layouts}}};")
                            replacements["inputLayouts"] = f"node{no}_input_layouts"
                        else:
                            replacements["inputLayouts"] = "nullptr"

                        if args["outputLayouts"]:
                            output_layouts = '"' + '", "'.join(args["outputLayouts"]) + '", ""'
                            out(f"const char* node{no}_output_layouts[] = {{{output_layouts}}};")
                            replacements["outputLayouts"] = f"node{no}_output_layouts"
                        else:
                            replacements["outputLayouts"] = "nullptr"

                        replacements.update(space.get_args(entry, ("pUserParams", "graphHandle")))
                        out(Flow.call(entry, replacements))
                    elif func_def.name == "synNodeCreateWithId":
                        v = space.add(entry["result"]["nodeUniqueId"], "synNodeId", f"node_id{no}", local=True)
                        args["nodeUniqueId"] = entry["result"]["nodeUniqueId"]

                        replacements = dict()
                        mapped_inputs = ", ".join(
                            [space.get(tensor, "synTensor") for tensor in args["pInputsTensorList"]]
                        )
                        if mapped_inputs:
                            out(f"synTensor node{no}_inputs[] = {{{mapped_inputs}}};")
                            replacements["pInputsTensorList"] = f"node{no}_inputs"
                        else:
                            replacements["pInputsTensorList"] = "nullptr"

                        mapped_outputs = ", ".join(
                            [space.get(tensor, "synTensor") for tensor in args["pOutputsTensorList"]]
                        )
                        if mapped_outputs:
                            out(f"synTensor node{no}_outputs[] = {{{mapped_outputs}}};\n")
                            replacements["pOutputsTensorList"] = f"node{no}_outputs"
                        else:
                            replacements["pOutputsTensorList"] = "nullptr"

                        if args["inputLayouts"]:
                            input_layouts = '"' + '", "'.join(args["inputLayouts"]) + '", ""'
                            out(f"const char* node{no}_input_layouts[] = {{{input_layouts}}};")
                            replacements["inputLayouts"] = f"node{no}_input_layouts"
                        else:
                            replacements["inputLayouts"] = "nullptr"

                        if args["outputLayouts"]:
                            output_layouts = '"' + '", "'.join(args["outputLayouts"]) + '", ""'
                            out(f"const char* node{no}_output_layouts[] = {{{output_layouts}}};")
                            replacements["outputLayouts"] = f"node{no}_output_layouts"
                        else:
                            replacements["outputLayouts"] = "nullptr"
                        replacements.update(space.get_args(entry, ("pUserParams", "graphHandle", "nodeUniqueId")))
                        out(Flow.call(entry, replacements))

                    elif func_def.name == "synNodeDependencySet":
                        replacements = dict()
                        mapped_blocking = ", ".join(
                            [space.get(str(node_id), "synNodeId") for node_id in args["pBlockingNodesIdList"]]
                        )

                        if mapped_blocking:
                            out(f"synNodeId node{no}_blocking[] = {{{mapped_blocking}}};")
                            replacements["pBlockingNodesIdList"] = f"node{no}_blocking"
                        else:
                            replacements["pBlockingNodesIdList"] = "nullptr"

                        mapped_blocked = ", ".join(
                            [space.get(str(node_id), "synNodeId") for node_id in args["pBlockedNodesIdList"]]
                        )
                        if mapped_blocked:
                            out(f"synNodeId node{no}_blocked[] = {{{mapped_blocked}}};\n")
                            replacements["pBlockedNodesIdList"] = f"node{no}_blocked"
                        else:
                            replacements["pBlockedNodesIdList"] = "nullptr"

                        replacements.update(space.get_args(entry, ("graphHandle",)))
                        out(Flow.call(entry, replacements))

                    elif func_def.name == "synMemCopyAsync":

                        dma_dir = synDmaDir(args["direction"])
                        if dma_dir in (synDmaDir.HOST_TO_DRAM, synDmaDir.DRAM_TO_DRAM):
                            space.map_device_suballocation(no, args["dst"], entry["args"]["size"])
                        elif dma_dir in (synDmaDir.DRAM_TO_HOST, synDmaDir.DRAM_TO_DRAM):
                            space.map_device_suballocation(no, args["src"], entry["args"]["size"])
                        else:
                            assert False, f"Unknown mem copy direction {args['direction']}"

                        replacements = space.get_args(entry, ("streamHandle", "src", "dst"))
                        replacements["direction"] = f"synDmaDir::{dma_dir.name}"
                        out(Flow.call(entry, replacements))
                    elif func_def.name == "synTrainingEnqueue":
                        v = space.add(entry["result"]["handle"], "synWaitHandle", f"waitEvent{no}", local=True)
                        args["handle"] = entry["result"]["handle"]
                        out(Flow.call(entry, space.get_args(entry, ("handle",))))

                    elif func_def.name == "synDestroyTensor":
                        out(Flow.call(entry, space.get_args(entry, ("tensor",))))
                    elif func_def.name == "synGraphCreate":
                        v = space.add(entry["result"]["pGraphHandle"], "synGraphHandle", f"graph{no}", local=True)
                        args["pGraphHandle"] = entry["result"]["pGraphHandle"]
                        out(Flow.call(entry, space.get_args(entry, ("pGraphHandle",))))
                    elif func_def.name == "synGraphCompile":
                        v = space.add(entry["result"]["pRecipeHandle"], "synRecipeHandle", f"recipe{no}", local=True)
                        args["pRecipeHandle"] = entry["result"]["pRecipeHandle"]
                        out(Flow.call(entry, space.get_args(entry, ("graphHandle", "pRecipeHandle"))))
                    elif func_def.name == "synGraphDestroy":
                        out(Flow.call(entry, space.get_args(entry, ("graphHandle",))))
                    elif func_def.name == "synCreateTensorEx":
                        v = space.add(entry["result"]["tensor"], "synTensor", f"tensor{no}", local=True)
                        args["tensor"] = entry["result"]["tensor"]
                        out(Flow.call(entry, space.get_args(entry, ("pDescriptor", "tensor", "userContext"))))
                    elif func_def.name == "synTensorCreate":
                        v = space.add(entry["result"]["pTensor"], "synTensor", f"tensor{no}", local=True)
                        args["pTensor"] = entry["result"]["pTensor"]
                        out(Flow.call(entry, space.get_args(entry, ("descriptor", "pTensor", "pSectionHandle"))))
                    elif func_def.name == "synConstTensorCreate":
                        v = space.add(entry["result"]["pTensor"], "synTensor", f"tensor{no}", local=True)
                        args["pTensor"] = entry["result"]["pTensor"]
                        out(Flow.call(entry, space.get_args(entry, ("descriptor", "pTensor"))))
                    elif func_def.name == "synSectionCreate":
                        v = space.add(
                            entry["result"]["sectionHandle"], "synSectionHandle", f"memsection{no}", local=True
                        )
                        args["sectionHandle"] = entry["result"]["sectionHandle"]
                        out(Flow.call(entry, space.get_args(entry, ("graph", "sectionHandle"))))
                    elif func_def.name == "synStreamCreate":
                        v = space.add(entry["result"]["pStreamHandle"], "synStreamHandle", f"stream{no}", local=True)
                        args["pStreamHandle"] = entry["result"]["pStreamHandle"]
                        replacements = space.get_args(entry, ("pStreamHandle",))
                        streamType = args["streamType"]
                        replacements["streamType"] = f"(synStreamType) {streamType}"
                        out(Flow.call(entry, replacements))
                    elif func_def.name in ("synStreamDestroy", "synStreamSynchronize"):
                        out(Flow.call(entry, space.get_args(entry, ("streamHandle",))))
                    elif func_def.name == "synStreamWaitEvent":
                        out(Flow.call(entry, space.get_args(entry, ("streamHandle", "eventHandle"))))
                    elif func_def.name in ("synSectionDestroy"):
                        out(Flow.call(entry, space.get_args(entry, ("sectionHandle",))))
                    elif func_def.name == "synLaunch":
                        replacements = space.get_args(entry, ("streamHandle", "pRecipehandle"))
                        mapped_tensors = [
                            space.map_device_suballocation(f"{no}_{i}", ptr)
                            for i, ptr in enumerate(args["launchTensorsInfo"][1::2])
                        ]
                        tensors_info = ", ".join(
                            f'{{"{n}", {f}}}' for n, f in zip(args["launchTensorsInfo"][::2], mapped_tensors)
                        )
                        out(f"synLaunchTensorInfo launch_tensors_info{no}[] = {{{tensors_info}}};")
                        replacements["launchTensorsInfo"] = f"launch_tensors_info{no}"
                        out(Flow.call(entry, replacements))
                    elif func_def.name == "synEventCreate":
                        v = space.add(
                            entry["result"]["pEventHandle"], "synEventHandle", f"event_handle{no}", local=True
                        )
                        args["pEventHandle"] = entry["result"]["pEventHandle"]
                        out(Flow.call(entry, space.get_args(entry, ("pEventHandle",))))
                    elif func_def.name in ("synEventRecord",):
                        out(Flow.call(entry, space.get_args(entry, ("eventHandle", "streamHandle"))))
                    elif func_def.name in ("synEventSynchronize", "synEventDestroy"):
                        out(Flow.call(entry, space.get_args(entry, ("eventHandle",))))
                    elif func_def.name == "synWorkspaceGetSize":
                        key = (MSPACE.WORKSPACE_SIZE, args["recipeHandle"])
                        v = space.add(key, "uint64_t", f"workspace_size{no}", local=True)
                        args["pWorkspaceSize"] = key
                        out(Flow.call(entry, space.get_args(entry, ("recipeHandle", "pWorkspaceSize"))))
                    elif func_def.name == "synDestroy":
                        out(Flow.call(entry, {}, wrap=None))
                    elif func_def.name == "HCL_Comm_Size":
                        v = space.add(entry["result"]["size"], "int", f"size{no}", local=False)
                        args["size"] = entry["result"]["size"]
                        entry["args"]["comm"] = '"{}"'.format(entry["args"]["comm"])
                        out(Flow.call(entry, space.get_args(entry, ("size",))))
                    elif func_def.name == "HCL_Comm_Rank":
                        v = space.add(entry["result"]["rank"], "HCL_Rank", f"rank{no}", local=False)
                        args["rank"] = entry["result"]["rank"]
                        entry["args"]["comm"] = '"{}"'.format(entry["args"]["comm"])
                        out(Flow.call(entry, space.get_args(entry, ("rank",))))
                    elif func_def.name == "HCL_Get_Intermediate_Buffer_size":
                        v = space.add(
                            entry["result"]["intermediateSize"], "uint64_t", f"intermediateSize{no}", local=False
                        )
                        args["intermediateSize"] = entry["result"]["intermediateSize"]
                        replacements = space.get_args(entry, ("intermediateSize",))
                        replacements["communicator"] = '"{}"'.format(entry["args"]["communicator"])
                        replacements["dataType"] = syn_types[int(entry["args"]["dataType"], 16)][0]
                        replacements["collectiveOp"] = hcl_collective_ops[int(entry["args"]["collectiveOp"], 16)]
                        out(Flow.call(entry, replacements))
                    elif func_def.name == "HCL_Allreduce":
                        replacements = space.get_args(entry, ("streamHandle", "intermediateSize"))
                        mapped_tensors = {
                            name: space.map_device_suballocation(f"{no}_{name}", entry["args"][name])
                            for name in ["sendBuffAddr", "receiveBuffAddr", "intermediateBufferAddr"]
                        }
                        replacements.update(mapped_tensors)
                        replacements["dataType"] = syn_types[int(entry["args"]["dataType"], 16)][0]
                        replacements["op"] = hcl_ops[int(entry["args"]["op"], 16)]
                        entry["args"]["communicator"] = '"{}"'.format(entry["args"]["communicator"])
                        out(Flow.call(entry, replacements))
                    elif func_def.name == "HCL_Bcast":
                        replacements = space.get_args(entry, ("streamHandle",))
                        replacements["dataType"] = syn_types[int(entry["args"]["dataType"], 16)][0]
                        replacements["communicator"] = '"{}"'.format(entry["args"]["communicator"])
                        mapped_tensors = {
                            name: space.map_device_suballocation(f"{no}_{name}", entry["args"][name])
                            for name in ["sendBuffAddr", "receiveBuffAddr"]
                        }
                        replacements.update(mapped_tensors)
                        out(Flow.call(entry, replacements))
                    elif func_def.name == "HCL_AllGather":
                        replacements = space.get_args(entry, ("streamHandle",))
                        replacements["communicator"] = '"{}"'.format(entry["args"]["communicator"])
                        replacements["dataType"] = syn_types[int(entry["args"]["dataType"], 16)][0]

                        mapped_tensors = {
                            name: space.map_device_suballocation(f"{no}_{name}", entry["args"][name])
                            for name in ["sendBufAddr", "receiveBuffAddr"]
                        }
                        replacements.update(mapped_tensors)

                        out(Flow.call(entry, replacements))
                    elif func_def.name == "HCL_Wait":
                        request = args["phRequest"].strip("{}")
                        request = request.split(",")
                        out(f"HCL_Request hcl_request{no};\n")
                        out(f"hcl_request{no}.event={request[0]};\n")
                        out(f"hcl_request{no}.index={request[1]};\n")
                        out(f"hcl_request{no}.pIndex={request[2]};\n")
                        out(Flow.call(entry, {"phRequest": f"hcl_request{no}"}))
                    else:
                        out(Flow.call(entry, {}))
                elif entry["name"][:5] == "mutex":
                    pass
                elif entry["name"] == "call" and entry["ph"] == "E":
                    pass
                else:
                    log.info(f"nothing for entry {entry}")

            except Exception as e:
                log.error(f"Error when processing entry line {no}\n{entry}")
                # log.error("--------------------------")
                # log.error(str(self.log))
                # log.error("--------------------------")
                raise
        log.info(
            f"You may rerun this generator with --device_address_limit=0x{space.max_used_device_address} to limit memory usage"
        )


def main():
    logging.basicConfig(level=logging.DEBUG)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "logger_test", ".local.src")
    parser = argparse.ArgumentParser(description="Produce bare synapse API gtest from synapse_logger trace")
    parser.add_argument("--input_file", default=".local.synapse_log.json")
    parser.add_argument("--output_dir", default=path)
    parser.add_argument("--device_address_limit", default=None, type=lambda x: int(x[2:], 16))
    parser.add_argument("--test_flavour", default="multithreaded", choices=("singlethreaded", "multithreaded"))
    parser.add_argument("--verbose", "-v", action="store_true", help="produce debug output")
    args = parser.parse_args()
    f = Flow(args.input_file)
    os.makedirs(os.path.join(args.output_dir, ".graph_dumps"), exist_ok=True)
    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.WARN))
    logging.getLogger("synapse_logger").setLevel(logging.DEBUG if args.verbose else logging.WARN)
    out_file = os.path.abspath(os.path.join(args.output_dir, "test.cxx"))
    log.info(f"writing output to {out_file}")
    log.debug(f"writing output to {out_file}")
    with open(out_file, "w") as src_file:

        if args.test_flavour == "singlethreaded":
            renderer = Flow.SingleThreadedRenderer()
        elif args.test_flavour == "multithreaded":
            renderer = Flow.MultiThreadedRenderer()

        f.dump_c(renderer, device_address_limit=args.device_address_limit)
        renderer.bin_file_size = f.bin_file_size

        def out_src(*args, **kwargs):
            print(*args, **kwargs, file=src_file)

        renderer.render(out_src)


if __name__ == "__main__":
    main()
