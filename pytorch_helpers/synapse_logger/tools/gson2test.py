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
import json
import logging
import os
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Mapping
from enum import Enum
from io import StringIO
from shutil import copy2

from browse_log import is_call
from gson_parsing import func_def_from_pretty_function, gson_iterator, hcl_collective_ops, hcl_ops, syn_types

log = logging.getLogger("synapse_logger.gson2test")


class ReferenceData:
    def __init__(self, recipe_name, output_tensor_name):
        self.recipe_name, self.output_tensor_name = recipe_name, output_tensor_name
        self.graph, self.dev_pointer, self.host_addr = (None,) * 3


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


def generate_array(no, type, name, nitems, items):
    var_name = f"{name}{no}"
    var_items = ", ".join(items)
    var_body = "{%s}" % var_items
    var_def = f"{type} {var_name}[{nitems}] = {var_body};"
    return var_name, var_def


class Flow:
    class ComputationResult:
        LAST_COMPILED_RECIPE = 1

    def _return_last_compiled_recipe(self):
        if self.last_compiled_recipe is None:
            raise Exception(
                "Flow is executed with computation_result==LAST_COMPILED_RECIPE, but no recipe has been compiled"
            )
        var = self.last_compiled_recipe
        if var[0] != "&":
            raise Exception("self.last_compiled_recipe is not an address of member field")
        var = var[1:]
        return f"return {var};"

    def __init__(self, input_iterator, computation_result=None):
        self.references = list()
        self.objs = {}
        self.bin_file_size = 0
        self.last_compiled_recipe = None
        if computation_result == Flow.ComputationResult.LAST_COMPILED_RECIPE:
            self.return_type = "synRecipeHandle"
            self.get_return_statement = self._return_last_compiled_recipe
        else:
            self.return_type = "void"
            self.get_return_statement = lambda: ""

        if isinstance(input_iterator, str):
            input_iterator = gson_iterator(input_iterator, False)
        self.var_idx = 0
        self.log = input_iterator

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
        class Space:
            def __init__(self, renderer, initial={}):
                self._space = defaultdict(str)
                self._space.update(initial)
                self.renderer = renderer

            def get(self, key, value_type):
                replacement, replacement_type = self._space.get(key, (f"MAP_FAIL({key})", "void*"))
                cast = ""
                repl_type_is_ptr, arg_type_is_ptr = (replacement_type.find("*") > 0, value_type.find("*") > 0)
                if arg_type_is_ptr and not repl_type_is_ptr:
                    cast = "&"
                elif replacement_type != value_type:
                    cast = f"({value_type})"
                    if (value_type.find("**") > 0) and not (replacement_type.find("**") > 0):
                        cast = cast + "&"
                return f"{cast}{replacement}"

            def add(self, key, var_type, var_name, size="", initializer="", local=False, extra_comment=""):
                if key in self._space.keys() and not local:  # local overrides current contents of map
                    var = self._space[key]
                else:
                    var = var_name, var_type
                    extra_comment = extra_comment if extra_comment else f"replacement of {key}"
                    self.renderer.var(f"{var_type} {var_name}{size}", extra_comment)

                    if initializer:
                        self.renderer.out(f"{var_name}{size}{initializer};  // {extra_comment}")

                self._space[key] = var
                return var[0]

        # maps objects from the log (pointers at time of log collection) to names of variables in C code
        def __init__(self, objs, renderer):
            self.memory = Flow.SpacesMap.Space(renderer, {"0": ("nullptr", "nullptr_t*")})
            self.node_id = Flow.SpacesMap.Space(renderer)

            self.device_allocations = dict()
            self.host_allocations = dict()
            self.objs = objs
            self.renderer = renderer
            # measure maximum accessed addr to possibly run test in environment
            # with less memory (e.g. capture on asic and reproduce in simulator)
            self.max_used_device_address = 0

        @staticmethod
        def is_typeptr(s):
            return s.find("*") > 0

        @staticmethod
        def unref_type(s):
            s = s.strip()
            if s[-1] != "*":
                raise Exception("cannot unreference type: " + s)
            return s[:-1]

        def get_one(self, key, value_type):
            return self.memory.get(key, value_type)

        def get(self, key, value_type):
            if type(key) is list:
                if self.is_typeptr(value_type):
                    referenced_value_type = self.unref_type(value_type)
                    elems = map(lambda x: self.get_one(x, referenced_value_type), key)
                    return list(elems)
                else:
                    raise Exception(
                        "Cannot handle argument --  key is a list, but the type is not recognized as pointer"
                    )
            else:
                return self.get_one(key, value_type)

        def get_args(self, entry, args):
            return {arg: self.memory.get(entry["args"][arg], entry["func"].args[arg]) for arg in args}

        def add(self, key, var_type, var_name, size="", initializer="", local=False, extra_comment=""):
            return self.memory.add(
                key, var_type, var_name, size=size, initializer=initializer, local=local, extra_comment=extra_comment
            )

        def map_host_data(self, ptr, size, name):
            self.renderer.out(f"std::vector<uint8_t> {name}_buffer({size}, 0);")
            initializer = f" = {name}_buffer.data()"
            ret = self.add(ptr, "uint8_t*", name, "", initializer)
            self.host_allocations[ptr] = size
            return ret

        def map_host_suballocation(self, no, host_address, size=0, src=False):
            result, _ = self.map_suballocation(
                f"hostmem_{no}", "uint8_t*", host_address, self.host_allocations, size=size
            )
            if src:
                entry = self.objs.get(host_address, False)
                offset = entry["args"]["data_offset"] if entry else 0
                self.renderer.out(f"std::copy(data_adr+{offset}, data_adr+{offset}+{size}, {result});")
            return result

        def map_device_suballocation(self, no, device_address, size=0):
            result, allocation_end = self.map_suballocation(
                f"devmem_{no}", "uint64_t", device_address, self.device_allocations, size=size
            )
            self.max_used_device_address = max(self.max_used_device_address, allocation_end)
            return result

        def map_suballocation(self, variable_name, variable_type, address, allocation_map, size=0):
            int_address = int(address[2:], 16)
            for str_alloc_addr, alloc_size in allocation_map.items():
                alloc_addr = int(str_alloc_addr[2:], 16)
                if int_address >= alloc_addr and int_address < alloc_addr + alloc_size:
                    base = self.get(str_alloc_addr, variable_type)
                    offset = int_address - alloc_addr
                    v = self.add(
                        address,
                        variable_type,
                        variable_name,
                        initializer=f"={base} + 0x{offset:X}",
                        extra_comment=f"allocation {str_alloc_addr}+0x{offset:X}, pool end 0x{alloc_size:X}",
                    )
                    return v, int(address[2:], 16) + size
            log.error(f"Cannot map suballocation at address {address}, seems it wasn't allocated")
            return f"MAP_FAIL({address})", 0

        def add_device_pool(self, no, device_ptr: str, size):
            return self.add_pool(self.device_allocations, device_ptr, f"devmem_{no}", "uint64_t", size)

        def add_host_pool(self, no, host_ptr: str, size):
            return self.add_pool(self.host_allocations, host_ptr, f"hostmem_{no}", "void*", size)

        def add_pool(self, allocation_map, ptr: str, variable_name, variable_type, size):
            allocation_map[ptr] = size
            alloc_end = int(ptr, 16) + size
            v = self.add(
                ptr, variable_type, variable_name, extra_comment=f"allocation {ptr}:0x{alloc_end:x} ({size} bytes)"
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

    class SingleThreadedV2Renderer:
        def __init__(self):
            self.vars = ""
            self.bin_file_size = "x"
            self.body = StringIO()
            self.return_type = "void"
            self.get_return_statement = None
            self.disable_bin_file = False

        def set_tid(self, tid):
            pass

        def out(self, *args, **kwargs):
            print(" ", *args, **kwargs, file=self.body)

        def sync(self, no):
            pass

        def var(self, var_def, comment="", initializer="{}"):
            self.vars += f"{var_def}{initializer};  // {comment}\n  "

        def str_render_main(self):
            lines = [
                "TEST(sample_test_case, sample_test) {",
                "  { std::unique_ptr<logger_test> t{new logger_test()};\n  t->run(); }",
                '  std::clog << "Finished\\n";',
                "}",
            ]
            return "\n".join(lines)

        def str_render_preambule(self):
            lines = [
                "#include <gtest/gtest.h>",
                "#include <iostream>",
                "#include <vector>",
                "#include <sys/mman.h>",
                "#include <fcntl.h>",
                "#include <sys/stat.h>",
                "#include <unistd.h>",
                "#include <hcl_api.h>",
                "#include <memory>",
                "#include <synapse_api.h>",
                "#include <synapse_api_types.h>",
                '#include "../compare.h"',
                "#include <perf_lib_layer_params.h>",
                "",
                "",
            ]

            return "\n".join(lines)

        def str_render_class_ctor_dtor(self):
            lines = [
                "logger_test() {",
                "  initialize();",
                "}",
                "",
                "void initialize() {",
                f"  size_t data_file_size={self.bin_file_size};",
                "  struct stat data_file_stat;",
                '  const char* bin_file_name=".local.synapse_log.data";',
                "  data_fd = open(bin_file_name, O_RDWR, 0);",
                "  int mmap_flags = MAP_PRIVATE | MAP_POPULATE;",
                "  if (data_fd != -1) {",
                "      ASSERT_EQ(fstat(data_fd, &data_file_stat), 0);",
                "      data_file_size = data_file_stat.st_size;",
                "  } else if (errno == ENOENT) {",
                '      printf("WARNING: cannot open `%s` so I am using zeros. This may affect test behavior.", bin_file_name);',
                "      mmap_flags |= MAP_ANONYMOUS;",
                "  } else",
                '      ASSERT_EQ(errno, 0) << "failed to open binary file";',
                "  data_adr = (unsigned char*) mmap(NULL, data_file_size, PROT_READ | PROT_WRITE, mmap_flags, data_fd, 0);",
                '  printf("mmap\'ed 0x%zx bytes data file at %p", data_file_size, data_adr);',
                "  close(data_fd);" '  ASSERT_NE(data_adr, MAP_FAILED) << "mmapping of bin file failed " << errno;',
                "} // constructor",
                "",
                "",
                "~logger_test() {" "  munmap(data_adr, data_file_size);" "}",
            ]
            return "\n".join(lines)

        def str_render_class_code(self):
            lines = [
                f"{self.return_type} run() {{",
                self.body.getvalue(),
                "}",
            ]
            return "\n".join(lines)

        def str_render_class(self):
            lines = [
                "struct logger_test {",
                "int data_fd;",
                "unsigned char* data_adr;",
                "size_t data_file_size{};",
                "uint32_t device_id;",
                self.vars,
                "",
                "",
            ]

            if self.bin_file_size and not self.disable_bin_file:
                lines.append(self.str_render_class_ctor_dtor())

            lines.append(self.str_render_class_code())
            lines.append("};")
            return "\n".join(lines)

        def render(self, out):
            out(self.str_render_preambule())
            out(self.str_render_class())
            out(self.str_render_main())

    class RecipeDumpingRenderer(SingleThreadedV2Renderer):
        def str_render_main(self):
            lines = [
                "int main(int argc, char **argv) {",
                "   if (argc != 2) {",
                '       printf("usage: %s [output-filename]\\n", argv[0]);',
                "       return -1;",
                "   }",
                "   logger_test t;",
                "   auto rh = t.run();",
                "   unlink(argv[1]);",
                "   synRecipeSerialize(rh, argv[1]);",
                "   return 0;" "}",
            ]
            return "\n".join(lines)

        def str_render_preambule(self):
            lines = [
                "#include <iostream>",
                "#include <vector>",
                "#include <unistd.h>",
                "#include <synapse_api.h>",
                "",
                '#define ASSERT_EQ(expected, expr) do { if ((expected) != (expr)) { printf("%s: failed!\\n", #expr); exit(1); }} while (0)',
                '#define EXPECT_EQ(expected, expr) do { if ((expected) != (expr)) { printf("%s: failed!\\n", #expr); exit(1); }} while (0)',
                "",
                "",
            ]
            return "\n".join(lines)

    def configure_renderer(self, renderer):
        renderer.return_type = self.return_type
        renderer.get_return_statement = self.get_return_statement

    def _handle_node_create(self, space, out, no, entry, replacements=None):
        args = entry["args"]
        out("{")
        if not replacements:
            replacements = dict()
        mapped_inputs = ", ".join([space.get(tensor, "synTensor") for tensor in args["pInputsTensorList"]])
        if mapped_inputs:
            out(f"synTensor node{no}_inputs[] = {{{mapped_inputs}}};")
            replacements["pInputsTensorList"] = f"node{no}_inputs"
        else:
            replacements["pInputsTensorList"] = "nullptr"

        mapped_outputs = ", ".join([space.get(tensor, "synTensor") for tensor in args["pOutputsTensorList"]])
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
        out("}")

    def _handle_reference(self, no, entry):
        assert entry["name"] == "reference"
        recipe_name, output_tensor_name = entry["args"]["to"].split(
            ":"
        )  # e.g. ".graph_dumps/habana_cluster_0_1-recipe_0:tensor3"
        self.references.append(ReferenceData(recipe_name, output_tensor_name))

    def _reference_match_graph(self, no, entry):
        for ref in self.references:
            if entry["args"]["pRecipeName"] == ref.recipe_name:
                ref.graph = entry

    def _reference_match_launch(self, no, entry):
        for ref, graph in ((ref, ref.graph) for ref in self.references if ref.graph):
            if entry["args"]["pRecipeHandle"] == graph["args"]["pRecipeHandle"]:
                out_patching = entry["args"]["launchTensorsInfo"]
                out_patching = zip(out_patching[::2], out_patching[1::2])
                dev_pointer = next(
                    dev_addr for enq_tensor_name, dev_addr in out_patching if enq_tensor_name == tensor_name
                )
                ref.dev_pointer = dev_pointer

    def _reference_match_memcopy(self, no, src, dst, size):
        for ref, dev_pointer in ((ref, ref.dev_pointer) for ref in self.references if ref.dev_pointer):
            if entry["args"]["src"] == dev_pointer:
                ref.host_addr = entry["args"]["dst"]

    def _reference_match_object(self, space, no, entry, out):
        for ref, host_addr in ((ref, ref.host_addr) for ref in self.references if ref.host_addr):
            if entry["at"] != host_addr:
                continue

            v = f"ref{no}"
            out(f"float* {v} = (float*)(data_adr + {args['data_offset']});\n")
            log.info(
                f"comparing against destination of a mamcpy {memcpy[1]['args']['src']} to {memcpy[1]['args']['dst']}"
            )
            data_ptr = space.get(memcpy[1]["args"]["dst"], args["data_cast"] + "*")
            out(f"ASSERT_TRUE(compare({v}, {data_ptr}, {args['length']}));")

    def map_memcopy(self, no, space, src, dst, size, dma_dir):
        if dma_dir == synDmaDir.HOST_TO_DRAM:
            space.map_host_suballocation(no, src, size, src=False)
            space.map_device_suballocation(no, dst, size)
        elif dma_dir == synDmaDir.DRAM_TO_HOST:
            space.map_device_suballocation(no, src, size)
            space.map_host_suballocation(no, dst, size)
            self._reference_match_memcopy(no, src, dst, size)
        elif dma_dir == synDmaDir.DRAM_TO_DRAM:
            space.map_device_suballocation(no, src, size)
            space.map_device_suballocation(no, dst, size)
        else:
            assert False, f"Unknown mem copy direction {dma_dir}"

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
                if entry["name"] == "reference":
                    self._handle_reference(no, entry)

                elif entry["name"] == "object":
                    self.objs[args["at"]] = entry
                    new_max_offset = args["data_offset"] + args["byte_size"] if "data_offset" in args else 0
                    self.bin_file_size = max(self.bin_file_size, new_max_offset)

                    if args["type"] == "std::vector<TransposePermutationDim>":
                        space.memory.add(
                            args["at"],
                            args["type"],
                            f"object_{no}",
                            initializer=f"={args['type']}{{"
                            + ",".join((TransposePermutationDim.from_int(p) for p in args["fields"]))
                            + "}",
                        )

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
                        v = space.memory.add(
                            args["at"],
                            args["type"],
                            f"tensor_descriptor_{no}",
                            initializer=f"={{{fields}}}",
                            local=True,
                        )
                    elif args["type"] == "synTensorGeometry":
                        fields = args["fields"]
                        fields["m_sizes"] = "{" + ",".join((str(dim) for dim in fields["m_sizes"])) + "}"
                        fields = ", ".join(f"/*.{k}*/{v}" for k, v in fields.items())
                        v = space.memory.add(
                            args["at"],
                            args["type"],
                            f"tensor_geometry_{no}",
                            initializer=f"={{{fields}}}",
                            local=True,
                        )
                    elif args["type"] == "synTensorDeviceLayout":
                        fields = args["fields"]
                        fields["strides"] = "{" + ",".join((str(dim) for dim in fields["strides"])) + "}"
                        fields["deviceDataType"] = "(synDataType){}".format(fields["deviceDataType"])
                        fields = ", ".join(f"/*.{k}*/{v}" for k, v in fields.items())
                        v = space.memory.add(
                            args["at"],
                            args["type"],
                            f"tensor_layout_{no}",
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
                            v = space.memory.add(
                                args["at"],
                                args["type"] + "*",
                                v,
                                initializer=f"=({args['type']}*) {v}_data",
                                local=True,
                            )
                        else:
                            v = space.map_host_suballocation(no, args["at"], args["byte_size"])
                        v = space.memory.add(args["at"], args["type"], v)

                elif entry["name"] == "call" and entry["ph"] == "B":
                    func_def = entry["func"]
                    if func_def.name == "synDeviceAcquireByDeviceType":
                        # this command can fail, if the device was not available
                        # it happens if the machine has GaudiM card instead of Gaudi
                        # in that case, we record only successful device acquisitions
                        if entry["result"]["status"] == 0:
                            out(Flow.call(entry, {"pDeviceId": "&device_id"}))
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
                    elif func_def.name == "synHostMalloc":
                        space.add_host_pool(no, entry["result"]["buffer"], args["size"])
                        args["buffer"] = entry["result"]["buffer"]
                        replacements = space.get_args(entry, ("buffer",))
                        out(Flow.call(entry, replacements, wrap="EXPECT_EQ"))
                    elif func_def.name == "synHostMap":
                        space.map_host_data(args["buffer"], args["size"], f"host_data_{no}")
                        out(Flow.call(entry, space.get_args(entry, ("buffer",))))
                    elif func_def.name in ("synHostUnmap", "synDeviceFree", "synHostFree"):
                        out(Flow.call(entry, space.get_args(entry, ("buffer",))))
                    elif func_def.name == "synNodeCreate":
                        self._handle_node_create(space, out, no, entry)
                    elif func_def.name == "synNodeCreateWithId":
                        v = space.node_id.add(entry["result"]["nodeUniqueId"], "synNodeId", f"node_id{no}", local=True)
                        replacements = {"nodeUniqueId": f"&{v}"}
                        self._handle_node_create(space, out, no, entry, replacements=replacements)

                    elif func_def.name == "synNodeDependencySet":
                        replacements = dict()
                        mapped_blocking = ", ".join(
                            [space.node_id.get(str(node_id), "synNodeId") for node_id in args["pBlockingNodesIdList"]]
                        )

                        if mapped_blocking:
                            out(f"synNodeId node{no}_blocking[] = {{{mapped_blocking}}};")
                            replacements["pBlockingNodesIdList"] = f"node{no}_blocking"
                        else:
                            replacements["pBlockingNodesIdList"] = "nullptr"

                        mapped_blocked = ", ".join(
                            [space.node_id.get(str(node_id), "synNodeId") for node_id in args["pBlockedNodesIdList"]]
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
                        self.map_memcopy(no, space, args["src"], args["dst"], args["size"], dma_dir)

                        replacements = space.get_args(entry, ("streamHandle", "src", "dst"))
                        replacements["direction"] = f"synDmaDir::{dma_dir.name}"
                        out(Flow.call(entry, replacements))

                    elif func_def.name == "synMemCopyAsyncMultiple":
                        dma_dir = synDmaDir(args["direction"])
                        src = args["src"]
                        dst = args["dst"]
                        size = args["size"]
                        numCopies = args["numCopies"]

                        for i in range(numCopies):
                            self.map_memcopy(no, space, src[i], dst[i], size[i], dma_dir)

                        replacements = space.get_args(entry, ("streamHandle", "src", "dst"))
                        replacements["direction"] = f"synDmaDir::{dma_dir.name}"

                        size_replacement, size_def = generate_array(no, "uint64_t", "size", numCopies, map(str, size))
                        src_replacement, src_def = generate_array(no, "uint64_t", "src", numCopies, replacements["src"])
                        dst_replacement, dst_def = generate_array(no, "uint64_t", "dst", numCopies, replacements["dst"])

                        replacements["size"] = size_replacement
                        replacements["src"] = src_replacement
                        replacements["dst"] = dst_replacement

                        out(size_def)
                        out(src_def)
                        out(dst_def)

                        out(Flow.call(entry, replacements))

                    elif func_def.name == "synTrainingEnqueue":
                        v = space.add(entry["result"]["handle"], "synWaitHandle", f"waitEvent{no}", local=True)
                        args["handle"] = entry["result"]["handle"]
                        out(Flow.call(entry, space.get_args(entry, ("handle",))))

                    elif func_def.name == "synTensorDestroy":
                        out(Flow.call(entry, space.get_args(entry, ("tensor",))))
                    elif func_def.name == "synGraphCreate":
                        v = space.add(entry["result"]["pGraphHandle"], "synGraphHandle", f"graph{no}", local=True)
                        args["pGraphHandle"] = entry["result"]["pGraphHandle"]
                        out(Flow.call(entry, space.get_args(entry, ("pGraphHandle",))))
                    elif func_def.name == "synTensorHandleCreate":
                        v = space.add(entry["result"]["pTensor"], "synTensor", f"tensor{no}", local=True)
                        args["pTensor"] = entry["result"]["pTensor"]
                        replacements = space.get_args(
                            entry,
                            (
                                "pTensor",
                                "graph",
                            ),
                        )
                        tensorType = args["type"]
                        replacements["type"] = f"(synTensorType) {tensorType}"
                        out(Flow.call(entry, replacements))

                    elif func_def.name == "synTensorSetGeometry":
                        replacements = space.get_args(
                            entry,
                            (
                                "tensor",
                                "geometry",
                            ),
                        )
                        geometryType = args["geometryType"]
                        replacements["geometryType"] = f"(synGeometryType) {geometryType}"
                        out(Flow.call(entry, replacements))

                    elif func_def.name == "synGraphCompile":
                        self._reference_match_graph(no, entry)
                        if entry["result"]["status"] == "incomplete":
                            entry["result"]["pRecipeHandle"] = "0xdeadbeef"
                        v = space.add(entry["result"]["pRecipeHandle"], "synRecipeHandle", f"recipe{no}", local=True)
                        args["pRecipeHandle"] = entry["result"]["pRecipeHandle"]
                        replacements = space.get_args(entry, ("graphHandle", "pRecipeHandle"))
                        self.last_compiled_recipe = replacements["pRecipeHandle"]
                        out(Flow.call(entry, replacements))
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
                    elif func_def.name == "synTensorAssignToSection":
                        replacements = space.get_args(
                            entry,
                            (
                                "tensor",
                                "section",
                            ),
                        )
                        out(Flow.call(entry, replacements))
                    elif func_def.name == "synStreamCreateGeneric":
                        v = space.add(entry["result"]["pStreamHandle"], "synStreamHandle", f"stream{no}", local=True)
                        args["pStreamHandle"] = entry["result"]["pStreamHandle"]
                        replacements = space.get_args(entry, ("pStreamHandle",))
                        out(Flow.call(entry, replacements))
                    elif func_def.name == "synDeviceGetNextStreamAffinity":
                        v = space.add(
                            entry["result"]["streamAffinityMask"], "availAffinity", f"affinity{no}", local=True
                        )
                        args["pStreamHandle"] = entry["result"]["streamAffinityMask"]
                        replacements = space.get_args(entry, ("streamAffinityMask",))
                        out(Flow.call(entry, replacements))
                    elif func_def.name in ("synStreamSetAffinity"):
                        out(Flow.call(entry, space.get_args(entry, ("streamAffinityMask",))))
                    elif func_def.name in ("synStreamDestroy", "synStreamSynchronize"):
                        out(Flow.call(entry, space.get_args(entry, ("streamHandle",))))
                    elif func_def.name == "synStreamWaitEvent":
                        out(Flow.call(entry, space.get_args(entry, ("streamHandle", "eventHandle"))))
                    elif func_def.name in ("synSectionDestroy"):
                        out(Flow.call(entry, space.get_args(entry, ("sectionHandle",))))
                    elif func_def.name == "synLaunch":
                        self._reference_match_launch(no, entry)
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

        if self.get_return_statement is not None:
            out(self.get_return_statement())


def main():
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
            renderer = Flow.SingleThreadedV2Renderer()
        elif args.test_flavour == "multithreaded":
            renderer = Flow.MultiThreadedRenderer()

        f.dump_c(renderer, device_address_limit=args.device_address_limit)
        renderer.bin_file_size = f.bin_file_size

        def out_src(*args, **kwargs):
            print(*args, **kwargs, file=src_file)

        renderer.render(out_src)


if __name__ == "__main__":
    main()
