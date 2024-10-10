#!/usr/bin/env python
import argparse
import collections
import logging
import os
import pickle
import sys
import textwrap
from copy import deepcopy

from gson_parsing import descriptor_byte_size, gson_iterator, syn_types, zip_launch_info

log = logging.getLogger(__name__)


def human_readable_size(size, decimal_places=1):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def is_call(entry, func):
    return entry["name"] == "call" and entry["ph"] == "B" and entry["func"].name == func


def print_wrapped(indent, items, separator=", "):
    lines = textwrap.wrap(separator.join((str(item) for item in items)))
    print(indent.join(lines))


def split_launch_id(recipe_info):
    name, addr = recipe_info
    # ".graph_dumps/habana_cluster_0_1_0-launch-0"
    split = name.find("-launch")
    return name[:split], name[split + 1 :], addr


class Utils:
    @staticmethod
    def generate_makefile(files):
        exes = list(map(lambda x: x[:-4] + ".exe", files))
        bins = list(map(lambda x: x[:-4] + ".bin", files))
        lines = [
            "SRCS=" + " ".join(files),
            "CXX=g++",
            "CXXFLAGS=-I${HOME}/trees/npu-stack/pytorch-integration/pytorch_helpers/synapse_logger/logger_test/gtest_stub -I${HOME}/trees/npu-stack/synapse/include -I${HOME}/trees/npu-stack/tpc_kernels/include",
            "LIBS=${HOME}/builds/synapse_release_build/lib/libSynapse.so",
            ".PHONY: compile clean run help" "",
            "all: help",
            "",
            "help:",
            "\t@echo 'make compile         - compiles all recipe-dumpers'",
            "\t@echo 'make run             - executes compiled program'",
            "",
            "compile: " + " ".join(exes),
            "\t@echo Done",
            "",
            "run: compile",
            "\n".join(map(lambda p: f"\t./{p[0]} {p[1]}", zip(exes, bins))),
            "",
            "%.exe: %.cpp",
            "\t${CXX} ${CXXFLAGS} -o $@ $< ${LIBS}",
            "",
        ]

        return "\n".join(lines)

    def write_makefile(path, files):
        content = Utils.generate_makefile(files)
        with open(path, "w") as fd:
            fd.write(content)


class Tensor:
    def __init__(
        self,
        creation_event=None,
        custom_name="",
        *,
        descriptor=None,
        memory_section=None,
        is_null=False,
        new_api=True,
        is_const=False,
        is_persistent=False,
    ):
        self.events = [creation_event] if creation_event is not None else []
        self.is_persistent = is_persistent
        self.update_from_descriptor(descriptor)
        if memory_section:
            self.events.append(memory_section)
        if descriptor:
            self.events.append(descriptor)
        self.is_null = is_null
        self.is_const = is_const if not is_null else creation_event["func"].name == "synConstTensorCreate"
        if custom_name:
            self.name = custom_name
        else:
            self.name = creation_event["args"]["tensorName"] if new_api else descriptor["args"]["fields"]["m_name"]
        self.dst = dict()
        self.src = None
        self.is_output = False
        if is_null:
            self.is_const = False
            log.info(f"new tensor {self.name}")
            return
        const = "const " if self.is_const else ""
        log.info(f"{creation_event['ts']}: new {const}tensor {self.name} at {creation_event['result']['pTensor']}")

    @classmethod
    def Null(cls, name):
        return Tensor(custom_name=name, is_null=True)

    def set_output(self):
        self.is_output = True

    def update_geometry(self, geometry_entry, geometry_object):
        self.events.append(geometry_object)
        self.events.append(geometry_entry)

    def update_layout(self, layout_entry, layout_object):
        self.events.append(layout_object)
        self.events.append(layout_entry)

    def update_memory_section(self, entry, memory_section):
        self.events.append(memory_section)
        self.events.append(entry)

    @property
    def is_arg(self):
        return self.src and self.src[0]["id"][:3] == "ARG"

    @property
    def is_ret(self):
        return bool(any(dst[:3] == "RET" for dst in self.dst))

    def update_from_descriptor(self, desc):
        self.dims = desc["args"]["fields"]["m_dims"] if desc else []
        self.shape = tuple(desc["args"]["fields"]["m_sizes"][: self.dims]) if desc else tuple()
        self.byte_size = descriptor_byte_size(desc["args"]) if desc else 0
        self.syn_type = syn_types[desc["args"]["fields"]["m_dataType"] if desc else 0]

    def __repr__(self):
        if self.is_null:
            return self.name
        flags = (" ret" * self.is_ret) + (" arg" * self.is_arg) + (" persistent" * self.is_persistent)
        shape = "x".join((str(x) for x in self.shape))

        return f"{self.name} {shape} of {self.syn_type[1]}{flags}"


class Graph:
    def __init__(self, create_entry):
        self.creat = create_entry
        self.workspace_size = 0
        self.compile_entry = None
        self.nodes = []
        self.dependencies = []
        self.sections = {}
        self.tensors = {}
        self.launches = []
        self.recipes = {}
        self.node_params = []
        self.name = create_entry["result"]["pGraphHandle"] + " not compiled"
        log.info(f"{self.creat['ts']}: new graph handle {self.creat['result']['pGraphHandle']}")

    def __repr__(self):
        pers = list(self.get_persistent_tensors())

        return (
            f"Graph '{self.name}', workspace size is {human_readable_size(self.workspace_size)}, "
            f"{len(self.nodes)} nodes, {len(self.dependencies)} dependencies, {len(self.tensors)} tensors ({len(pers)} persistent), launched {len(self.launches)} times"
        )

    def events(self):
        yield self.creat
        yield self.compile_entry
        for p in self.node_params:
            yield p
        for s in self.sections.values():
            yield s
        for n in self.nodes:
            if n["args"]["pGuid"] not in ("RET", "ARG"):
                yield n
        for t in self.tensors.values():
            for te in t.events:
                yield te
        for d in self.dependencies:
            yield d

    def compile(self, entry):
        self.name = entry["args"]["pRecipeName"][1:-1]
        self.compile_entry = entry
        self._generate_port_nodes()
        log.info(f"compiled graph {self.name}")

    def register_launch(self, launch):
        self.launches.append(launch)
        log.info(f"graph {self.name} added launch {len(self.launches)-1}")
        return len(self.launches) - 1

    def add_section(self, entry):
        self.sections[entry["result"]["sectionHandle"]] = entry

    def add_dependency(self, entry):
        self.dependencies.append(entry)

    def add_node(self, no, entry, tensors, params=None):
        args = entry["args"]
        log.info(f"{entry['ts']}: new node {entry['args']['pGuid']} added to graph {entry['args']['graphHandle']}")

        self.nodes.append(entry)
        if params:
            self.node_params.append(params)
        null_input = f"null{no}"
        pInputsTensorList = deepcopy(args["pInputsTensorList"])
        for inum, ptr in enumerate(pInputsTensorList):
            if ptr != "0":
                t = tensors[ptr]
            else:
                t = Tensor.Null(null_input)
                pInputsTensorList[inum] = null_input
                tensors[null_input] = t
            t.dst[entry["id"]] = entry, inum
        for onum, ptr in enumerate(args["pOutputsTensorList"]):
            assert not tensors[
                ptr
            ].src, f"tensor {tensors[ptr].name} at {ptr} already has source set to {tensors[ptr].src}"
            tensors[ptr].set_output()
            tensors[ptr].src = entry, onum
        self.tensors.update(
            {
                ptr: tensors[ptr]  # deepcopy(tensors[ptr])
                for collection in (pInputsTensorList, args["pOutputsTensorList"])
                for ptr in collection
            }
        )

    def get_persistent_tensors(self):
        return (tensor_name for tensor_name, tensor in self.tensors.items() if tensor.is_persistent)

    def get_input_tensors_names(self):
        return (tensor.name for tensor in self.tensors.values() if tensor.is_arg)

    def get_output_tensors_names(self):
        return (tensor.name for tensor in self.tensors.values() if tensor.is_ret)

    def get_launches_duration(self, end_ts="end_ts"):
        return (l.duration for l in self.launches)

    def _generate_port_nodes(self):
        arg_num, ret_num = 0, 0

        def no_print(*args):
            pass

        pnt = no_print
        if "cluster_17_11_19" in self.name:
            pnt = print
        pnt(self.tensors)
        for ptr, tensor in self.tensors.items():
            if tensor.is_persistent or tensor.is_null:
                if not tensor.src:
                    node_name = "NULL" if tensor.is_null else "ARG"
                    node_name = f"{node_name}{arg_num}"
                    pnt(self.name, tensor.name, f"adding {node_name}")

                    node = {
                        "id": node_name,
                        "port": True,
                        "args": {"pInputsTensorList": [], "pOutputsTensorList": [ptr], "pGuid": "ARG"},
                    }
                    self.nodes.append(node)
                    tensor.src = node, 0
                    arg_num += 1
                elif not tensor.is_ret:
                    pnt(self.name, tensor.name, f"adding RET_{ret_num}")
                    node = {
                        "id": f"RET{ret_num}",
                        "port": True,
                        "args": {"pInputsTensorList": [ptr], "pOutputsTensorList": [], "pGuid": "RET"},
                    }
                    self.nodes.append(node)
                    tensor.dst[node["id"]] = node, 0
                    ret_num += 1

    def tensor_launch_address(self, launch_id):
        return self.launches[launch_id].get_patching_table(to_int=False)

    def tensor_shapes(self, ptrs):
        tensor_shapes = (
            (
                (tensor.name, tensor.shape, tensor.syn_type)
                if "tensor" in tensor.name
                else ("null", (0,), syn_types[0])
            )  # some spatial_convolution inputs are called 'nullXXX'
            for tensor in self.tensors.values()
        )
        return tensor_shapes

    def tensor_shape_map(self):
        tensor_shapes = self.tensor_shapes(self.tensors)
        return {tensor_name: str(tensor_size) for tensor_name, tensor_size, _ in tensor_shapes}

    def to_graphviz(self, tensor_labels={}):
        from graphviz import Digraph

        dot = Digraph(self.name, node_attr={"shape": "record"})
        for node in self.nodes:
            if "port" in node:
                shape = "point" if node["id"].startswith("NULL") else "circle"
                dot.node(node["id"], shape=shape)
            else:
                colspan = max(node["args"]["numberInputs"], node["args"]["numberOutputs"])
                ins = " ".join(f"<td port='i{i}' style='rounded'>{i}</td>" for i in range(node["args"]["numberInputs"]))
                if ins:
                    ins = f"<tr><td><table border='0' cellborder='1'><tr>{ins}</tr></table></td></tr>"
                ous = " ".join(
                    f"<td port='o{i}' style='rounded'>{i}</td>" for i in range(node["args"]["numberOutputs"])
                )
                if ous:
                    ous = f"<tr><td><table border='0' cellborder='1'><tr>{ous}</tr></table></td></tr>"
                label = f"<\n<table border='1' cellborder='0'>{ins}<tr><td>{node['args']['pGuid'][1:-1]}</td></tr>{ous}</table>\n>"
                dot.node(node["id"], label, shape="plaintext")

        for ptr, tensor in self.tensors.items():
            src, ouno = tensor.src
            log.debug(
                f"tensor {tensor.name} from {src['id']} has {len(tensor.dst)} outputs : " + ", ".join(tensor.dst.keys())
            )
            for did, (dst, inno) in tensor.dst.items():
                s = src["id"] if src["id"].startswith("ARG") or src["id"].startswith("NULL") else f"{src['id']}:o{ouno}"
                d = dst["id"] if dst["id"].startswith("RET") else f"{dst['id']}:i{inno}"
                dot.edge(s, d, tensor.name + "\n" + tensor_labels.get(tensor.name, ""))
        dot.format = "svg"
        return dot

    def get_nodes_summary(self):  # , tensor_map):
        from collections import Counter

        guids = Counter(n["args"]["pGuid"] for n in self.nodes)
        return ", ".join(f"{no} {guid[1:-1]}" for guid, no in guids.items())

    def print_nodes(self):  # , tensor_map):
        for n in self.nodes:
            in_tensors = list(
                "x".join(str(s) for s in shape)
                for name, shape, syn_type in self.tensor_shapes(n["args"]["pInputsTensorList"])
            )
            out_tensors = list(
                "x".join(str(s) for s in shape)
                for name, shape, syn_type in self.tensor_shapes(n["args"]["pOutputsTensorList"])
            )
            print(n["args"]["pGuid"], " ".join(in_tensors) + " ==> " + " ".join(out_tensors))


class Durations:
    def __init__(self, input_file=None, limit=None, **kwargs):
        self.durations = list()
        nid = 0
        line_limit = int(limit)
        previous = 0
        for no, entry in gson_iterator(input_file):
            try:
                if line_limit and no == line_limit:
                    log.info(f"parser reached line limit of {line_limit}")
                    return
                if entry["name"] == "call" and entry["ph"] == "B":
                    if not previous:
                        previous = entry["end_ts"]
                    else:
                        self.durations.append((entry["ts"] - previous, "gap", no))
                        previous = entry["end_ts"]
                    d = entry["end_ts"] - entry["ts"]
                    arg = entry["func"].name
                    if entry["func"].name == "synLaunch":
                        recipe = entry["args"]["pRecipeInfo"][0]
                        recipe = recipe[recipe.rfind("/") + 1 : recipe.rfind("-launch")]
                        arg = arg + " " + recipe
                    self.durations.append((d, arg, no))

            except:
                log.error(f"at line {no}")
                log.error(entry)
                raise


class MemBinding(collections.namedtuple("MemBinding", ["agent", "name", "area"])):
    def __repr__(self):
        name = f" {self.name}" if self.name else ""
        return f"{self.agent}{name} wrote {self.area[0]:08x}:{self.area[0]+self.area[1]:08x}"


def area_overlaps(a1, a2):
    return a1[0] < a2[0] + a2[1] and a2[0] < a1[0] + a1[1]


class MemMap:
    """Models device memory space in synapse-log time.

    Upon each log entry of a call that writes to a device memory the map
    remembers that area starting at some address and spanning some size was
    produced by this operation. Then ar any time it is possible to query
    MemMap about producers of data that now is referenced. Subsequent
    producers may partially or completely overwrite previous writes which
    is also modeled. E.g if A wrote memory producing [   AAAAAA   ] and
    then B writes in the middle, MemMap might split area writen by A and
    contain something like [   AABBAA   ]"""

    class Agent:
        def __init__(self, entry):
            self.sources, self.destinations = list(), list()
            self.entry = entry
            self.ts = entry["ts"]

        def __repr__(self):
            return f"MemMap.Agent {self.entry['name']} at {self.ts}"

        def set_dst(self, source_ptr, source_size, agent, input_name):
            self.destinations.append(MemBinding(agent, input_name, (source_ptr, source_size)))

        def set_src(self, source_ptr, source_size, agent, destination_name):
            self.sources.append(MemBinding(agent, destination_name, (source_ptr, source_size)))

        def source_by_area(self, pointer, size):
            return sorted((s for s in self.sources if area_overlaps((pointer, size), s.area)), key=lambda s: s.area[0])

    def __init__(self):
        self.mem = collections.OrderedDict()
        self.watch = None  # (0xbd97c100, 0xbdb04100-0xbd97c100)
        self.watch_end = 1582655668496039

    def collisions(self, addr, size, trim=False):
        """Generates all sources that produced a given memory area.
        Resulting generator produces 3-tuples of producer address, size and log
        entry.

            @param: addr beginning of the area of interest

            @param: size size of the area

            @param: trim control if collisions should be trimmed to only addr and size.
        """
        for iaddr, (isize, value) in self.mem.items():
            # print(f"addr {addr:x} < iaddr + isize {iaddr+isize:x} and iaddr {iaddr:x} < addr + size {addr+size:x}")
            if area_overlaps((addr, size), (iaddr, isize)):
                if trim:
                    start, end = max(iaddr, addr), min(iaddr + isize, addr + size)
                    yield start, end - start, value
                else:
                    yield iaddr, isize, value

    def add_destination(self, device_ptr, size, agent, input_name):
        """Register consumer of a memory area within all participating producers.
        In general, a buffer might have been partially prepared by multiple
        sources in which case each source is bound to destination. This call
        is producing bidirectional connections between consumer and sources.
        E.g. a synLaunch is registered as a destination for all producers
        that _participated_ in any input tensor.

            @param: device_ptr starting address of consumed buffer

            @param: size size of consumed buffer.

            @param: agent consumer entry. In course of this call entry key
            'sources' is appended one or more entries that are spatially
            separate. I.e. there may have been multiple operations that
            produced parts of an input tensor and whole input tensor area must
            be covered but it's a bug in MemMap if sources overlap.

            @param: input_name a name associated with the buffer.  Tensor name
            in case of synLaunch
        """
        assert isinstance(agent, MemMap.Agent), "wrong agent type"
        for source_ptr, source_size, (source, source_name) in self.collisions(device_ptr, size, trim=True):
            # assert len(ptr_events) == 1, f"expected one producer of location {device_ptr:x} {size:x}, got " + str(list(f"{a:x}:{a+s:x}" for a,s,e in ptr_events))

            source.set_dst(source_ptr, source_size, agent, input_name)
            agent.set_src(source_ptr, source_size, source, source_name)

    def fragment_count(self):
        return len(self.mem)

    def update(self, addr, size, agent, name):
        """Updates map with new producer agent
        @param: name identifier of the map update, e.g. tensor name. Agent may update multiple areas, each identified by 'name'.
        """

        def get_watch_collisions():
            return list(sorted(self.collisions(*self.watch, trim=True), key=lambda x: x[0])) if self.watch else None

        def print_collision(colls):
            for idx, (adr, s, ag) in enumerate(colls):
                print(f"    {idx:<3} {adr:x}:{adr+s:x} {ag}")

        assert isinstance(agent, MemMap.Agent), "wrong agent type"
        end = addr + size
        log.debug(f"{len(self.mem):<4} update {addr:x}:{end:x} to {agent} {name}")
        c = list(self.collisions(addr, size))
        watch_collisions_before = get_watch_collisions()
        for iaddr, isize, old_agent in c:

            iend = iaddr + isize
            e, entry = self.mem[iaddr]
            # log.debug(f"     deleting {iaddr:x}:{iaddr+self.mem[iaddr][0]:x} {self.mem[iaddr]}")
            self.mem.pop(iaddr)
            naddr, nend = iaddr, iend  # new start and end after trimming
            if iaddr >= addr and iaddr < end:  # overlap starts in new area, trim old start to new end
                naddr = end
            if iend >= addr and iend <= end:  # overlap ends in the new area, do trim old end to new start
                nend = addr
            if addr > iaddr and end < iend:  # new area lands inside old area, break old into two
                log.debug(
                    f"     split {iaddr:x}:{iaddr+isize:x} {old_agent} into {naddr:x}:{addr:x} and {end:x}:{nend:x}"
                )
                self.mem[naddr] = (addr - naddr, entry)
                self.mem[end] = (nend - end, entry)
            elif nend > naddr:
                log.debug(f"     cut {iaddr:x}:{iaddr+isize:x} {old_agent} into {naddr:x}:{nend:x}")
                self.mem[naddr] = (nend - naddr, entry)
            else:
                log.debug(f"     rem {iaddr:x}:{iaddr+isize:x} {old_agent}")
        self.mem[addr] = (size, (agent, name))
        if self.watch_end and agent.ts >= self.watch_end:
            self.watch = None
        if not self.watch:
            return
        watch_collisions_after = get_watch_collisions()
        for idx, ((b_ptr, b_size, b_src), (a_ptr, a_size, a_src)) in enumerate(
            zip(watch_collisions_before, watch_collisions_after)
        ):
            if b_ptr != a_ptr or b_size != a_size or b_src != a_src:
                print(
                    f"watch of {self.watch[0]:x}:{self.watch[0]+self.watch[1]:x} changed at {idx} due to {addr:x}:{addr+size:x}",
                    agent,
                    name,
                )
                print_collision(watch_collisions_after)
                break
        else:
            if len(watch_collisions_before) != len(watch_collisions_after):
                print(
                    f"watch of {self.watch[0]:x}:{self.watch[0]+self.watch[1]:x} changed due to  {addr:x}:{addr+size:x}",
                    agent,
                    name,
                )
                print_collision(watch_collisions_after)

    def size_histogram(self):
        import matplotlib as mp

        mp.histogram((size for size, _ in self.mem.values()))


class Memcpy(MemMap.Agent):
    def __init__(self, log, entry):
        super().__init__(entry)
        if entry["args"]["direction"] == 0:
            self.direction = "H2D"
            device_ptr = int(entry["args"]["dst"][2:], 16)
            log.memory.update(device_ptr, entry["args"]["size"], self, "")
        else:
            self.direction = "D2H"
            device_ptr = int(entry["args"]["src"][2:], 16)
            log.memory.add_destination(device_ptr, entry["args"]["size"], self, "")

    def __repr__(self):
        return f"{self.entry['ts']} {self.direction}"


class Launch(MemMap.Agent):
    UNPATCHED = "unpatched"

    def __init__(self, log, entry, graph):
        super().__init__(entry)
        args = entry["args"]
        self.end_ts = self.ts
        self.graph = graph
        self.launch_no = self.graph.register_launch(self)
        self.inputs, self.outputs = collections.OrderedDict(), collections.OrderedDict()
        patching_table = self.get_patching_table()

        for name in (name for name in self.graph.get_input_tensors_names() if name in patching_table):
            device_ptr = patching_table[name]
            tdef = log.ntensors[name]
            size = tdef.byte_size
            self.inputs[name] = (device_ptr, size)
            log.memory.add_destination(device_ptr, size, self, name)

        for name in (name for name in self.graph.get_output_tensors_names() if name in patching_table):
            device_ptr = patching_table[name]
            tdef = log.ntensors[name]
            # print("<", tdef['descriptor']['args']['fields']['m_name'], name,">", end=" ")
            size = tdef.byte_size
            self.outputs[name] = (device_ptr, size)
            log.memory.update(device_ptr, size, self, name)

    def get_patching_table(self, to_int=True):
        convert = (lambda x: int(x[2:], 16)) if to_int else (lambda x: x)
        patching_table = collections.defaultdict(lambda: "unpatched")
        patching_table.update({name: convert(addr) for name, addr in zip_launch_info(self.entry)})
        return patching_table

    def get_sources(self):
        for name, (pointer, size) in self.inputs.items():
            yield name, pointer, size, list(self.source_by_area(pointer, size))

    @property
    def graph_name(self):
        return self.graph.name

    @property
    def duration(self):
        return self.end_ts - self.ts

    def set_completion(self, entry):
        self.end_ts = entry["args"]["end_ts"]

    def __repr__(self):
        return f"{self.entry['ts']} {self.graph.name}/{self.launch_no}"


class Log:
    def __init__(self, input_file=None, limit=None, **kwargs):
        self.tensors = {}
        self.ntensors = {}
        self.graphs = {}
        self.ngraphs = {}
        self.memory = MemMap()
        self.launches = list()
        self.rawlogs = []
        nid = 0
        launch_limit, graph_limit, line_limit = None, None, None
        if limit:
            if limit.find("launch") >= 0:
                launch_limit = limit
            elif limit.find("graph-") == 0:
                graph_limit = int(limit[6:])
            else:
                line_limit = int(limit)
        graph_no = 0
        stream_state = collections.defaultdict(list)
        crecipe = {}
        event_map = {}  # map recorded event to stream
        memory_sections = {}
        objects = {"0": None}
        for no, entry in gson_iterator(input_file, end_on_error=False):
            self.rawlogs.append((no, entry))
            try:
                if (
                    "result" in entry
                    and "args" in entry["result"]
                    and "status" in entry["result"]["args"]
                    and entry["result"]["args"]["status"] == "incomplete"
                ):
                    # calls are incomplete when trace abruptly ends due to a crash
                    log.warn(f"stopping json parsing on an incomplete call at line {no} {entry}")
                    break
                if line_limit and no == line_limit:
                    log.info(f"parser reached line limit of {line_limit}")
                    return

                if entry["name"] == "object":
                    objects[entry["args"]["at"]] = entry
                    log.debug(f"object at {entry['args']['at']}")
                    if entry["args"]["type"] == "synTensorDescriptor":
                        tdesc = entry
                    continue
                elif entry["name"] != "call" or entry["ph"] != "B":
                    continue
                result = entry["result"]
                args = entry["args"]

                if is_call(entry, "synMemCopyAsync"):

                    agent = Memcpy(self, entry)
                    stream_state[entry["args"]["streamHandle"]].append(entry)

                if is_call(entry, "synSectionCreate") and entry["ph"] == "B":
                    memory_sections[entry["result"]["sectionHandle"]] = entry
                    graph = self.graphs[args["graph"]]
                    graph.add_section(entry)
                if is_call(entry, "synTensorAssignToSection") and entry["ph"] == "B":
                    tensor = entry["args"]["tensor"]
                    self.tensors[tensor].update_memory_section(entry, memory_sections[entry["args"]["section"]])
                if is_call(entry, "synTensorCreate") or is_call(entry, "synConstTensorCreate"):
                    assert entry["args"]["descriptor"] == tdesc["args"]["at"]
                    section = None
                    if "pSectionHandle" in entry["args"] and entry["args"]["pSectionHandle"] != "0":
                        section = memory_sections[entry["args"]["pSectionHandle"]]
                    t = Tensor(entry, custom_name="", descriptor=tdesc, memory_section=section, new_api=False)
                    tdesc = None
                    self.tensors[result["pTensor"]] = t
                    self.ntensors[t.name] = t
                if is_call(entry, "synTensorHandleCreate"):
                    t = Tensor(entry, "")
                    self.tensors[result["pTensor"]] = t
                    self.ntensors[t.name] = t
                if is_call(entry, "synTensorSetGeometry"):
                    geometry_object = objects[entry["args"]["geometry"]]
                    tensor = entry["args"]["tensor"]
                    self.tensors[tensor].update_geometry(entry, geometry_object)
                if is_call(entry, "synGraphCreate"):
                    graph = Graph(entry)
                    self.graphs[result["pGraphHandle"]] = graph
                if is_call(entry, "synNodeDependencySet"):
                    graph = self.graphs[args["graphHandle"]]
                    graph.add_dependency(entry)
                if is_call(entry, "synNodeCreate") or is_call(entry, "synNodeCreateWithId"):
                    entry["id"] = f"OP{nid}"
                    nid += 1
                    graph = self.graphs[args["graphHandle"]]
                    graph.add_node(no, entry, self.tensors, objects[entry["args"]["pUserParams"]])
                if is_call(entry, "synGraphCompile"):
                    graph = self.graphs[args["graphHandle"]]
                    graph.compile(entry)
                    if "pRecipeHandle" in result:  # false if call never ended
                        crecipe[result["pRecipeHandle"]] = graph
                    self.ngraphs[graph.name] = graph

                if is_call(entry, "synWorkspaceGetSize"):
                    crecipe[args["recipeHandle"]].workspace_size = int(result["pWorkspaceSize"], 16)
                if is_call(entry, "synLaunch"):
                    graph = crecipe[args["pRecipehandle"]]
                    launch = Launch(self, entry, graph)

                    graph_no += 1
                    if graph_limit and graph_no == graph_limit:
                        log.info(f"parser reached graph limit of {graph_limit}")
                        return

                    self.launches.append(launch)
                    stream_state[entry["args"]["streamHandle"]].append(entry)

                if is_call(entry, "synEventRecord"):
                    stream_state[entry["args"]["streamHandle"]].append(entry)
                    event_map[entry["args"]["eventHandle"]] = entry["args"]["streamHandle"]
                if is_call(entry, "synEventSynchronize"):
                    stream = event_map[entry["args"]["eventHandle"]]
                    completed = 0
                    for e in stream_state[stream]:
                        if "launch" in e:
                            e["launch"].set_completion(entry)
                        if is_call(e, "synEventRecord") and e["args"]["eventHandle"] == entry["args"]["eventHandle"]:
                            break
                        completed += 1

                    stream_state[stream] = stream_state[stream][completed + 1 :]

                if is_call(entry, "synStreamSynchronize"):
                    pass
            except:
                log.error(f"at line {no}")
                log.error(entry)
                raise

    def first_graph(self):
        return next(iter(self.graphs.values()))

    def select_launches(self, recipe_id):
        """unpack recipe_id into enumeration of graphs. this is extracted to offer consistent behavior of --recipe_id:
        * when not provided select all graphs in log, but provide no launch
        * when given as graph id return iterator with one graph and no launch
        * when given as graph_G/L return iterator with one graph and one launch
        * when given as graph_G/* return iterator with one graph and all launches"""
        if recipe_id:
            recipe_id = (recipe_id,)
        else:
            recipe_id = self.ngraphs

        for r in recipe_id:
            graph_id, launch_id = r, None
            tensor_labels = {}
            launch_id = None
            try:
                pos = r.rfind("/")
                launch_id = r[pos + 1 :]
                if launch_id != "*":
                    launch_id = int(r[pos + 1 :])
                graph_id = r[:pos]
            except:
                pass
            if launch_id == "*":
                graph = self.ngraphs[graph_id]
                for l in graph.launches:
                    yield (graph, l.launch_no)
            elif isinstance(launch_id, int):
                launches = (launch_id,)
                yield (self.ngraphs[graph_id], launch_id)
            else:
                g = self.ngraphs[graph_id]
                yield (g, None)

    def write_graph_test(self, graph: Graph, output, computation_result=None):
        from gson2test import Flow

        def collect_tss():
            tss = set([e["ts"] for e in graph.events()])
            tss |= set([e["end_ts"] for e in graph.events() if "end_ts" in e])
            return tss

        test_op_list = set(
            [
                "synStreamCreateGeneric",
                "synStreamDestroy",
                # "synStreamSynchronize",
                # "synWorkspaceGetSize",
                "synConfigurationSet",
                "synDeviceMalloc",
                "synDeviceGetMemoryInfo",
                "synDeviceFree",
                "synDeviceRelease",
                "synDestroy",
                "synInitialize",
                "synDeviceAcquireByDeviceType",
            ]
        )

        selected_tss = collect_tss()

        def pred(elem):
            entry = elem[1]
            if entry["ts"] in selected_tss:
                return True
            if "func" in entry and entry["func"].name in test_op_list:
                return True
            return False

        flow = Flow(filter(pred, self.rawlogs), computation_result=computation_result)
        with open(output, "w") as src_file:
            log.info(f"Writing to {output}")

            if computation_result == Flow.ComputationResult.LAST_COMPILED_RECIPE:
                disable_bin_file = True
                Renderer = Flow.RecipeDumpingRenderer
            elif computation_result is None:
                disable_bin_file = False
                Renderer = Flow.SingleThreadedV2Renderer

            renderer = Renderer()
            flow.configure_renderer(renderer)
            flow.dump_c(renderer)
            renderer.bin_file_size = flow.bin_file_size
            renderer.disable_bin_file = disable_bin_file

            def out_src(*args, **kwargs):
                print(*args, **kwargs, file=src_file)

            renderer.render(out_src)

    def cmd_extract_all_graphs(self, output_dir, **kwargs):
        def handle_dir():
            if os.path.exists(output_dir):
                if len(os.listdir(output_dir)) != 0:
                    raise Exception(f"output directory {output_dir} exists and is not empty")
            else:
                os.makedirs(output_dir)

        def convert_to_path(graph_name: str):
            fname = graph_name.replace(".", "_").replace("/", "__")
            return f"{output_dir}/{fname}.cpp"

        def dump_graphs_with_makefile():
            from gson2test import Flow

            graphs = self.ngraphs.values()
            paths = list(map(convert_to_path, self.ngraphs.keys()))

            for fname, graph in zip(paths, graphs):
                self.write_graph_test(graph, fname, Flow.ComputationResult.LAST_COMPILED_RECIPE)

            srcs = list(map(os.path.basename, paths))
            mkpath = f"{output_dir}/Makefile"

            Utils.write_makefile(mkpath, srcs)

        handle_dir()
        dump_graphs_with_makefile()

    def cmd_write_graph_test(self, recipe_id, **kwargs):
        for g, launch_id in self.select_launches(recipe_id):
            return self.write_graph_test(g, "test.cxx")

    def cmd_list_nodes(self, recipe_id, **kwargs):
        for g, launch_id in self.select_launches(recipe_id):
            print(g)
            g.print_nodes()

    def cmd_runtime_summary(self, recipe_id=None, output=None, **kwargs):
        import numpy as np

        for g, launch_id in self.select_launches(recipe_id):
            launches = textwrap.wrap(", ".join((f"{idx+1}:{syn_launch}" for idx, syn_launch in enumerate(g.launches))))
            print(g)
            l = np.asarray(list(g.get_launches_duration()))
            print("\tsynLaunch duration distribution avg,std: ", np.mean(l), np.std(l))
            in_sizes = np.asarray(
                list(
                    np.prod((data_type[2],) + shape)
                    for name, shape, data_type in g.tensor_shapes(g.get_input_tensors_names())
                )
            )
            out_sizes = np.asarray(
                list(
                    np.prod((data_type[2],) + shape)
                    for name, shape, data_type in g.tensor_shapes(g.get_output_tensors_names())
                )
            )
            in_count, out_count = in_sizes.size, out_sizes.size
            in_total, out_total = tuple(human_readable_size(np.sum(x)) for x in (in_sizes, out_sizes))
            print(
                f"\t{len(g.tensors)} tensors : {len(g.tensors)-in_count-out_count} internal,  {in_count} inputs totaling {in_total}, {out_count} outputs totaling {out_total},"
            )

    def cmd_list_graphs(self, *args, **kwargs):
        for g in self.graphs.values():
            print(g)
            lines = textwrap.wrap(", ".join((f"{idx+1}:{syn_launch}" for idx, syn_launch in enumerate(g.launches))))
            for n in lines:
                print("\t", n)
            lines = textwrap.wrap(g.get_nodes_summary())
            print("\tnodes:")
            for n in lines:
                print("\t", n)

    def cmd_list_launches(self, graph_id=None, **kwargs):
        if not graph_id:
            graph_id = next(iter(self.ngraphs))
        print("list_launches", graph_id)
        g = self.ngraphs[graph_id]
        print(g)
        for idx, syn_launch in enumerate(g.launches):
            launch_id = f"{g.name}-{idx}"
            print(f"\t{idx: >4}: at time {syn_launch.ts} '{launch_id}'")

    def cmd_draw(self, recipe_id=None, output=None, **kwargs):
        results = []
        for g, launch_id in self.select_launches(recipe_id):
            if launch_id:
                output = output if output else f"{g.name}-{launch_id}"
                log.info("draw %s of %s into %s.svg", launch_id, recipe_id, output)
                tensor_labels = g.tensor_launch_address(launch_id)
            else:
                output = output if output else g.name
                tensor_labels = g.tensor_shape_map()
                log.info("draw %s into %s.svg", g.name, output)

            dot = g.to_graphviz(tensor_labels=tensor_labels)
            dot.format = "svg"
            results.append(dot)
            if output != "-":
                dot.render(output)
            output = None
        return results

    def cmd_timeline(self, recipe_id=None, verbose=True, **kwargs):
        if recipe_id:
            launches = (g.launches[launch_id] for g, launch_id in self.select_launches(recipe_id))
        else:
            launches = self.launches
        for launch in launches:
            sources = launch.sources
            dma_sources = list(s for s in sources if isinstance(s.agent, Memcpy))
            compute_sources = list(s for s in sources if isinstance(s.agent, Launch))
            destinations = launch.destinations
            dma_destinations = list(s for s in destinations if isinstance(s.agent, Memcpy))
            compute_destinations = list(s for s in destinations if isinstance(s.agent, Launch))
            total_inputs, total_outputs = len(launch.inputs), len(launch.outputs)
            if verbose:
                print(
                    launch,
                    launch.duration,
                    f"dma_sources {len(dma_sources)}, compute_sources {len(compute_sources)}, dma_dests {len(dma_destinations)}, compute_dests {len(compute_destinations)}",
                )
                print(f"  {len(launch.inputs)} inputs:")
                for name, pointer, size, sources in launch.get_sources():
                    print(f"    {name} {pointer:x}:{pointer+size:x} from {len(sources)} sources:")
                    for s in sources:
                        print(f"      {s.area[0]:08x}:{s.area[0]+s.area[1]:08x} from {s.agent}/{s.name}")
                print(f"  {len(launch.outputs)} outputs:")
                for name, (pointer, size) in launch.outputs.items():
                    print(f"    {name} {pointer:x}:{pointer+size:x}")

            else:
                print(
                    launch,
                    launch.duration,
                    f"dma_sources {len(dma_sources)}, compute_sources {len(compute_sources)}, dma_dests {len(dma_destinations)}, compute_dests {len(compute_destinations)}",
                )
            if launch.ts in (235406391.0, 340392135.0):
                for no, (t, tdef) in enumerate(launch.inputs.items()):
                    print(f"  {no:04} {t} {tdef[0]:08x}:{tdef[0]+tdef[1]:08x}")
                    for s in launch.sources:
                        if area_overlaps(s.area, tdef):
                            print(f"    T-{launch.ts-s.agent.ts}: {s}")


if __name__ == "__main__":
    sys.setrecursionlimit(3000)
    parser = argparse.ArgumentParser(description="synapse log explorer")
    parser.add_argument(
        "--input_file",
        default=".local.synapse_log.json",
        help="input file. May be either gson log file, or gzipped log file or a *.logpickle file created before with --pickle",
    )
    verbosity_map = [
        (logging.WARNING, logging.WARNING),
        (logging.INFO, logging.WARNING),
        (logging.INFO, logging.INFO),
        (logging.DEBUG, logging.INFO),
        (logging.DEBUG, logging.DEBUG),
    ]
    max_verbosity = "v" * (len(verbosity_map) + 1)
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help=f"produce diagnostic output, maximum level is -{max_verbosity}",
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="when given int N finish after N lines, when given graph-G finish after parsing G graphs. "
        "When given 'launch-L' stop parsing after given launch. Use this option to speed up working with large logs.",
    )
    parser.add_argument(
        "--pickle",
        default=None,
        help="when given <value> write preprocessed log to <value>.logpickle that can be later passed as --input_file",
    )
    parser.add_argument(
        "--recipe_id",
        default=None,
        help="graph or lauch identifier, e.g. .graph_dumps/cluster_x_y or .graph_dumps/cluster_x_y/z ",
    )
    subparsers = parser.add_subparsers(dest="command", description="command to execute")
    p_list_graphs = subparsers.add_parser("list-graphs", help="writes down summary of all graphs present in the log")
    p_list_launches = subparsers.add_parser(
        "list-launches", help="prints information about launches of a particular graph"
    )
    p_list_launches.add_argument("graph_id", help="graph identifier, e.g. .graph_dumps/cluster_x_y")
    p_draw_launch = subparsers.add_parser(
        "draw",
        help="creates svg with directed graph diagram of a graph, optionally with tensor addresses of particular launch",
    )
    p_runtime_summary = subparsers.add_parser(
        "runtime-summary", help="prints summary of runtime behavior of each graph"
    )
    p_extrat_all_graphs = subparsers.add_parser("extract-all-graphs", help="produce API-level graphs")
    p_extrat_all_graphs.add_argument("output_dir", help="output directory")
    subparser = subparsers.add_parser("list-nodes", help="prints nodes of graph along with input/output tensors")
    subparser = subparsers.add_parser("write-graph-test", help="produce API-level test ofgraph compilation")
    subparser = subparsers.add_parser("timeline", help="prints launches timeline")

    # no pickling for now, later-on automatically produce .local.synapse_log.pickle and accept it as input
    args = parser.parse_args()

    master_verbosity, lib_verbosity = verbosity_map[args.verbose]
    logging.basicConfig(level=master_verbosity)
    logging.getLogger("synapse_logger").setLevel(lib_verbosity)
    log.debug(str(args))
    if not args.command:
        args.command = "list-launches"
    if args.input_file.endswith(".logpickle"):
        l = pickle.load(open(args.input_file, "rb"))
    else:
        l = Log(**args.__dict__)
    if args.pickle:
        pickle.dump(l, open(args.pickle + ".logpickle", "wb"))
    func_name = "cmd_" + args.command.replace("-", "_")
    f = getattr(l, func_name)
    f(**args.__dict__)
