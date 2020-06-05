import argparse
import textwrap
import sys
import logging
import collections
from gson_parsing import gson_iterator, syn_types


def human_readable_size(size, decimal_places=1):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f}{unit}"


def is_call(entry, func):
    return entry["name"] == "call" and entry["ph"] == "B" and entry["func"].name == func


def split_launch_id(recipe_info):
    name, addr = recipe_info
    # ".graph_dumps/habana_cluster_0_1_0-launch-0"
    split = name.find("-launch")
    return name[:split], name[split + 1 :], addr


class Graph:
    def __init__(self, create_entry):
        self.creat = create_entry
        self.compile = None
        self.nodes = []
        self.tensors = {}
        self.launches = {}
        self.recipes = {}
        self.name = None

    def __repr__(self):
        pers = list(self.get_persistent_tensors())

        return (
            f"Graph '{self.name}', workspace size is {human_readable_size(self.workspace_size)}, recipe size is {human_readable_size(self.recipe_size)}, "
            f"{len(self.nodes)} nodes, {len(self.tensors)} tensors ({len(pers)} persistent), launched {len(self.launches)} times"
        )

    def get_persistent_tensors(self):
        return (
            tensor_name
            for tensor_name, tensor in self.tensors.items()
            if "descriptor" in tensor and tensor["descriptor"]["args"]["fields"]["m_isPersistent"] == 1
        )

    def get_input_tensors(self):
        return (tensor_name for tensor_name, tensor in self.tensors.items() if tensor["src"][0]["id"][:3] == "ARG")

    def get_output_tensors(self):
        return (
            tensor_name
            for tensor_name, tensor in self.tensors.items()
            if any(dst[:3] == "RET" for dst in tensor["dst"])
        )

    def get_launches_duration(self, end_ts="end_ts"):
        return ((ln, l[end_ts] - l["ts"]) for ln, l in self.launches.items())

    def generate_port_nodes(self):
        arg_num, ret_num = 0, 0
        for ptr, tensor in self.tensors.items():
            if not "src" in tensor:
                node = {
                    "id": f"ARG{arg_num}",
                    "port": True,
                    "args": {"pInputsTensorList": [], "pOutputsTensorList": [ptr], "pGuid": "ARG"},
                }
                self.nodes.append(node)
                tensor["src"] = node, 0
                arg_num += 1

            if not tensor["dst"]:
                node = {
                    "id": f"RET{ret_num}",
                    "port": True,
                    "args": {"pInputsTensorList": [ptr], "pOutputsTensorList": [], "pGuid": "RET"},
                }
                self.nodes.append(node)
                tensor["dst"][node["id"]] = node, 0
                ret_num += 1

    def tensor_launch_address(self, launch_id):
        launch = self.launches[launch_id]["args"]
        launch_map = collections.defaultdict(lambda: "unpatched")
        launch_map.update(
            {
                name: addr
                for name, addr in zip(launch["enqueueInputTensorsInfo"][::2], launch["enqueueInputTensorsInfo"][1::2])
            }
        )
        launch_map.update(
            {
                name: addr
                for name, addr in zip(launch["enqueueOutputTensorsInfo"][::2], launch["enqueueOutputTensorsInfo"][1::2])
            }
        )
        return launch_map

    def tensor_shapes(self, ptrs):
        tensors = ((ptr, self.tensors[ptr]) for ptr in ptrs)
        tensor_shapes = (
            (
                tensor["name"],
                tuple(
                    tensor["descriptor"]["args"]["fields"]["m_sizes"][
                        : tensor["descriptor"]["args"]["fields"]["m_dims"]
                    ]
                ),
                syn_types[tensor["descriptor"]["args"]["fields"]["m_dataType"]],
            )
            if "tensor" in tensor["name"]
            else ("null", (0,), syn_types[0])  # some spatial_convolution inputs are called 'nullXXX'
            for tensor_ptr, tensor in tensors
        )
        return tensor_shapes

    def tensor_shape_map(self):
        tensor_shapes = self.tensor_shapes(self.tensors)
        return {tensor_name: str(tensor_size) for tensor_name, tensor_size, _ in tensor_shapes}

    def to_graphviz(self, tensor_labels={}):
        from graphviz import Digraph

        dot = Digraph(self.name, node_attr={"shape": "record"})
        self.generate_port_nodes()
        for node in self.nodes:
            if "port" in node:
                dot.node(node["id"], shape="circle")
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
            src, ouno = tensor["src"]
            logging.debug(
                f"tensor {tensor['name']} from {src['id']} has {len(tensor['dst'])} outputs : "
                + ", ".join(tensor["dst"].keys())
            )
            for did, (dst, inno) in tensor["dst"].items():
                s = src["id"] if "ARG" in src["id"] else f"{src['id']}:o{ouno}"
                d = dst["id"] if "RET" in dst["id"] else f"{dst['id']}:i{inno}"
                dot.edge(s, d, tensor["name"] + "\n" + tensor_labels.get(tensor["name"], ""))
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


class Log:
    def __init__(self, input_file=None, limit=None, **kwargs):
        self.tensors = {}
        self.graphs = {}
        self.ngraphs = {}
        nid = 0
        launch_limit, graph_limit = None, None
        if limit:
            if limit.find("launch") >= 0:
                launch_limit = limit
            else:
                graph_limit = int(limit)
        graph_no = 0
        crecipe = {}
        for no, entry in gson_iterator(input_file):
            try:
                if entry["name"] == "object" and entry["args"]["type"] == "synTensorDescriptorTr":
                    tdesc = entry
                    continue
                elif entry["name"] != "call" or entry["ph"] != "B":
                    continue
                result = entry["result"]
                args = entry["args"]

                if is_call(entry, "synTensorCreate"):
                    t = {"descriptor": tdesc, "creat": entry}
                    t["name"] = tdesc["args"]["fields"]["m_name"]
                    t["dst"] = dict()
                    tdesc = None
                    self.tensors[result["pTensor"]] = t

                if is_call(entry, "synGraphCreate"):
                    graph = Graph(entry)
                    self.graphs[result["pGraphHandle"]] = graph
                if is_call(entry, "synNodeCreate"):
                    entry["id"] = f"OP{nid}"
                    nid += 1
                    graph = self.graphs[args["graphHandle"]]
                    graph.nodes.append(entry)

                    null_input = f"null{no}"
                    for inum, ptr in enumerate(args["pInputsTensorList"]):
                        if ptr != "0":
                            t = self.tensors[ptr]
                        else:
                            t = {"name": null_input, "dst": dict()}
                            args["pInputsTensorList"][inum] = null_input
                            self.tensors[null_input] = t
                        t["dst"][entry["id"]] = entry, inum
                    for onum, ptr in enumerate(args["pOutputsTensorList"]):
                        assert (
                            not "src" in self.tensors[ptr]
                        ), f"tensor {self.tensors[ptr]['name']} at {ptr} already has source set to {self.tensors[ptr]['src']}"
                        self.tensors[ptr]["src"] = entry, onum
                    graph.tensors.update(
                        {
                            ptr: self.tensors[ptr].copy()
                            for collection in (args["pInputsTensorList"], args["pOutputsTensorList"])
                            for ptr in collection
                        }
                    )
                if is_call(entry, "synGraphCompile"):
                    graph = self.graphs[args["graphHandle"]]
                    graph.name = args["pRecipeName"][1:-1]
                    graph.compile = entry
                    crecipe[result["pRecipeHandle"]] = graph
                    self.ngraphs[graph.name] = graph

                if is_call(entry, "synRecipeGetSize"):
                    crecipe[args["recipeHandle"]].recipe_size = int(result["pRecipeSize"], 16)
                if is_call(entry, "synWorkspaceGetSize"):
                    crecipe[args["recipeHandle"]].workspace_size = int(result["pWorkspaceSize"], 16)
                if is_call(entry, "synRecipeUpload"):
                    graph_name, launch_name, addr = split_launch_id(args["pRecipeInfo"])
                    cgraph = crecipe[args["recipeHandle"]]
                    cgraph.name = graph_name
                    self.ngraphs[graph_name] = cgraph
                    graph_no += 1
                    if graph_limit and graph_no == graph_limit:
                        logging.info(f"parser reached graph limit of {graph_limit}")
                        return
                if is_call(entry, "synLaunch"):
                    claunch = entry
                    graph_name, launch_name, addr = split_launch_id(args["pRecipeInfo"])
                    self.ngraphs[graph_name].launches[launch_name] = entry
                    if launch_limit and launch_name == launch_limit:
                        logging.info(f"parser reached launch limit of {launch_limit}")
                        return
                if is_call(entry, "synStreamSynchronize"):
                    if args["streamHandle"] == "0x300000000":  # claunch['args']['streamHandle']:
                        claunch["sync_end"] = entry["end_ts"]
                        claunch = None
            except:
                logging.error(f"at line {no}")
                logging.error(entry)
                raise

    def first_graph(self):
        return next(iter(self.graphs.values()))

    def select_launches(self, recipe_id):
        """unpack recipe_id into enumeration of graphs. this is extracted to offer consistent behavior of --recipe_id:
            * when not provided select all graphs in log, but provide no launch
            * when given as graph id return iterator with one graph and no launch
            * when given as graph_G-launch-L return iterator with one graph and one launch"""
        if recipe_id:
            recipe_id = (recipe_id,)
        else:
            recipe_id = self.ngraphs

        for r in recipe_id:
            graph_id, launch_id = r, None
            tensor_labels = {}
            if "launch" in r:
                graph_id, launch_id, _ = split_launch_id((draw_id, None))
                launches = (launch_id,)
                yield (self.ngraphs[graph_id], launch_id)
            else:
                g = self.ngraphs[graph_id]
                yield (g, None)

    def cmd_list_nodes(self, recipe_id, **kwargs):
        for g, launch_id in self.select_launches(recipe_id):
            g.generate_port_nodes()
            print(g)
            g.print_nodes()

    def cmd_runtime_summary(self, recipe_id=None, output=None, **kwargs):
        import numpy as np

        for g, launch_id in self.select_launches(recipe_id):
            g.generate_port_nodes()
            launches = textwrap.wrap(
                ", ".join(
                    (f"{idx+1}:{launch_suffix}" for idx, (launch_suffix, syn_launch) in enumerate(g.launches.items()))
                )
            )
            print(g)
            l = np.asarray(list(dur for name, dur in g.get_launches_duration()))
            print("\tsynLaunch duration distribution avg,std: ", np.mean(l), np.std(l))
            l = np.asarray(list(dur for name, dur in g.get_launches_duration("sync_end")))
            print("\tsynLaunch+synStreamSynchronize duration distribution avg,std: ", np.mean(l), np.std(l))
            in_sizes = np.asarray(
                list(
                    np.prod((data_type[2],) + shape)
                    for name, shape, data_type in g.tensor_shapes(g.get_input_tensors())
                )
            )
            out_sizes = np.asarray(
                list(
                    np.prod((data_type[2],) + shape)
                    for name, shape, data_type in g.tensor_shapes(g.get_output_tensors())
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
            lines = textwrap.wrap(
                ", ".join(
                    (f"{idx+1}:{launch_suffix}" for idx, (launch_suffix, syn_launch) in enumerate(g.launches.items()))
                )
            )
            for n in lines:
                print("\t", n)
            lines = textwrap.wrap(g.get_nodes_summary())
            print("\tnodes:")
            for n in lines:
                print("\t", n)

    def cmd_list_launches(self, graph_id=None, **kwargs):
        print("list_launches", graph_id)
        g = self.ngraphs[graph_id]
        print(g)
        for idx, (launch_suffix, syn_launch) in enumerate(g.launches.items()):
            launch_id = g.name + "-" + launch_suffix
            print(f"\t{idx: >4}: at time {syn_launch['ts']} '{launch_id}'")

    def cmd_draw(self, recipe_id=None, output=None, **kwargs):
        for g, launch_id in self.select_launches(recipe_id):
            if launch_id:
                output = output if output else f"{g.name}-{launch_id}"
                logging.info("draw %s of %s into %s.svg", launch_id, graph_id, output)
                tensor_labels = g.tensor_launch_address(launch_id)
            else:
                output = output if output else g.name
                tensor_labels = g.tensor_shape_map()
                logging.info("draw %s into %s.svg", g.name, output)

            dot = g.to_graphviz(tensor_labels=tensor_labels)
            dot.format = "svg"
            dot.render(output)
            output = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="synapse log explorer")
    parser.add_argument("--input_file", default=".local.synapse_log.gson")
    parser.add_argument(
        "--limit",
        default=None,
        help="when given an int N finish after parsing N graphs. "
        "When given 'launch-L' stop parsing after given launch. Use this option to speed up working with large logs.",
    )
    parser.add_argument(
        "--recipe_id",
        default=None,
        help="graph or lauch identifier, e.g. .graph_dumps/cluster_x_y or .graph_dumps/cluster_x_y-launch-z ",
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
    subparser = subparsers.add_parser("list-nodes", help="prints nodes of graph along with input/output tensors")

    # no pickling for now, later-on automatically produce .local.synapse_log.pickle and accept it as input
    args = parser.parse_args()
    print(args)
    l = Log(**args.__dict__)
    func_name = "cmd_" + args.command.replace("-", "_")
    f = getattr(l, func_name)
    f(**args.__dict__)
