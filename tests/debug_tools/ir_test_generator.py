"""
Utility that does code generation using the Log IR
"""

import queue
import random as r
import re

from ir_graph_generator import *
from ir_graph_visualizer import *


class SubGraphTestCaseException(BaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


file_name = f"TestAutoGen_{r.randint(0, 20000)}"
test_case = "TestCase"
NO_LOG_OP = set()
REMOVE_TEMP = {"_fused_dropout"}  # does not produce tensors
is_tuple = {"max_pool2d_with_indices"}


def insert_char(s, idx, c):
    return s[:idx] + c + s[idx:]


def process_info_line(info_line):
    """returns  [(time_stamp, op_name, args, info_line)]"""
    #  read line by line, each line has the format [assumption] :
    #  [<time>][<Log_Name>][<log_level> ] <extra_info> <op_name> : (<arg_name>=<dtype>?<value> )+
    try:
        time_info, op_info, str_args = re.search(r"(\[[\d\.\:]+\]).*(\s.*\s\:)(.*)", info_line).groups()
        time_info, op_info, str_args = time_info.strip()[1:-1], op_info.strip()[:-2], str_args.strip()
    except AttributeError:
        time_info, op_info, str_args = re.search(r"(\[[\d\.\:]+\]).*(\s.*\s?\:)(.*)", info_line).groups()
        time_info, op_info, str_args = time_info.strip()[1:-1], op_info.strip()[:-1], str_args.strip()
        # correcting logs in places
        if "dropout" in op_info:
            str_args = insert_char(str_args, str_args.index("train"), " ")
            str_args = insert_char(str_args, str_args.index("p"), " ")
    if "_reshape_alias" in op_info:
        str_args = insert_char(str_args, str_args.index("stride") + len("stride"), "=")
    #  extract arguments as a pair list
    var_value_list = []
    key, value = "", ""
    brack_open, key_added = False, False
    for c in str_args:
        if not key_added and c != "=":
            key += c
        elif not key_added and c == "=":
            key_added = True
            var_value_list.append((key, []))
            key = ""
        elif key_added and not brack_open and c == " ":
            key_added = False
            var_value_list[-1][-1].append(value)
            value = ""
        elif key_added and c == "[":
            if len(value) > 0:
                var_value_list[-1][-1].append(value)
            value = "{"
            brack_open = True
        elif key_added and c == "]":
            value += "}"
            brack_open = False
        else:
            value += c
    var_value_list[-1][-1].append(value)
    extra = op_info.split(" ")
    if len(extra) > 1:
        op_info = extra[-1]
    return (time_info, op_info.lower(), var_value_list, info_line)


def process_info_lines(info_lines, graph):
    """generate a list of [(time_stamp, op_name, args, info_line)] from info lines until op with index from <graph = (adj_list, op with index)>"""
    root = graph[-1]
    info_list, c = {}, 0
    s = root[0].split("::")[-1]
    for info_line in info_lines:
        info = None
        try:
            info = process_info_line(info_line)
        except Exception as e:
            print("Unable to process line (hence ignoring):", info_line)  # ignoring and listing unparsable lines
            continue
        if not info:
            print("Unable to process line (hence ignoring):", info_line)  # ignoring and listing unparsable lines
            continue
        if info[1] not in info_list:
            info_list[info[1]] = list()
        info_list[info[1]].append(info)
        if info[1] == s or (info[1] == "convolution" and info[1] in s):  # closest key, as there is no 1 vs 1 mapping
            c += 1
            if root[1] == c:
                break
    return info_list


def graph_path_gen(graph, enriched=False, return_level=False):
    """BFS Path generator"""
    root = graph[-1]
    adj = graph[0]
    q = queue.Queue()
    q.put((root, 1))
    vis = {}
    # assuming the log has level ordered entities
    while not q.empty():
        node, l = q.get()
        try:
            N = adj[node] if not enriched else adj[node]["adj"]
        except TypeError:
            N = adj[node]
        for n in N:
            if n not in vis:
                q.put((n, l + 1))
                vis[n] = True
        if return_level:
            yield (node, l)
        else:
            yield node


def enchrich_graph_info(info_list, graph):
    """graph traversal retaining info present in the path and also enriches the graph"""
    keys = list(info_list.keys())
    adj = graph[0]
    # print('Path Traversal\n')
    for node in graph_path_gen(graph):
        s = node[0].split("::")[-1]
        b = False
        for k in keys:
            if k == s or (k == "convolution" and k in s):  # closest key, as there is no 1 vs 1 mapping
                v = info_list[k].pop()
                adj[node] = {"adj": adj[node], "log_info": v}
                b = True
                break
        if not b:
            print(f"Ignoring as No log is found for : {node[0]} in info")
            NO_LOG_OP.add(node)
    return graph


def get_aux_method_body(func_name, var_value_list, is_hpu):
    #  generate initializations and function string store it in dummy result
    # something like :
    #  auto x = randn(<>) and return func_call(x, args)
    body = ""
    func_string: str = func_name + "("
    var_name, method_name, auto_init, ret_str = "v", "m", "auto ", "return"
    random_init = "torch::randn("
    inits = []
    var_count = 0
    for k, e in var_value_list:
        # key = k.strip()
        val = e[-1].strip()
        dtype = e[0].strip()
        # if key == 'self' or key == 'input' or key == 'grad_output' or key == 'weight'  or key == 'bias_opt':
        if "Type" in dtype and "HPU" in dtype:
            inits.append(random_init + val + ");")
            func_string += var_name + str(var_count) + ","
            var_count += 1
            continue
        if len(val) == 0 or val == "None":
            func_string += "c10::nullopt,"
            continue
        func_string += val + ","

    func_string = func_string[:-1]
    if func_string.startswith("std::get<0>"):
        func_string += ")"
    if is_hpu:
        func_string += ').to("hpu");'
    else:
        func_string += ");"
    body += "torch::manual_seed(10);\n"
    for i, c in enumerate(inits):
        body += auto_init + var_name + str(i) + "=" + inits[i] + "\n"
    body += auto_init + "res = " + func_string + "\n"
    body += ret_str + " res;" + "\n"
    return body


def wrap_in_aux_method(node, var_value_list, test_count, mtype):
    """individual hpu / cpu computation function"""
    name = "M" + str(test_count) + mtype + "()"
    method_string = "auto " + name + "{\n"
    return (
        method_string
        + get_aux_method_body(node.replace("_overrideable", ""), var_value_list, (mtype == "hpu"))
        + "}\n",
        name,
    )


def create_test_Case(method_names, test_count):
    test_string = "TEST(" + file_name + "," + test_case + str(test_count) + ")"
    test_string += "{\n"
    test_string += "auto v1 = " + method_names[0] + ";\n"
    test_string += "auto v2 = " + method_names[1] + ";\n"
    test_string += "EXPECT_EQ(allclose(v1, v2, 0.001, 0.001), true);\n"
    test_string += "}\n"
    return test_string


def generate_test_case(graph):
    """generate cpp test methods using info"""
    test_count = 0
    test_content = ""
    adj = graph[0]
    for node in graph_path_gen(graph, True):
        # print(node)
        if node in NO_LOG_OP or node[0].split(":")[-1] in REMOVE_TEMP:
            continue
        log_comment = "// FOR " + adj[node]["log_info"][-1] + "\n"
        func_name = (
            "std::get<0>(" + "torch::" + adj[node]["log_info"][1]
            if adj[node]["log_info"][1] in is_tuple
            else "torch::" + adj[node]["log_info"][1]
        )
        cpu = wrap_in_aux_method(func_name, adj[node]["log_info"][2], test_count, "cpu")
        hpu = wrap_in_aux_method(func_name, adj[node]["log_info"][2], test_count, "hpu")
        test_case = create_test_Case([cpu[-1], hpu[-1]], test_count)
        test_content += log_comment + cpu[0] + hpu[0] + test_case
        test_count += 1
    return test_content


def generate_inits(var_value_list, func_name, prev_var_names, is_hpu, level=0):
    """generate code like this : var_name = <> and intermediate_result = <>"""
    body = ""
    var_name, auto_init = f"v_{level}_", "auto "
    random_init = "torch::randn("
    inits = []
    var_count, num_vars = 0, len(prev_var_names)
    func_string: str = func_name + "("
    for k, e in var_value_list:
        # key = k.strip()
        val = e[-1].strip()
        dtype = e[0].strip()
        # if key == 'self' or key == 'input' or key == 'grad_output' or key == 'weight'  or key == 'bias_opt':
        if "Type" in dtype and "HPU" in dtype:
            inits.append(random_init + val + ");")
            if num_vars > var_count:
                func_string += prev_var_names[var_count] + ","
            else:
                func_string += var_name + str(var_count) + ","
            var_count += 1
            continue

        if len(val) == 0 or val == "None":
            func_string += "c10::nullopt,"
            continue
        func_string += val + ","

    func_string = func_string[:-1]
    if func_string.startswith("std::get<0>"):
        func_string += ")"
    if is_hpu:
        func_string += ').to("hpu");'
    else:
        func_string += ");"

    if num_vars == 0:
        body += "torch::manual_seed(10);\n"
        prev_var_names = []

    k = len(inits) - num_vars
    new_var_names = []
    for i in range(num_vars + k):  # after num_vars variables all other variables are initialized randomly
        if i >= num_vars:
            vname = var_name + str(i)
            body += auto_init + vname + "=" + inits[i] + "\n"
            new_var_names.append(vname)

    new_res_name = f"res_{level}_{num_vars}"
    body += auto_init + new_res_name + " = " + func_string + "\n"

    return body, [new_res_name]


def get_neighbors(adj_node):
    try:
        return adj_node["adj"]
    except TypeError:
        return adj_node


def wrap_in_aux_method_depth(adj, path_nodes, mtype):
    """generate cpu/hpu methods for the path_nodes of a given depth"""
    name = "M_depth_" + mtype + "()"
    method_string = "auto " + name + "{\n"
    _content = ""
    node_var_names = {}
    for l, node in enumerate(reversed(path_nodes)):
        if node in NO_LOG_OP or node[0].split(":")[-1] in REMOVE_TEMP:
            raise SubGraphTestCaseException(f"node : {node} returns a non-tensor or no-info-log present")
        var_names = []
        for n in get_neighbors(adj[node]):
            if n in node_var_names:
                var_names.extend(node_var_names[n])
        func_name = (
            "std::get<0>(" + "torch::" + adj[node]["log_info"][1]
            if adj[node]["log_info"][1] in is_tuple
            else "torch::" + adj[node]["log_info"][1]
        )
        instr, var_names = generate_inits(
            adj[node]["log_info"][2], func_name.replace("_overrideable", ""), var_names, (mtype == "hpu"), l
        )
        node_var_names[node] = var_names.copy()
        _content += instr
    _content += "return " + var_names[0] + ";\n"
    return method_string + _content + "}\n", name


def generate_test_case_cumulative(graph, depth):
    """generate cpp testcase for depth from the target node
    Ex => (target_node -calls-> another_node -calls-> ...d_depth)"""
    path_nodes = []  # store all nodes until depth from target_node,
    d = depth
    test_content = "\n"
    for node, l in graph_path_gen(graph, True, True):
        if (depth - l) < 0:
            continue  # not consider deeper nodes
        path_nodes.append(node)

    if len(path_nodes) == 0:
        return ""
    log_comment = "// Sub-graph testcase with depth : " + str(d) + f" for node {graph[1]}\n"
    adj = graph[0]
    cpu = wrap_in_aux_method_depth(adj, path_nodes, mtype="cpu")
    hpu = wrap_in_aux_method_depth(adj, path_nodes, mtype="hpu")
    test_case = create_test_Case((cpu[1], hpu[1]), "_depth")
    test_content += log_comment + cpu[0] + hpu[0] + test_case
    return test_content


def generate_tests(file_name, marker, graph, depth=0):
    """generate test cases using the ir graph from log and op info log file"""
    cpp_test_content = "#include <gtest/gtest.h>\n#include <torch/torch.h>\n#include <ATen/ATen.h>\n\n"
    info_lines = read_under_marker(file_name, marker)
    info_list = process_info_lines(info_lines, graph)
    enriched_graph = enchrich_graph_info(info_list, graph)
    cpp_test_content += generate_test_case(enriched_graph)
    try:
        cpp_test_content += generate_test_case_cumulative(enriched_graph, depth)
    except SubGraphTestCaseException as e:
        print(f"SubgraphGenError => {e.message} hence not generating the depth test-case")
    return cpp_test_content


if __name__ == "__main__":
    import argparse
    import os

    home = os.environ["HOME"]
    parser = argparse.ArgumentParser("python ir_test_generator")
    parser.add_argument(
        "-op", "--op_name", dest="op_name", help="op name like conv/relu/dropout etc", required=True, type=str
    )
    parser.add_argument(
        "-idx",
        "--op_index",
        dest="op_idx",
        help="search for op at {idx}, -1 / blank for last op occurence",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "-m",
        "--marker_name",
        dest="marker",
        help="marker name from file to search from, example: <mark start : 0>...",
        required=True,
        type=str,
    )
    parser.add_argument("-ilog", "--info_log", dest="info_log", help="HPU_INFO_LOG file path", required=False, type=str)
    parser.add_argument("-irlog", "--ir_log", dest="ir_log", help="IR Graph log file path", required=True, type=str)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", help="Print Generated file and Adj matrices", action="store_true"
    )
    parser.add_argument(
        "-donnx",
        "--dest_onnx",
        dest="onnx_file_path",
        help="onnx destination file path",
        type=str,
        default=f"{home}/ir_log.onnx",
    )
    parser.add_argument(
        "-dtest",
        "--dest_test",
        dest="cpp_file_path",
        help="cpp test destination file path",
        type=str,
        default=f"{home}/sample_test.cpp",
    )
    parser.add_argument("-depth", "--depth_test", dest="depth", help="depth for a subgraph test", type=int, default=-1)
    parser.add_argument("-no_test", "--no_test", dest="no_test", help="Does not generate tests", action="store_true")
    args = vars(parser.parse_args())
    op_name, op_idx, marker, ilog, irlog, verbose = (
        args["op_name"],
        args["op_idx"],
        args["marker"],
        args["info_log"],
        args["ir_log"],
        args["verbose"],
    )
    onnx_file_path, cpp_file_path = args["onnx_file_path"], args["cpp_file_path"]
    depth = args["depth"]
    no_test = args["no_test"]
    graph = process_into_graph(irlog, marker, op_name, op_idx)
    if verbose:
        print("--" * 10)
        print("Adjacency list : [(op_name) => {previous op call}], root")
        print("--" * 10)
        print(graph[0])
        print(f"\nroot : {graph[1]}")
        print("--" * 10)
    if not no_test:
        cpp_test_content = generate_tests(ilog, marker, graph, depth)
    if verbose and not no_test:
        print("TEST FILE CONTENT :")
        print("--" * 100)
        print(cpp_test_content)
        print("--" * 100)
        print("Final graph : [(op_name) => {previous op call, info}], root, enriched with logs")
        print(graph)
        print("--" * 10)
        print("saving cpp file")
    if not no_test:
        with open(cpp_file_path, "w+") as f:
            f.write(cpp_test_content)
        print(f"cpp file saved at : {cpp_file_path}")
    print("creating onnx file")
    create_onnx_file(graph, onnx_file_path)
    print("--" * 10)
