"""
Constructing sub-graphs from IR logs
"""

import os


class OpNotFoundException(BaseException):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


# assuming marker to be mark start : <id>
MARKER = "mark start"


def read_reverse(file_name):
    """simple generator to read file backwards line by line"""
    with open(file_name, "rb") as read_obj:
        read_obj.seek(0, os.SEEK_END)
        pointer_location = read_obj.tell()
        buffer = bytearray()
        while pointer_location >= 0:
            read_obj.seek(pointer_location)
            pointer_location = pointer_location - 1
            new_byte = read_obj.read(1)
            if new_byte == b"\n":
                yield buffer.decode()[::-1]
                buffer = bytearray()
            else:
                buffer.extend(new_byte)
        if len(buffer) > 0:
            yield buffer.decode()[::-1]


def read_under_marker(file_name, mark):
    """read under this mark step"""
    lines = []
    for line in read_reverse(file_name):
        if mark in line:
            break
        lines.insert(0, line)
        if MARKER in line:
            lines.clear()
    return lines


def extract_graph_logs(ir_lines: str):
    """returns a list of strings where each has element is of pattern graph(*)*\n"""
    graph_log_list = []
    read = False
    for line in ir_lines:
        line = line.strip()
        if line.startswith("graph("):
            read = True
            graph_log_list.append(list())
        if read:
            graph_log_list[-1].append(line)
        if line.startswith("return"):
            read = False

    return graph_log_list


def build_var_table(graph_log: str):
    """returns the var_table : var_name as %.... => namespace::op_name? , [arg_var1, arg_var2,....], op_count_idx"""
    # assuming a parse schema of : [space character plays a central role]
    # ...<%_var_name_1> : type... = nsp::name[...](arg1, arg2, arg3)....  for single var
    # ...<%_var_name_1> : type..., ...<%_var_name_2> : type = nsp::name[...](%...., %...., %....)....
    var_table, opnamecount = {}, {}
    for line in graph_log:
        if line.startswith("return"):
            continue
        (
            var_name,
            op_name,
        ) = (
            "",
            "",
        )
        curr_var_list = []
        i = 0
        read = False
        # read variables / keys in var table
        while i < len(line):
            if line[i] == "%":
                read = True
            if line[i] == " " and read:
                curr_var_list.append(var_name)
                var_name, read = "", False
            if read:
                var_name += line[i]
            if line[i] == " " and i < len(line) and line[i + 1] == "=":
                i += 3
                break
            i += 1
        # read op_name after '=<space>' and args = (...)
        read_code = 1
        op_name, arg_names_list = "", []
        while i < len(line):
            if read_code == 1 and line[i] == "[" or line[i] == "(":
                read_code = 0  # sometimes names don't end with [] instead ()
            if line[i] == ")":
                break  # end of args
            if line[i - 1] == "(":
                read_code = 2
                arg_names_list.append("")
            if line[i] == "," and read_code == 2:
                arg_names_list.append("")
                i += 2  # skip space and comma
            if read_code == 1:
                op_name += line[i]
            if read_code == 2:
                arg_names_list[-1] += line[i]
            i += 1

        # filter arg_names_list (retain only variables)
        arg_names_list = [name for name in arg_names_list if "%" in name]
        for name in curr_var_list:
            if op_name in opnamecount:
                opnamecount[op_name] += 1
            else:
                opnamecount[op_name] = 1
            var_table[name] = (op_name, arg_names_list, opnamecount[op_name])

    return var_table


def parse_all_ops(graph_log: str, op: str, op_index):
    """returns a (dictionary {op_name with index=> {previous_connected_ops}}), root for op, true/false (op_found) for this task in pattern graph(*)*\n"""
    op_call_dict = {}  # (op_name, namespace, idx) => [other keys]
    found, root = False, (None, 0)
    var_table = build_var_table(graph_log)
    for val in var_table.values():
        if len(val[0]) == 0 or len(val[1]) == 0:
            continue
        neighbors = [
            (var_table[e][0], var_table[e][-1]) for e in val[1] if len(var_table[e][0]) > 0 and len(var_table[e][1]) > 0
        ]
        op_call_dict[(val[0], val[-1])] = neighbors
        if op_index == val[-1] and op in val[0]:
            found, root = True, (val[0], val[-1])
        if op_index == -1 and op in val[0]:  # if the index is unspecified merge with the last op
            found, root = True, (val[0], max(val[-1], root[-1]))
    return (op_call_dict, root), found


def process_into_graph(file_name, mark_step, op, op_index=-1):
    """constructs sub-graph of all connected ops for op with index op_index"""
    ir_lines = read_under_marker(file_name, mark_step)
    graph_log_list = extract_graph_logs(ir_lines)
    for graph_log in graph_log_list:
        op_call_dict, is_found = parse_all_ops(graph_log, op, op_index)
        if is_found:
            # returning here with a simple assumption of single op search
            return op_call_dict
            # add visualization here if required

    raise OpNotFoundException(f"{op} not found")
