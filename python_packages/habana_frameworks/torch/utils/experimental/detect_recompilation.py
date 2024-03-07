import inspect
import math
import time
from types import MethodType

import torch
from torch.utils.data import DataLoader, Dataset

TAG = "[DETECT_RECOMPILE_AUTO]"
STEP_COUNT = "step_count"
INP_HASH = "inp_hash"
OUT_HASH = "out_hash"
DYNSHAPE_FIELD = "_dynshape_info"


class Node:
    def __init__(self, name, parent=None):
        assert type(name) == str
        self.name = name
        self.children = []
        self.parent = parent
        if self.parent != None:
            # self could get readded even though it might already exist
            self.add_self_as_child(parent)

    def add_self_as_child(self, parent):
        present = False
        for k in self.parent.children:
            if k.name == self.name:
                present = True
        if not present:
            self.parent.children += [self]

    def __repr__(self):
        parent_name = "None" if self.parent is None else self.parent.name
        return f'{parent_name} <- {self.name} <- [{",".join([k.name for k in self.children])}]'


class Table:
    def __init__(self, field_names, max_char=120):
        self.field_names = field_names
        self.rows = [self.field_names]
        self.max_char = max_char

    def add_row(self, lst):
        assert len(lst) == len(self.field_names)
        self.rows += [lst]

    def _fattest_per_col(self):
        return [max([len(str(row[col_idx])) for row in self.rows]) for col_idx in range(len(self.field_names))]

    def _budget_per_col(
        self,
    ):
        fattest_per_col = self._fattest_per_col()
        ratio = [k / sum(fattest_per_col) for k in fattest_per_col]
        return [int(r * self.max_char) + 1 for r in ratio]

    def __repr__(self):
        budget_col = self._budget_per_col()
        num_cols = len(self.rows[0])
        st = ""

        def row_boundary(ch):
            return "\033[96m" + ch * (sum(budget_col) + num_cols + 1) + "\033[0m" + "\n"

        for rowidx, row in enumerate(self.rows):
            if rowidx == 0:
                st += row_boundary("=")
            assert len(budget_col) == len(row)
            # splitting a row across multiple rows
            max_rows_needed = max(
                [int(math.ceil(len(str(item)) / char_budget)) for item, char_budget in zip(row, budget_col)]
            )
            start_col = ("", "\033[91m")[rowidx == 0]
            end_col = ("", "\033[1m")[rowidx == 0]
            for k in range(max_rows_needed):
                for col_idx, item in enumerate(row):
                    orig_str = str(item)
                    slice = orig_str[k * budget_col[col_idx] : (k + 1) * budget_col[col_idx]]
                    st += (
                        ("", "\033[96m|\033[0m")[col_idx == 0]
                        + start_col
                        + slice
                        + end_col
                        + " " * (budget_col[col_idx] - len(slice))
                        + "\033[96m|\033[0m"
                    )
                st += "\n"
            st += (row_boundary("-"), row_boundary("="))[rowidx == 0]
        return st

    def dump_to_csv(self, flname):
        with open(flname, "w") as f:
            for row in self.rows:
                f.write(",".join(map(str, row)) + "\n")


def _make_table(rowname, sorteddict, max_width, csv_out):
    x = Table(rowname, max_width)
    for k, v in sorteddict:
        for it in v:
            lst = [k] + list(it)
            x.add_row(lst)
    print(x)
    if csv_out is not None:
        x.dump_to_csv(csv_out)


def _prettyprint(recompiling_modules, recompiling_modules_count, csv_out):
    sorted_by_step = sorted(recompiling_modules.items(), key=lambda kv: kv[0])
    _make_table(
        ["Step", "Recompiling modules", "New in", "New out", "Class", "Location", "Comment"],
        sorted_by_step,
        120,
        csv_out + "_1.csv",
    )
    _make_table(
        ["Module name", "Recompile count"],
        [(k, [[v]]) for k, v in sorted(recompiling_modules_count.items(), key=lambda kv: -kv[1])],
        80,
        csv_out + "_2.csv",
    )


# trying to find nodes which recompiled even though immediate children did not recompile
def _process_tree(root, output, potential_dyn_modules):
    an_immediate_child_recompiled = False
    curr_immediate_child_recompiled = False
    for k in root.children:
        dyn_from_children, curr_immediate_child_recompiled = _process_tree(k, [], potential_dyn_modules)
        output += dyn_from_children
        an_immediate_child_recompiled = an_immediate_child_recompiled or curr_immediate_child_recompiled
    is_root_dyn = root.name in potential_dyn_modules and not curr_immediate_child_recompiled
    if is_root_dyn:
        output += [root.name]
    return output, is_root_dyn


def _parse(lines):
    recompile_detector_lines = [ln for ln in lines if "DETECT_RECOMPILE_AUTO" in ln]
    recompiling_modules = {}
    recompiling_modules_count = {}
    step = -1
    top_module_name = None
    all_names = []
    module_files = {}
    step_from_ln = 0
    tmp_list = []
    for ln in recompile_detector_lines:
        if "[DETECT_RECOMPILE_AUTO] Registering hooks for" in ln:
            splt = ln.split("[DETECT_RECOMPILE_AUTO] Registering hooks for")
            mdlname = splt[1].split("of type")[0].strip()
            lhs, rhs = splt[1].split("of type")[1].split("probably defined here")
            classnm = lhs.strip().split("class")[1].strip(">").strip().strip("'")
            rhs = rhs.strip().strip(">").strip("<")
            modulenm, filenm = rhs.split(" from ")
            filenm = filenm.strip("'")
            modulenm = modulenm.split("module ")[-1].strip("'")
            if mdlname in module_files:
                classnm1, modulenm1, filenm1 = module_files[mdlname]
                assert (mdlname, classnm, modulenm, filenm) != (mdlname, classnm1, modulenm1, filenm1)
            module_files[mdlname] = (classnm, modulenm, filenm)

        step_done = False
        if "DETECT_RECOMPILE_AUTO" in ln:
            if "step" in ln:
                p0, p1 = ln.split("step:")
                step_from_ln = int(p1.split(" ")[0])
                assert step_from_ln > step
                top_module_name_ = ln.split("][")[1].split("]")[0]
                if top_module_name is not None:
                    assert top_module_name == top_module_name_
                else:
                    top_module_name = top_module_name_
                step_done = True

        if "Recompilation" in ln:
            if "step" in ln:
                p0, _ = ln.split("step:")
            else:
                p0 = ln.strip().split(" ")[0]
            module_name = p0.split("[")[-1].strip().strip("]")
            classnm, modulenm, filenm = module_files[module_name]
            new_inp = "Found new input signature" in ln
            new_out = "Found new output signature" in ln
            # every one compiles in first step, so construct module tree here
            if step == -1 and module_name not in all_names:
                all_names = all_names + [module_name]

            if new_inp:
                comment = "Recompiled due to new input shape"
            else:
                if new_out:
                    comment = "Already processed input shape still recompiled and has new output shape. Maybe dyn ops"
                else:
                    comment = "Already processed input shape still recompiled. Maybe dyn ops"
            tmp_list += [(module_name, new_inp, new_out, classnm, filenm, comment)]
            recompiling_modules_count[module_name] = recompiling_modules_count.get(module_name, 0) + 1
        if step_done:
            step = step_from_ln
            step_done = False
            if len(tmp_list) > 0:
                recompiling_modules[step_from_ln] = tmp_list
            tmp_list = []

    """
    The name a/b/c
    creates the following: a/b -> a/b/c, a -> a/b
    """
    created_nodes = {}
    for nm in all_names:
        split = nm.split("/")
        for i in range(len(split) - 1, 0, -1):
            lhs = "/".join(split[:i])
            rhs = "/".join(split[: i + 1])
            if lhs not in created_nodes:
                lhs_node = Node(lhs)
                created_nodes[lhs] = lhs_node
            lhs_node = created_nodes[lhs]
            if rhs not in created_nodes:
                rhs_node = Node(rhs, parent=lhs_node)
                created_nodes[rhs] = rhs_node
            else:
                assert created_nodes[rhs].parent is None or created_nodes[rhs].parent.name == lhs_node.name
                created_nodes[rhs].parent = lhs_node
                created_nodes[rhs].add_self_as_child(lhs_node)

    potential_dyn_modules = set()
    for step in recompiling_modules:
        for mdlname, newinp, newout, classnm, filenm, comment in recompiling_modules[step]:
            if not newinp:
                potential_dyn_modules.update([mdlname])

    treeinfo = []
    treeinfo, _ = _process_tree(created_nodes[top_module_name], treeinfo, potential_dyn_modules)
    for step_idx in recompiling_modules:
        for idx, (module_name, new_inp, new_out, classnm, filenm, comment) in enumerate(recompiling_modules[step_idx]):
            if "Already processed input shape still recompiled" in comment and module_name not in treeinfo:
                comment += ". Could be due to dynamic child"
                recompiling_modules[step_idx][idx] = (module_name, new_inp, new_out, classnm, filenm, comment)
    return recompiling_modules, recompiling_modules_count, treeinfo, top_module_name, created_nodes


def _wrap_fn(old_fn, tag1, write_to, level=0, waittime=1):
    import habana_frameworks.torch.core as htcore
    import torch
    from habana_frameworks.torch.hpu.metrics import metric_localcontext

    def forward(self, *args, **kwargs):
        assert not torch.distributed.is_initialized(), "Expected 1x run, but torch being used in distributed fashion"
        field_contents = getattr(self, DYNSHAPE_FIELD)
        assert INP_HASH in field_contents
        inp_hash = htcore.hpu.input_hash((args, kwargs))
        htcore.mark_step()
        time.sleep(waittime)
        with metric_localcontext(f"graph_compilation") as local_metric:
            res = old_fn(*args, **kwargs)
            htcore.mark_step()
            time.sleep(waittime)
        out_hash = htcore.hpu.input_hash(res)
        metrics = {k: v for k, v in local_metric.stats()}
        if inp_hash not in field_contents[INP_HASH]:
            field_contents[INP_HASH].update([inp_hash])
            new_inp_string = "Found new input signature "
        else:
            new_inp_string = ""

        if out_hash not in field_contents[OUT_HASH]:
            field_contents[OUT_HASH].update([out_hash])
            new_out_string = "Found new output signature "
        else:
            new_out_string = ""
        if STEP_COUNT in field_contents:
            step = field_contents[STEP_COUNT]
            field_contents[STEP_COUNT] = field_contents[STEP_COUNT] + 1
            setattr(self, DYNSHAPE_FIELD, field_contents)
            step_string = f"step:{step+1}"
        else:
            step_string = ""
        num_graphs = metrics["TotalNumber"]
        changed = num_graphs != 0
        debug_str = (
            f'{"  "*level}{TAG}[{tag1}]{step_string} {new_inp_string}{new_out_string}num_graphs:{num_graphs} '
            + ("", "Recompilation!")[changed]
        )
        write_to.append(debug_str)
        return res

    return forward


def _get_analyser(csv_out):
    def analyse_dynamicity(self):
        recompiling_modules, recompiling_modules_count, treeinfo, top_module_name, created_nodes = _parse(
            self.raw_logs()
        )
        _prettyprint(recompiling_modules, recompiling_modules_count, csv_out)

    return analyse_dynamicity


def detect_recompilation_auto_model(model, mdlname="Net", waittime=1, csv_out="out"):
    """
    Accepts a pytorch nn.Module and returns the same module with \
    hooks added to detect recompilations

    Parameters:
            model (Pytorch nn.Module): A pytorch module
            mdlname (str): Optional, A model name
            waittime (int): Optional, Wait time, to make sure graph compile counts are correct.
                            May need larger number for modules with large forward functions
            csv_out (str): Optional, a file name based on which output files will be generated
    Returns:
            model (Pytorch nn.Module): Modified model,
                                       with analyse_dynamicity() and raw_logs() functions added
    """
    edited_modules = set()

    def helper(model, write_to, mdlname="Net", level=0, waittime=1):
        model.__name__ = mdlname
        if hasattr(model, DYNSHAPE_FIELD):
            assert hash(model) in edited_modules, f"The model already has field {DYNSHAPE_FIELD}"
            # we potentially have a situation like:
            # parent
            #   self.c1 = child1()
            #   self.c2 = child2(self.c1)
            #   self.c3 = child3(self.c1)
            # that is parent nn module will call helper on self.c1,
            # but the same module is passed to self.c2 and self.c3, which try to edit the forward of c1 again
            # so we will skip adding again for self.c1
            return model

        assert hash(model) not in edited_modules
        edited_modules.update([hash(model)])
        setattr(model, DYNSHAPE_FIELD, {})
        field_contents = getattr(model, DYNSHAPE_FIELD)
        # counter to count how many times fwd pass was called on the top level model
        if level == 0:
            field_contents[STEP_COUNT] = -1

        registration = (
            f'{"  "*level}{TAG} Registering hooks for '
            + mdlname
            + " of type "
            + str(model.__class__)
            + " probably defined here "
            + str(inspect.getmodule(model))
        )
        write_to.append(registration)
        # Set of input hashes to detect if we encounter a new input/output signature
        field_contents[INP_HASH] = set()
        field_contents[OUT_HASH] = set()
        setattr(model, DYNSHAPE_FIELD, field_contents)
        model.forward = MethodType(
            _wrap_fn(model.forward, mdlname, level=level, waittime=waittime, write_to=write_to), model
        )
        for name, layer in model.named_children():
            layer = helper(layer, write_to, mdlname + "/" + name, level + 1, waittime=waittime)
        if level == 0:
            assert not hasattr(model, "raw_logs"), f"The model already has field raw_logs"
            model.raw_logs = MethodType(lambda self: write_to, model)
            assert not hasattr(model, "analyse_dynamicity"), f"The model already has field analyse_dynamicity"
            model.analyse_dynamicity = MethodType(_get_analyser(csv_out), model)
        return model

    write_to = []
    return helper(model, write_to, mdlname=mdlname, level=0, waittime=waittime)


def get_shape(item):
    if type(item) == type(torch.tensor([])):
        return tuple(item.shape)
    elif type(item) == type([]):
        return tuple([get_shape(k) for k in item])
    elif type(item) == type(tuple()):
        return tuple([get_shape(k) for k in item])
    elif isinstance(item, dict):
        return tuple((get_shape(k), get_shape(v)) for k, v in item.items())
    else:
        print(f"Found type{type(item)}, not of types list, tuple, dict or tensor so using hash")
        return hash(item)


def data_dynamicity(dl):
    """
    Accepts a dataloader and publishes a report about its dynamicity

    Parameters:
            dl (Pytorch torch.utils.data.DataLoader): A pytorch dataloader
    Returns:
            hist (dictionary): A histogram of inputs
    """
    assert not torch.distributed.is_initialized(), "Expected 1x run, but torch being used in distributed fashion"
    hist = {}
    for dt in dl:
        shp_key = get_shape(dt)
        hist[shp_key] = hist.get(shp_key, 0) + 1
    print_result(hist)
    return hist


def print_result(hist):
    tbl = Table(["Shape", "Count"], 120)
    for k, v in sorted(hist.items(), key=lambda item: -item[1]):
        tbl.add_row([k, v])
    print(tbl)
    num_shapes = len(hist)
    print("Number of unique shapes: ", num_shapes)
    if num_shapes >= 1:
        if num_shapes > 20:
            modifier = "a lot of"
        elif num_shapes > 5:
            modifier = "some"
        elif num_shapes > 1:
            modifier = "a little"
        elif num_shapes == 1:
            modifier = "no"
        else:
            assert False
        print(f"There is {modifier} dynamicity in input data shapes")
    else:
        print("Dataset looks empty")


def const_shape_dataloader(dl, maxlen):
    """
    Accepts a dataloader and filters it to return a new shorter
    dataloader which has data of same shape- and length<=maxlen

    Parameters:
            dl (Pytorch torch.utils.data.DataLoader): A pytorch dataloader
            maxlen (int): nMax length of new shorter, constant shape dataloader
    Returns:
            dataloader (Pytorch torch.utils.data.DataLoader): The new dataloader
    """

    def helper(lst):
        return DataLoader(lst, batch_size=None, collate_fn=None)

    const_shape_dt = {}
    for dt in dl:
        shp_key = get_shape(dt)
        if shp_key not in const_shape_dt:
            const_shape_dt[shp_key] = []
        if len(const_shape_dt[shp_key]) == maxlen:
            return helper(const_shape_dt[shp_key])
        const_shape_dt[shp_key] = const_shape_dt[shp_key] + [dt]

    maxlength = -1
    for k in const_shape_dt:
        currlen = len(const_shape_dt[k])
        assert currlen < maxlen
        if maxlength < currlen:
            maxlength = currlen
            maxlenkey = k
    assert maxlength > 0, "Perhaps dataset is empty"
    return helper(const_shape_dt[maxlenkey])
