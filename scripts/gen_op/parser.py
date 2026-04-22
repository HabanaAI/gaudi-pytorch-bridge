###############################################################################
# Copyright (c) 2021-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################


import lark

_GRAMMAR = r"""
    start: type fnname "(" params ")"
    type: CONST? core_type refspec?
    fnname: CNAME
    refspec: REF
           | PTR
    core_type: template
        | TNAME
    template: TNAME "<" typelist ">"
    typelist: type
            | type "," typelist
    REF: "&"
    PTR: "*"
    CONST: "const"
    TNAME: /[a-zA-Z0-9_:]+/
    HEXNUMBER: /0x[0-9a-fA-F]+/
    params: param
          | param "," params
    param: type param_name param_defval?
    param_name: CNAME

    param_defval: "=" init_value
    init_value: "true"
              | "false"
              | "{}"
              | NUMBER
              | SIGNED_NUMBER
              | HEXNUMBER
              | ESCAPED_STRING

    %import common.CNAME -> CNAME
    %import common.NUMBER -> NUMBER
    %import common.SIGNED_NUMBER -> SIGNED_NUMBER
    %import common.ESCAPED_STRING -> ESCAPED_STRING
    %import common.WS
    %ignore WS
    """

_PARSER = lark.Lark(_GRAMMAR, parser="lalr", propagate_positions=True)

_XPARSER = lark.Lark(_GRAMMAR, parser="lalr", propagate_positions=True, keep_all_tokens=True)


class StringEmit:
    def __init__(self, sref):
        self.sref = sref
        self.sval = ""
        self.pos = -1

    def __repr__(self):
        return self.sval

    def advance(self, t):
        start = t.column - 1
        end = t.end_column - 1
        pos = self.pos if self.pos >= 0 else start
        if start > pos:
            self.sval += self.sref[pos:start]
        self.sval += t.value
        self.pos = end

    def skip(self, t):
        self.pos = last_match(t) if self.pos >= 0 else -1

    def append(self, s):
        self.sval += s
        self.pos = -1


def last_match(t):
    if isinstance(t, lark.lexer.Token):
        return t.end_column - 1
    if not isinstance(t, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    return last_match(t.children[-1])


def for_every_token(t, fn):
    if isinstance(t, lark.lexer.Token):
        fn(t)
    else:
        if not isinstance(t, lark.tree.Tree):
            raise AssertionError("Not a lark.tree.Tree instance")
        for c in t.children:
            for_every_token(c, fn)


def emit_string(t, emit, emit_fn):
    status = emit_fn(t)
    if status > 0:

        def do_emit(tok):
            emit.advance(tok)

        for_every_token(t, do_emit)
    elif status == 0:
        if isinstance(t, lark.lexer.Token):
            emit.advance(t)
        else:
            if not isinstance(t, lark.tree.Tree):
                raise AssertionError("Not a lark.tree.Tree instance")
            for c in t.children:
                emit_string(c, emit, emit_fn)
    else:
        emit.skip(t)


def typed_child(t, n, ttype):
    if not isinstance(t, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    if not n < len(t.children):
        raise AssertionError("Not enought children")
    c = t.children[n]
    if not isinstance(c, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    if not c.data == ttype:
        raise AssertionError(t.pretty())
    return c


def create_stdfunc_sig(tree, orig_sig):
    def emit_fn(t):
        if isinstance(t, lark.lexer.Token):
            return 0
        return -1 if t.data == "param_name" else 0

    emit = StringEmit(orig_sig)
    # Emit full function return type.
    emit_string(typed_child(tree, 0, "type"), emit, emit_fn)
    emit.append("(")
    # Emit parameter list w/out parameter names.
    emit_string(typed_child(tree, 3, "params"), emit, emit_fn)
    emit.append(")")
    return str(emit)


def create_map_sig(tree, orig_sig):
    def emit_fn(t):
        if isinstance(t, lark.lexer.Token):
            return -1 if t.type in ["CONST", "REF", "PTR"] else 0
        return -1 if t.data in ["param_name", "param_defval"] else 0

    emit = StringEmit(orig_sig)
    # Emit full function return type.
    emit_string(typed_child(tree, 1, "fnname"), emit, emit_fn)
    emit.append("(")
    # Emit parameter list w/out parameter names.
    emit_string(typed_child(tree, 3, "params"), emit, emit_fn)
    emit.append(") -> ")
    emit_string(typed_child(tree, 0, "type"), emit, emit_fn)
    return str(emit)


# Returns core_type from lark tree
# recursive - is for templates. Type extraction is limited to just one type so:
#   - for type std::optional<int> it will return std::optional<int>
#   - for type std::optional<ArrayRef<int>> it will return std::optional<ArrayRef> as
#        further type extraction is not necessary in that case
def type_core(t, recursive=True):
    if not isinstance(t, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    for c in t.children:
        if isinstance(c, lark.tree.Tree) and c.data == "core_type":
            c = c.children[0]
            if isinstance(c, lark.lexer.Token):
                return c.value
            if not isinstance(c, lark.tree.Tree) and c.data == "template":
                raise AssertionError("Not a lark.tree.Tree instance")
            if recursive:
                try:
                    for c2 in c.children:
                        if isinstance(c2, lark.tree.Tree) and c2.data == "typelist":
                            return f"{c.children[0].value}<{type_core(c2.children[0], False)}>"
                except:
                    pass
            return c.children[0].value
    raise RuntimeError(f"Not a type tree: {t}")


def type_is_const(t):
    if not isinstance(t, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    c = t.children[0]
    return isinstance(c, lark.lexer.Token) and c.value == "const"


def extract_list(t, result_list):
    if not isinstance(t, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    result_list.append(t.children[0])
    if len(t.children) == 2:
        c = t.children[1]
        if isinstance(c, lark.tree.Tree) and c.data == t.data:
            extract_list(c, result_list)
    return result_list


def get_function_signature(t, orig_sig, namefn):
    emit = StringEmit(orig_sig)
    # Emit full function return type.
    emit_string(typed_child(t, 0, "type"), emit, lambda t: 0)
    fnname = typed_child(t, 1, "fnname").children[0]
    emit.append(f" {namefn(fnname.value)}(")
    # Emit parameter list w/out parameter names.
    emit_string(typed_child(t, 3, "params"), emit, lambda t: 0)
    emit.append(")")
    return str(emit), fnname.value


def get_parameters(t):
    if not isinstance(t, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    c = t.children[2]
    if not isinstance(c, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    if not c.data == "params":
        raise AssertionError("Not params")
    params = []
    extract_list(c, params)
    return params


def param_name(t):
    if not isinstance(t, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    c = t.children[1]
    if not isinstance(c, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    if not c.data == "param_name":
        raise AssertionError("Not a param name")
    token = c.children[0]
    if not isinstance(token, lark.lexer.Token):
        raise AssertionError("Not a lark.lexer.Token instance")
    return token.value


def param_type(t):
    if not isinstance(t, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    c = t.children[0]
    if not isinstance(c, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    return c


def get_return_type_str(t, orig_sig):
    if not isinstance(t, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    fname = t.children[1]
    if not isinstance(fname, lark.tree.Tree):
        raise AssertionError("Not a lark.tree.Tree instance")
    if not fname.data == "fnname":
        raise AssertionError("Incorrect fname")
    token = fname.children[0]
    if not isinstance(token, lark.lexer.Token):
        raise AssertionError("Not a lark.lexer.Token instance")
    return orig_sig[0 : token.column - 2]


def parse(s):
    return _PARSER.parse(s)


def xparse(s):
    return _XPARSER.parse(s)
