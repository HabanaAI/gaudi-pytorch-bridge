###############################################################################
#
#  Copyright (c) 2021-2024 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
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
    assert isinstance(t, lark.tree.Tree)
    return last_match(t.children[-1])


def for_every_token(t, fn):
    if isinstance(t, lark.lexer.Token):
        fn(t)
    else:
        assert isinstance(t, lark.tree.Tree)
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
            assert isinstance(t, lark.tree.Tree)
            for c in t.children:
                emit_string(c, emit, emit_fn)
    else:
        emit.skip(t)


def typed_child(t, n, ttype):
    assert isinstance(t, lark.tree.Tree)
    assert n < len(t.children)
    c = t.children[n]
    assert isinstance(c, lark.tree.Tree)
    assert c.data == ttype, t.pretty()
    return c


def rewrite_sig(tree, orig_sig, emit_fn=lambda x: 0):
    emit = StringEmit(orig_sig)
    emit_string(tree, emit, emit_fn)
    return str(emit)


def rewrite_signature(sig, tmap):
    def rewrite(t):
        if t.type == "TNAME":
            new_type = tmap.get(t.value, None)
            if new_type is not None:
                t.value = new_type

    def emit_fn(t):
        if isinstance(t, lark.lexer.Token):
            return 0
        return -1 if t.data == "param_defval" else 0

    xtree = _XPARSER.parse(sig)
    for_every_token(xtree, rewrite)
    return rewrite_sig(xtree, sig, emit_fn=emit_fn)


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
    assert isinstance(t, lark.tree.Tree)
    for c in t.children:
        if isinstance(c, lark.tree.Tree) and c.data == "core_type":
            c = c.children[0]
            if isinstance(c, lark.lexer.Token):
                return c.value
            assert isinstance(c, lark.tree.Tree) and c.data == "template"
            if recursive:
                try:
                    for c2 in c.children:
                        if isinstance(c2, lark.tree.Tree) and c2.data == "typelist":
                            return f"{c.children[0].value}<{type_core(c2.children[0], False)}>"
                except:
                    pass
            return c.children[0].value
    raise RuntimeError("Not a type tree: {}".format(t))


def type_is_const(t):
    assert isinstance(t, lark.tree.Tree)
    c = t.children[0]
    return isinstance(c, lark.lexer.Token) and c.value == "const"


def extract_list(t, l):
    assert isinstance(t, lark.tree.Tree)
    l.append(t.children[0])
    if len(t.children) == 2:
        c = t.children[1]
        if isinstance(c, lark.tree.Tree) and c.data == t.data:
            extract_list(c, l)
    return l


def get_function_signature(t, orig_sig, namefn):
    emit = StringEmit(orig_sig)
    # Emit full function return type.
    emit_string(typed_child(t, 0, "type"), emit, lambda t: 0)
    fnname = typed_child(t, 1, "fnname").children[0]
    xfname = namefn(fnname.value)
    emit.append(" {}(".format(xfname))
    # Emit parameter list w/out parameter names.
    emit_string(typed_child(t, 3, "params"), emit, lambda t: 0)
    emit.append(")")
    return str(emit), fnname.value, xfname


def get_parameters(t):
    assert isinstance(t, lark.tree.Tree)
    c = t.children[2]
    assert isinstance(c, lark.tree.Tree)
    assert c.data == "params"
    params = []
    extract_list(c, params)
    return params


def param_name(t):
    assert isinstance(t, lark.tree.Tree)
    c = t.children[1]
    assert isinstance(c, lark.tree.Tree)
    assert c.data == "param_name"
    token = c.children[0]
    assert isinstance(token, lark.lexer.Token)
    return token.value


def param_type(t):
    assert isinstance(t, lark.tree.Tree)
    c = t.children[0]
    assert isinstance(c, lark.tree.Tree)
    return c


def get_return_type_str(t, orig_sig):
    assert isinstance(t, lark.tree.Tree)
    fname = t.children[1]
    assert isinstance(fname, lark.tree.Tree)
    assert fname.data == "fnname"
    token = fname.children[0]
    assert isinstance(token, lark.lexer.Token)
    return orig_sig[0 : token.column - 2]


def parse(s):
    return _PARSER.parse(s)


def xparse(s):
    return _XPARSER.parse(s)
