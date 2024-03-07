# The imports used in this block
# from onnx import checker
from onnx import helper as h
from onnx import save


def create_get_node(onnx_nodes, name, op_type, root_name):
    if name not in onnx_nodes:
        node = h.make_node(
            op_type,
            name=name,
            inputs=[],
            outputs=[],
            doc_string=f"no info log line found or this op is not in the present path till/of : {root_name}",
        )
        onnx_nodes[name] = node
    return onnx_nodes[name]


def create_onnx_graph(graph):
    root = graph[-1]
    adj = graph[0]  # contains either node=>dict if enriched else node=>list
    onnx_nodes = {}  # name=>onnx_node
    get_name = lambda k: k[0] + "_idx_" + str(k[1])
    root_name = get_name(root)
    # input node
    for k in adj.keys():
        name = get_name(k)
        node = create_get_node(onnx_nodes, name, k[0], root_name)
        N = adj[k]["adj"] if type(adj[k]) == dict else adj[k]
        node_names = list(map(lambda x: get_name(x), N))
        node.output.extend([name])  # node channel
        node.input.extend(node_names)  # subscribe to all input node channels
        if type(adj[k]) == dict:
            node.doc_string = adj[k]["log_info"][-1]
    g = h.make_graph(
        nodes=list(onnx_nodes.values()),
        name="ir_graph_log",
        inputs=[],
        outputs=[h.make_empty_tensor_value_info(root_name)],
        doc_string="visualization of ir log graph",
    )
    return g


def create_onnx_file(graph, path):
    """create and save onnx file"""
    g = create_onnx_graph(graph)
    m = h.make_model(g, producer_name="log-ir-graph-demo")
    #      checker.check_model(m) # opname mismatch might be present in onnx domain
    save(m, path)
    print(f"finished saving onnx file at : {path}")
