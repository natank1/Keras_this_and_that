import onnx
from onnx import helper
import numpy as np
import onnxruntime as ort
import copy
def _add_constant(graph, name, value, data_type, shape):
    constant_node = helper.make_node(
        'Constant',
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(
            name=f"{name}_const",
            data_type=data_type,
            dims=shape,
            vals=[value],
        ),
    )
    graph.node.insert(1, constant_node)
    graph.input.extend([helper.make_tensor_value_info(name, data_type, shape=shape)])
    return
def build_simplified_model(base_model, nodes_to_add=None):
    """Builds a simplified ONNX model incrementally."""
    if base_model is None:
        return None

    graph = base_model.graph
    simplified_nodes = []
    simplified_inputs = []
    simplified_outputs = []
    needed_inputs = set()
    added_if = False
    used_names = set()
    if_node = None

    def _generate_unique_name(base_name):
        """Generates a unique name by appending a counter."""
        counter = 0
        new_name = base_name
        while new_name in used_names:
            new_name = f"{base_name}_{counter}"
            counter += 1
        used_names.add(new_name)
        return new_name

    def _add_node(node):
        node.name = _generate_unique_name(node.name)
        for i, output in enumerate(node.output):
            node.output[i] = _generate_unique_name(output)
        simplified_nodes.append(node)
        for input_name in node.input:
            needed_inputs.add(input_name)

    # 1. Add nodes BEFORE and INCLUDING If_0
    for node in graph.node:
        _add_node(node)
        if node.name == "If_0":
            added_if = True
            if_node = node
            break

    # 2. Add nodes from If_0's subgraphs (Only if If_0 exists)
    if nodes_to_add is not None and added_if:
        for node in list(graph.node):
            if node.name == "If_0":
                for attr in node.attribute:
                    if attr.name in ("then_branch", "else_branch"):
                        subgraph = attr.g
                        new_subgraph_nodes = []
                        for sub_node in subgraph.node:
                            if sub_node.name in nodes_to_add:
                                new_subgraph_nodes.append(sub_node)
                                for input_name in sub_node.input:
                                    needed_inputs.add(input_name)

                        # Create new subgraph with the selected nodes
                        new_subgraph = helper.make_graph(
                            nodes=new_subgraph_nodes,
                            name=subgraph.name,
                            inputs=subgraph.input,
                            outputs=subgraph.output,
                            initializer=subgraph.initializer if hasattr(subgraph, 'initializer') else [],
                            value_info=subgraph.value_info if hasattr(subgraph, 'value_info') else []
                        )

                        # Create new If node with the updated subgraph
                        new_if_node = helper.make_node(
                            'If',
                            inputs=if_node.input,
                            outputs=if_node.output,
                            then_branch=new_subgraph if attr.name == "then_branch" else attr.g,
                            else_branch=new_subgraph if attr.name == "else_branch" else attr.g,
                        )

                        # Replace the old If node with the new one
                        for i in range(len(simplified_nodes)):
                            if simplified_nodes[i].name == "If_0":
                                simplified_nodes[i] = new_if_node
                                break
                    break

        #Update the value info after adding the nodes
        simplified_inputs = []
        simplified_outputs = []
        for input_name in needed_inputs:
            found = False
            for vi in graph.input:
                if vi.name == input_name:
                    simplified_inputs.append(vi)
                    found = True
                    break
            if not found:
                for vi in graph.value_info:
                    if vi.name == input_name:
                        simplified_inputs.append(vi)
                        found = True
                        break
            if not found:
                for node in graph.node:
                    for output in node.output:
                        if output == input_name:
                            for simplified_node in simplified_nodes:
                                if simplified_node.name == node.name:
                                    for i, o in enumerate(simplified_node.output):
                                        if o == input_name:
                                            for attr in graph.node[graph.node.index(node)].attribute:
                                                if attr.name == "value" and hasattr(attr.t, 'dims'):
                                                    simplified_inputs.append(helper.make_tensor_value_info(input_name, attr.type, shape=attr.t.dims))
                                                elif attr.name == "shape" and hasattr(attr, 'ints'):
                                                    simplified_inputs.append(helper.make_tensor_value_info(input_name, onnx.TensorProto.INT64, shape=attr.ints))
                                                elif hasattr(attr, 'type'):
                                                    simplified_inputs.append(helper.make_tensor_value_info(input_name, attr.type, shape=[]))
                                            break
                                    break
                            break
        output_names = set()
        for node in simplified_nodes:
            for output_name in node.output:
                output_names.add(output_name)

        for output_name in output_names:
            found = False
            for vi in graph.output:
                if vi.name == output_name:
                    simplified_outputs.append(vi)
                    found = True
                    break
            if not found:
                for vi in graph.value_info:
                    if vi.name == output_name:
                        simplified_outputs.append(vi)
                        found = True
                        break
            if not found:
                for node in graph.node:
                    for output in node.output:
                        if output == output_name:
                            for simplified_node in simplified_nodes:
                                if simplified_node.name == node.name:
                                    for i, o in enumerate(simplified_node.output):
                                        if o == output_name:
                                            for attr in graph.node[graph.node.index(node)].attribute:
                                                if attr.name == "value" and hasattr(attr.t, 'dims'):
                                                    simplified_outputs.append(helper.make_tensor_value_info(output_name, attr.type, shape=attr.t.dims))
                                                elif attr.name == "shape" and hasattr(attr, 'ints'):
                                                    simplified_outputs.append(helper.make_tensor_value_info(output_name, onnx.TensorProto.INT64, shape=attr.ints))
                                                elif hasattr(attr, 'type'):
                                                    simplified_outputs.append(helper.make_tensor_value_info(output_name, attr.type, shape=[]))
                                            break
                                    break
                            break

    # 4. Create the simplified graph and model
    simplified_graph = helper.make_graph(
        nodes=simplified_nodes,
        name="simplified_graph",
        inputs=simplified_inputs,
        outputs=simplified_outputs,
    )
    simplified_model = helper.make_model(simplified_graph, producer_name="incremental_simplified_model")
    return simplified_model
original_model = onnx.load("fixed_model44.onnx")
def remove_input(graph, input_name):
    """Removes an input with the given name from the graph."""
    for i, input in enumerate(graph.input):
        if input.name == input_name:
            del graph.input[i]
            print(f"Input '{input_name}' removed.")
            return # Exit after removing the input
    print(f"Input '{input_name}' not found.")
def find_nodes_with_output(model, output_name):
    nodes = []
    for node in model.graph.node:
        if output_name in node.output:
            nodes.append(node)
    return nodes
def has_output(graph, output_name):
    """Checks if a graph has a specific output."""
    for node in graph.node:
        if output_name in node.output:
            return True
    return False

original_model = onnx.load("fixed_model44.onnx")

# Check if "sr" exists as an output in the original model's graph or any subgraph
sr_exists_in_graph = has_output(original_model.graph, "sr")

for node in original_model.graph.node:
    for attr in node.attribute:
        if hasattr(attr, 'g'):
            if has_output(attr.g, "sr"):
                sr_exists_in_graph = True
                break
    if sr_exists_in_graph:
     break
remove_input(original_model.graph, "sr")
sr_exists_in_graph = has_output(original_model.graph, "sr")

for node in original_model.graph.node:
    for attr in node.attribute:
        if hasattr(attr, 'g'):
            if has_output(attr.g, "sr"):
                sr_exists_in_graph = True
                break
    if sr_exists_in_graph:
     break

a=1
nodes_with_sr_output = find_nodes_with_output(original_model, "sr")
# Load your original model
original_model = onnx.load("fixed_model44.onnx")
onnx.checker.check_graph(original_model.graph)
# original_model = onnx.load("silero_vad.onnx")
# onnx.checker.check_graph(original_model.graph)
nodes_with_sr_output = find_nodes_with_output(original_model, "sr")
if nodes_with_sr_output:
    print(f"Nodes with 'sr' as output in the original model:")
    for node in nodes_with_sr_output:
        print(f"Node name: {node.name}, OpType: {node.op_type}")
onnx.checker.check_graph(original_model.graph)
print(1)
remove_input(original_model.graph, "sr")
print(2)
# _add_constant(original_model.graph, "sr", 16000, onnx.TensorProto.INT64, (1,))

onnx.checker.check_graph(original_model.graph)
print(3)
print(len(nodes_with_sr_output))
nodes_with_sr_output = find_nodes_with_output(original_model, "sr")
if nodes_with_sr_output:
    print(f"Nodes with 'sr' as output in the original model:")
    for node in nodes_with_sr_output:
        print(f"Node name: {node.name}, OpType: {node.op_type}")
print(len(nodes_with_sr_output))
onnx.checker.check_graph(original_model.graph)
exit(22)
# Remove "sr" from inputs FIRST
remove_input(original_model.graph, "sr")
input_names = [input.name for input in original_model.graph.output]
print (input_names)
sr_count = input_names.count("sr")

# Then add the constant
_add_constant(original_model.graph, "sr", 16000, onnx.TensorProto.INT64, (1,))
input_names = [input.name for input in original_model.graph.input]
print (input_names)
# input_names = [input.name for input in original_model.graph.input]
# print (input_names)
# sr_count = input_names.count("sr")
# print(f"Number of 'sr' inputs in original model: {sr_count}")
# # Add the constant to the ORIGINAL model ONCE
# _add_constant(original_model.graph, "sr", 16000, onnx.TensorProto.INT64, (1,))
# input_names = [input.name for input in original_model.graph.input]
# sr_count = input_names.count("sr")
nodes_with_sr_output = find_nodes_with_output(original_model, "sr")
if nodes_with_sr_output:
    print(f"Nodes with 'sr' as output in the original model:")
    for node in nodes_with_sr_output:
        print(f"Node name: {node.name}, OpType: {node.op_type}")
print(len(nodes_with_sr_output))

onnx.checker.check_graph(original_model.graph)
# print(f"Number of 'sr' inputs in original model: {sr_count}")
exit(333)

original_model = onnx.load("fixed_model44.onnx")

_add_constant(original_model.graph, "sr", 16000, onnx.TensorProto.INT64, (1,))
a=1
onnx.checker.check_graph(original_model.graph)
# Build the initial simplified model (before If_0)
simplified_model_before_if = build_simplified_model(original_model)
a=1


onnx.checker.check_graph(simplified_model_before_if.graph)
onnx.checker.check_model(simplified_model_before_if)


