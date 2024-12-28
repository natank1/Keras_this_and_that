import onnx
from onnx import helper, numpy_helper
import copy
import onnxruntime as ort
import onnx
from onnx import helper, numpy_helper
import copy


def split_if_node(model):
    """
    Splits an 'If' node in an ONNX model into separate 'then' and 'else' branches,
    properly handling inputs and outputs for the new nodes.

    Parameters:
        model (onnx.ModelProto): The ONNX model containing the 'If' node.

    Returns:
        onnx.ModelProto: The modified ONNX model with the 'If' node split.
    """
    nodes = model.graph.node
    inputs = model.graph.input
    outputs = model.graph.output

    new_nodes = []  # List to hold new nodes for the updated model
    new_outputs = []  # List to hold new outputs after splitting the if node
    original_if_node = None
    input_names = {input.name for input in model.graph.input}  # Set of original input names

    # Iterate through the nodes to find and process 'If' nodes
    for node in nodes:

        if node.op_type == "If":
            cond0=node.input[0]
            outp0 =[i for i in node.output]
            start_node= node.input
            original_if_node = node
            stp0=[]
            stp1=[]
            else_branc=node.attribute[0].g
            then_brach=node.attribute[1].g
            for attr in node.attribute:

                l_nodes=[]
                for node in attr.g.node:
                    so = [l + '_' + attr.name if 'v_118' in l else l for l in node.output]
                    si = [l + '_' + attr.name if 'v_118' in l else l for l in node.input]

                    new_node = copy.deepcopy(node)
                    new_node.input[:]=si
                    new_node.output[:]=so

                    l_nodes.append(new_node)

                new_nodes.extend(l_nodes)

                       # new_nodes.extend([concat_node_0,concat_node_1])
            for l,i in enumerate(new_nodes):
                # if i.name=='':
                    i.name="node_"+str(l)+"_is_here"

            a=2
            # then_outputs = [output for node in then_brach .node for output in node.output]
            # else_outputs = [output for node in else_branc.node for output in node.output]
            then_subgraph_outputs = then_brach .output  # These should be the final outputs of the 'then' branch
            else_subgraph_outputs = else_branc.output

            where_nodes = []
            final_outputs = []

            for i, (then_out, else_out) in enumerate(zip(then_subgraph_outputs, else_subgraph_outputs)):
                final_output_name = f"final_output_{i}"
                where_node = onnx.helper.make_node(
                    "Where",
                    inputs=[cond0, then_out.name, else_out.name],
                    outputs=[final_output_name]
                )
                where_nodes.append(where_node)
                final_outputs.append(final_output_name)

            # Add the Where nodes to the main graph
            new_nodes.extend(where_nodes)
            a=2
            squeeze_node0 = helper.make_node(
                "Squeeze",  # Node type
                inputs=new_nodes[92].attribute[0].g.node[75].input[:],  # Input tensor
                outputs=["squeeze_00"],  # Output tensor
                # axes=[1],  # Specify the axes to squeeze
            )
            squeeze_node1 = helper.make_node(
                "Squeeze",  # Node type
                inputs=new_nodes[92].attribute[1].g.node[80].input[:],  # Input tensor
                outputs=["squeeze_01"],  # Output tensor
                # axes=[1],  # Specify the axes to squeeze
            )
            # new_nodes[92].attribute[0].g.node[75].input[:]=new_nodes[92].attribute[0].g.node[76].input[:]+["squeeze_00"]
            # new_nodes[92].attribute[0].g.node.insert(76, squeeze_node0)
            # new_nodes[92].attribute[1].g.node[81].input[:] = new_nodes[92].attribute[1].g.node[81].input[:] + [
            #     "squeeze_01"]
            # new_nodes[92].attribute[1].g.node.insert(81, squeeze_node1)
            # ss=[i for i in new_nodes[92].attribute[0].g.node[76].input[:] if not('e_8_' in i)]
            # new_nodes[92].attribute[0].g.node[76].input[:] =ss
            # del(new_nodes[92].attribute[0].g.node[75])
            # ss = [i for i in new_nodes[92].attribute[1].g.node[81].input[:] if not ('e_8_' in i)]
            # new_nodes[92].attribute[1].g.node[81].input[:] = ss
            # del (new_nodes[92].attribute[1].g.node[80])

            a=1
        else:
            # Keep other nodes in the model as they are
            new_nodes.append(node)
    # new_input = helper.make_tensor_value_info(new_nodes[35].name, onnx.TensorProto.FLOAT, None)

    new_nodes[-1].input[:]=new_nodes[-3].output[:]
    new_nodes[-2].input[:] = new_nodes[-4].output[:]
    simplified_inputs = [i for i in model.graph.input]
    simplified_outputs = [i for i in model.graph.output]
    # # simplified_nodes.append(identity_node)
    generic_output_tensor = helper.make_tensor_value_info(
        "generic_output",  # Name of the new output
        onnx.TensorProto.INT64,  # Data type
        [1, 128]  # Shape: None for dynamic shapes
    )

    simplified_graph = helper.make_graph(
        nodes=new_nodes ,
        name="simplified_graph",
        inputs=simplified_inputs,
        outputs=simplified_outputs,
        initializer=model.graph.initializer if hasattr(model.graph, 'initializer') else [],
        value_info=model.graph.value_info if hasattr(model.graph, 'value_info') else []
    )
    # nn=34
    # new_nodes[nn].output[:]=[new_input.name]
    # new_nodes[2].input[:]=[]
    # ss =[i for i in new_nodes[35].input if not (i==new_nodes[1].output[0])]
    # ss.append("input")
    # new_nodes[35].input[:] =ss
    # simplified_graph = helper.make_graph(
    #     nodes=new_nodes[:nn+1],
    #     name="simplified_graph",
    #     inputs=simplified_inputs,
    #     # outputs=[generic_output_tensor],
    #     outputs=[new_input]
    # )

    simplified_model = helper.make_model(simplified_graph, producer_name="incremental_simplified_model")
    onnx.checker.check_graph(simplified_model.graph)
    onnx.checker.check_model(simplified_model)
    onnx.save(simplified_model,"my_new_model.onnx")
    ort.InferenceSession("my_new_model.onnx")
    # ort.InferenceSession("my_new_model.onnx")
    print ("ok")

    exit(44)
    for node in new_nodes:
        for i, input_name in enumerate(node.input):
            # Ensure input nodes are connected correctly
            if input_name in input_names:
                continue  # This input already exists, no need to modify
            elif input_name == original_if_node.input[0]:
                # Map this to the first input of the original 'If' node
                node.input[i] = original_if_node.input[0]

    # Extend the model with new nodes and outputs
    del model.graph.node[:]  # Clear existing nodes
    model.graph.node.extend(new_nodes)  # Add all new nodes
    model.graph.output.extend(new_outputs)  # Add new outputs for split branches

    return model





# Load your ONNX model
model = onnx.load('trial88.onnx')
model1 =onnx.load("silero_vad.onnx")
# Call the function to split the 'If' node
model = split_if_node(model)
onnx.checker.check_graph(model.graph)
# Save the updated model
onnx.save(model, 'updated_model.onnx')

print("Model with split 'If' nodes saved as 'updated_model.onnx'")
