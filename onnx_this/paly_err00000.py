import onnx
import onnxruntime as ort
from onnx import helper, shape_inference,TensorProto
import numpy as np



def get_model_input():
    print("Original Model Inputs:")
    # create inpuys
    input_data = np.random.randn(1, 512).astype(np.float32)  # Adjust the 100 as needed
    state_data = np.random.randn(2, 1, 128).astype(np.float32)  # Adjust the 1 as needed
    sr_data = np.array(16000).astype(np.int64)  # Assuming 16000 as sample rate

    # Create input dictionary
    ort_inputs = {
        "input": input_data,
        "state": state_data,
        "sr": sr_data
    }
    return ort_inputs

def  final_ccheck(verofy_mode, model_path):

    onnx.checker.check_model(verofy_mode)
    onnx.checker.check_graph(verofy_mode.graph)
    print ( "hecker ok")
    onnx.save(verofy_mode,model_path)
    session1 = ort.InferenceSession(model_path)
    inputs =get_model_input()
    print ("inputs")
    needed_inputs = {input.name: inputs[input.name] for input in session1.get_inputs()}

    ort_outputs = session1.run(None, needed_inputs)
    print(f"Outputs for simplified_before_if.onnx: {ort_outputs}")

    print ("cool")
    return

def build_simplified_model(original_model):
    """Builds a simplified ONNX model incrementally.

    Args:
        original_model: The original ONNX model.
        nodes_to_add: A list of node names (strings) to add from the If_0 subgraphs.
                     If None, only the part before If_0 is included.
    """


    graph = original_model.graph
    simplified_nodes = []
    simplified_inputs = []
    simplified_outputs = []
    fmodel = onnx.load("israel.onnx")
    fmodel1 =onnx.load("silero_vad.onnx")

    # ses1 = ort.InferenceSession("silero_vad.onnx")
    # print(ses1.run(None, get_model_input()))
    # exit(33)
    a=1

    for l,node in enumerate(graph.node):
        if l==3:
            continue
        if node.name == "If_0":
            added_if = True
            simplified_nodes.append(node)

            # node.output=['If_0_outputs_0', 'If_0_outputs_1']
            # break  # Stop before If_0
        else:
            simplified_nodes.append(node)


    simplified_nodes[2].output[:]= ['If_0_outputs_0', 'If_0_outputs_1']
    simplified_nodes[3].input[:]=['If_0_outputs_0']
    simplified_nodes[4].input[:] = ['If_0_outputs_1']

    simplified_inputs=[i for i in graph.input]
    simplified_outputs=[i for i in fmodel.graph.output]
    # # simplified_nodes.append(identity_node)

    simplified_graph = helper.make_graph(
        nodes=simplified_nodes,
        name="simplified_graph",
        inputs=simplified_inputs,
        outputs=simplified_outputs,
    )

    del (simplified_graph.node[2].attribute[1])
    del (simplified_graph.node[2].attribute[0])
    simplified_graph.node[2].attribute.extend(fmodel.graph.node[2].attribute)
    simplified_model = helper.make_model(simplified_graph, producer_name="incremental_simplified_model")
    onnx.checker.check_graph(simplified_model.graph)
    onnx.checker.check_model(simplified_model)
    onnx.save(simplified_model,"trial88.onnx")
    print ("gggggg")
    for node in simplified_model.graph.node:
        print (node.name,node.op_type )
    ses1=ort.InferenceSession("trial88.onnx")
    print (ses1.run(None, get_model_input()))
    print ("ffggg")
    print ("lalalala")
    exit(33)
original_model = onnx.load("fixed_model44.onnx")
build_simplified_model(original_model)
