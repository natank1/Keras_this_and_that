import onnx
import onnxruntime as ort
from onnx import helper, shape_inference,TensorProto
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from onnx import version_converter
def create_mode(inputs,nodes,outputs ):

    simplified_graph = helper.make_graph(
        nodes=nodes,
        name="simplified_graph",
        inputs=inputs,
        outputs=outputs,
    )
    simplified_model = helper.make_model(simplified_graph, producer_name="incremental_simplified_model")
    return simplified_model



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
def visualize_graph(nx_graph):
        plt.figure(figsize=(10, 8))
        nx.draw(nx_graph, with_labels=True, node_size=3000, node_color="lightblue", font_size=10)
        plt.show()

def visua_alll(model)  :
    nx_graph = nx.DiGraph()
    for node in model.graph.node:
        for input_name in node.input:
            for output_name in node.output:
                print(f"Adding edge: {input_name} -> {output_name}")
                nx_graph.add_edge(input_name, output_name)
    visualize_graph(nx_graph)
def conver_version(onnx_model):

    target_opset_version = 18  # Change this to your desired opset version

    # Convert the model to the specified opset version
    converted_model = version_converter.convert_version(onnx_model, target_opset_version)

    opset_version = converted_model.opset_import[0].version if len(converted_model.opset_import) > 0 else None
    print(f"Model opset version: {opset_version}")
    onnx.checker.check_model(converted_model)
    onnx.checker.check_graph(converted_model.graph)
    return