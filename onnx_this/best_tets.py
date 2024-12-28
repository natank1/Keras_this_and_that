import matplotlib.pyplot as plt
import networkx as nx
import onnx
import numpy as np
from onnx import shape_inference

# Load your model
import onnxruntime as ort

def get_model_input():
    print("Original Model Inputs:")
    # create inpuys
    input_data = np.random.randn(1, 512).astype(np.float32)  # Adjust the 100 as needed
    state_data = np.random.randn(2,1,128).astype(np.float32)  # Adjust the 1 as needed
    sr_data = np.array(16000).astype(np.int64)  # Assuming 16000 as sample rate

    # Create input dictionary
    ort_inputs = {
        "input": input_data,
        "state": state_data,
        "sr": sr_data
    }
    return ort_inputs

# dd =onnx.load("emulated_model.onnx")
# onnx.checker.check_model(dd)
# onnx.checker.check_graph(dd.graph)

# for node in dd.graph.node:
#     if node.op_type == "Pad":
#         print(f"Pad node found: {node.name}")
#         for attr in node.attribute:
#             if attr.name == "pads":
#                 print(f"Pads attribute: {onnx.numpy_helper.to_array(attr.t)}")
# inferred_model = shape_inference.infer_shapes(dd)
# for node in inferred_model.graph.node:
#     if node.op_type == "Pad":
#         print(f"Pad Node: {node.name}")
#
#
# # onnx.save(inferred_model,"blabla .onnx")
# # ses1=ort.InferenceSession("blabla .onnx")
# # ses1.run(None, get_model_input())
# # exit(22)
if __name__=='__main__':
    ses1=ort.InferenceSession("my_new_model.onnx")
    dd=onnx.load("my_new_model.onnx")
    # dd1 = onnx.load("trial88.onnx")
    onnx.checker.check_model(dd)
    onnx.checker.check_graph(dd.graph)
    print("ggg")

    xx=ses1.run(None, get_model_input())
    # print ("hahah ",xx)
    from generic_functions import visua_alll
    visua_alll(dd)

