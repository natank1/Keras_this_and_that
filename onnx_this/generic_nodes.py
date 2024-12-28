import onnx
import onnxruntime as ort
from onnx import helper, shape_inference,TensorProto

cond0 = "node.input[0]" #bring the value
where_node = onnx.helper.make_node(
                    "Where",
                    inputs=["cond0", "then_out.name", "else_out.name"], #values
                    outputs=[final_output_name]
                )
squeeze_node = helper.make_node(
    "Squeeze",              # Node type
    inputs=["input_tensor"], # Input tensor
    outputs=["output_tensor"], # Output tensor
    axes=[1],               # Specify the axes to squeeze
)
concat_node = helper.make_node(
    'Concat',
    # inputs=[node.output[0] for node in graph.node if len(node.output) > 0],#put whateveru wish
    outputs=['combined_output'],
    # to=onnx.TensorProto.FLOAT,
    axis=0  # Adjust this axis as needed
)
identity_node = helper.make_node(
        "Identity",
        inputs=["If_0_combined_output"],  # Connect to the existing node's output
        outputs=["generic_output"],  # New output name
        name="GenericOutputNod")

generic_output_tensor = helper.make_tensor_value_info(
    "generic_output",  # Name of the new output
    TensorProto.FLOAT,  # Data type
    [1, 128]  # Shape: None for dynamic shapes
)
cast_node_0 = helper.make_node(
    "Cast",  # Cast operator
    ["If_0_combined_output_0"],  # Input tensor from If node's output
    ["If_0_combined_output_0_cast"],  # New output tensor after casting
    to=TensorProto.FLOAT  # Convert to float type (or the required type for Split)
)
cast_node = helper.make_node(
    'Cast',
    inputs=['input_int64'],
    outputs=['input_int64_casted'],
    to=onnx.TensorProto.FLOAT
)

cast_node_1 = helper.make_node(
    "Cast",  # Cast operator
    ["If_0_combined_output_1"],  # Input tensor from If node's second output
    ["If_0_combined_output_1_cast"],  # New output tensor after casting
    to=TensorProto.INT64  # Convert to float type (or the required type for Split)
)
