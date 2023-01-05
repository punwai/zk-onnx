#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import io

import torch
import onnx
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.utils
from onnx import numpy_helper
from torch.onnx import OperatorExportTypes

# import torch.onnx.symbolic_registry as sym_registry

# import module

try:
    import tensorflow as tf  # noqa
    import tf2onnx

    TF_AND_TF2ONNX = True
except ImportError:
    TF_AND_TF2ONNX = False


def from_onnx(onnx_string_or_file):
    """
    Converts an ONNX model serialized in an `onnx_string_or_file` to a CrypTen el.
    """
    onnx_model = _load_onnx_model(onnx_string_or_file)
    return _to_crypten(onnx_model)


def from_pytorch(pytorch_model, dummy_input):
    """
    Converts a PyTorch model `pytorch_model` into a CrypTen model by tracing it
    using the input `dummy_input`.
    """

    # construct CrypTen model:
    f = _from_pytorch_to_bytes(pytorch_model, dummy_input)
    crypten_model = from_onnx(f)
    f.close()

    # set model architecture to export model back to pytorch model
    crypten_model.pytorch_model = copy.deepcopy(pytorch_model)

    # make sure training / eval setting is copied:
    crypten_model.train(mode=pytorch_model.training)
    return crypten_model


def from_tensorflow(tensorflow_graph_def, inputs, outputs):
    """
    Function that converts Tensorflow model into CrypTen model based on
    https://github.com/onnx/tensorflow-onnx/blob/master/tf2onnx/convert.py
    The model is returned in evaluation mode.
    Args:
        `tensorflow_graph_def`: Input Tensorflow GraphDef to be converted
        `inputs`: input nodes
        `outputs`: output nodes
    """
    raise DeprecationWarning(
        "crypten.nn.from_tensorflow is deprecated. ",
        "CrypTen will no longer support model conversion from TensorFlow.",
    )


def _from_pytorch_to_bytes(pytorch_model, dummy_input):
    """
    Returns I/O stream containing ONNX graph for `pytorch_model` traced with
    input `dummy_input`.
    """

    # first export is only used to obtain the PyTorch-to-ONNX symbolic registry:
    with io.BytesIO() as f:
        _export_pytorch_model(f, pytorch_model, dummy_input)

    # update ONNX symbolic registry with CrypTen-specific functions:
    # _update_onnx_symbolic_registry()

    # export again so the graph is created with CrypTen-specific registry:
    f = io.BytesIO()
    f = _export_pytorch_model(f, pytorch_model, dummy_input)
    f.seek(0)
    return f


def _export_pytorch_model(f, pytorch_model, dummy_input):
    """
    Returns a binary I/O stream containing ONNX-exported pytorch_model that was
    traced with input `dummy_input`.
    """
    kwargs = {
        "do_constant_folding": False,
        "export_params": True,
        "input_names": ["input"],
        "operator_export_type": OperatorExportTypes.ONNX,
        "output_names": ["output"],
    }
    torch.onnx.export(pytorch_model, dummy_input, f, **kwargs)
    return f

# This class represents the State of the transpiler during the transpilation.
class TranspilerState():
    def __init__(self):
        # Create a K-V store between variable name and its information
        variables = {}

class Gemm():
    @staticmethod
    def process(tensors_map, node):
        print("###############")
        print("PROCESSING GEMM NODE")
        print("###############")
        print(node)

        # GEMM
        # output = \alpha AB + \beta C

        # Try to find this, if it does not exist then... RIP
        A = node.input[0]
        B = node.input[1]
        C = node.input[2]
        print(tensors_map[node.input[1]])

# TODO: Change Tensors Map
class DataNode():
    def __init__(self, dim, name):
        self.dim = dim
        self.name = name
 
class Relu():
    @staticmethod
    def process(tensors_map, node):
        print("###############")
        print("PROCESSING RELU NODE")
        print("###############")
        print(node)

        inp = node.input[0]
        out = node.output[0]

        tensors_map[out] = out

# Define the method for dispatching a component to a function
OPS_HANDLER = {
    "Gemm": Gemm,
    "Relu": Relu
}

# Edit this function to compile to ZK proofs instead
def _to_crypten(onnx_model):
    input_names, output_names = _get_input_output_names(onnx_model)
    assert len(output_names) == 1, "Only one output per model supported."

    tensors_map = {}

    for node in onnx_model.graph.input:
        print(node)
        tensors_map[node.name] = node
        
    for node in onnx_model.graph.output:
        tensors_map[node.name] = node

    for node in onnx_model.graph.initializer:
        tensors_map[node.name] = DataNode(node.dims, node.name)
        print(tensors_map[node.name])

    # loop over all nodes:
    for node in onnx_model.graph.node:
        attributes = {attr.name: _get_attribute_value(attr) for attr in node.attribute}
        # crypten_class = _get_operator_class(node.op_type, attributes)
        print(node)
        
        # OPS_HANDLER[node.op_type].process(tensors_map, node)
        
        # input_names = list(node.input)
        # for inp in input_names:
        #     if inp not in tensors_map:
        #         print(inp)
        #         raise RuntimeError("Input {} not found in variables".format(inp))
        #     input_dim = tensors_map[inp]

        # output_names = list(node.output)

def _load_onnx_model(onnx_string_or_file):
    """
    Loads ONNX model from file or string.
    """
    if hasattr(onnx_string_or_file, "seek"):
        onnx_string_or_file.seek(0)
        return onnx.load(onnx_string_or_file)
    return onnx.load_model_from_string(onnx_string_or_file)


def _get_input_output_names(onnx_model):
    """
    Return input and output names of the ONNX graph.
    """
    input_names = [input.name for input in onnx_model.graph.input]
    output_names = [output.name for output in onnx_model.graph.output]
    assert len(input_names) >= 1, "number of inputs should be at least 1"
    assert len(output_names) == 1, "number of outputs should be 1"
    return input_names, output_names

def _get_attribute_value(attr):
    """
    Retrieves value from an ONNX attribute.
    """
    if attr.HasField("f"):  # floating-point attribute
        return attr.f
    elif attr.HasField("i"):  # integer attribute
        return attr.i
    elif attr.HasField("s"):  # string attribute
        return attr.s  # TODO: Sanitize string.
    elif attr.HasField("t"):  # tensor attribute
        return torch.from_numpy(numpy_helper.to_array(attr.t))
    elif len(attr.ints) > 0:
        return list(attr.ints)
    elif len(attr.floats) > 0:
        return list(attr.floats)
    raise ValueError("Unknown attribute type for attribute %s." % attr.name)

def _get_operator_class(node_op_type, attributes):
    print(node_op_type)

# 1. Turn into ONNX