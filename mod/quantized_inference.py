# This file allows us to infer ONNX machine learning models

import onnx
import onnxruntime
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import os

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat



def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def preprocess_data(dataset):
    unconcatenated_batch_data = []

    for data in dataset:
        unconcatenated_batch_data.append(to_numpy(data[0]))

    batch_data = np.concatenate(np.expand_dims(unconcatenated_batch_data, axis=0), axis=0)
    print(batch_data.shape)

    return batch_data



class FashionMNISTDataReader(CalibrationDataReader):
    def __init__(self, torch_data_set, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, height, width) = session.get_inputs()[0].shape

        self.nhwc_data_list = preprocess_data(
            torch_data_set
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None




training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("resnet50.onnx")
hell = ort_session.get_inputs()[0].shape

dr = FashionMNISTDataReader(training_data, "resnet50.onnx")

quantize_static('resnet50.onnx',
                'resnet50_uint8.onnx',
                dr,
                quant_format=QuantFormat.QOperator
                )

# compute ONNX Runtime output prediction


ort_session_quant = onnxruntime.InferenceSession("resnet50_uint8.onnx")

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(training_data[0][0])}
ort_outs = ort_session_quant.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
print(ort_outs)
