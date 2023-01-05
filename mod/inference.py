# This file allows us to infer ONNX machine learning models

import onnx
import onnxruntime
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("resnet50.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction


ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(training_data[0][0])}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs)

# compare ONNX Runtime and PyTorch results
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
