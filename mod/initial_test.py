import torch
import convert
import io
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import onnxruntime as onnxrt

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_dataloader = DataLoader(test_data, batch_size=64)

onnx_session= onnxrt.InferenceSession("resnet50.onnx")
onnx_inputs = {
    onnx_session.get_inputs()[0].name:
        to_numpy(img)
}

onnx_output = onnx_session.run(None, onnx_inputs)

img_label = onnx_outputort_outs[0]
