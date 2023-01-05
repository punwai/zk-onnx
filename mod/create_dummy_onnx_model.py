import torch
import convert
import io
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import onnxruntime as onnxrt

# Mock Model
# class PyTorchLinear(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.quant = torch.quantization.QuantStub()
#         self.fc1 = torch.nn.Linear(10, 5)
#         self.relu = torch.nn.ReLU()
#         self.fc2 = torch.nn.Linear(5, 1)
#         self.dequant = torch.quantization.DeQuantStub()

#     def forward(self, x):
#         # print(x)
#         x = self.quant(x)
#         # print(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.dequant(x)
#         return x

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        logits = self.dequant(logits)
        return logits

device = "cuda" if torch.cuda.is_available() else "cpu"

model = NeuralNetwork().to(device)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = torch.nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Train the model
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


torch.backends.quantized.engine = 'qnnpack'

# """Tests construction of crypten model from onnx graph"""
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
#  pytorch_model_fp_fused = torch.quantization.fuse_modules(pytorch_model_fp, [['linear', 'relu']])

pytorch_model_fp_prepared = torch.quantization.prepare(model)
model_int8 = torch.quantization.convert(pytorch_model_fp_prepared)

# # output = model_int8(dummy_input)

input_names = [ "input_image" ]
output_names = [ "output" ]

dummy_input = torch.randn(1, 28, 28)

torch.onnx.export(model,
                    dummy_input,
                    "resnet50.onnx",
                    verbose=False,
                    input_names=input_names,
                    output_names=output_names,
                    export_params=True,
                )


# calibration_dataset_path = args.calibrate_dataset
# dr = resnet50_data_reader.ResNet50DataReader(
#     calibration_dataset_path, input_model_path
# )


quantize_static(
    "resnet50.onnx",
    "resnet50_quantized.onnx"
)


# with io.BytesIO() as f:
#     f = convert._export_pytorch_model(f, model_int8, dummy_input)
#     f.seek(0)
#     with open("input_model.onnx", "wb") as file:
#         file.write(f.getbuffer())

#     crypten_model = convert.from_onnx(f)

# with io.BytesIO() as f:
#     f = convert._export_pytorch_model(f, pytorch_model_fp, dummy_input)
#     f.seek(0)
#     with open("input_model.onnx", "wb") as file:
#         file.write(f.getbuffer())

#     crypten_model = convert.from_onnx(f)
