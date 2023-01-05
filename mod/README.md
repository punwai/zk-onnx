### Introduction

ZK-ONNX allows any machine model that is expressed as a computational graph in ONNX to be compiled into a zero-knowledge proof. However, due to the heavy computational costs of proving floating point arithmetic in prime fields, ZK-ONNX will only compile models and any floating-point machine learning model will simply fail to be compiled.

##### Quantization
Because it is very expensive to prove floating point arithmetic in a finite field, we exclusively work with quantized neural networks. Quantization is a standard approach in reducing the computation cost of neural networks. In Torch, neural networks are trained as a FP32 network, but can then later be converted into INT8 quantization. 