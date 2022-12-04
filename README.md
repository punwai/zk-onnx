### INTRODUCTION

ZK-ONNX allows any machine model that is expressed as a computational graph in ONNX to be compiled into a zero-knowledge proof. However, due to the heavy computational costs of proving floating point arithmetic in prime fields, ZK-ONNX will only compile models and any floating-point machine learning model will simply fail to be compiled.
