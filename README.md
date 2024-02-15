### What's this?
This repo summarizes my investigation into whether `TensorRT` could speed up a machine learning (ML) model's inference.

The ML model in question is an EfficientNetV2 model that was trained by Thorn and procured during [GSD#36523](https://vault.shopify.io/gsd/projects/36523).

We first learned about TensorRT through [this WxM post](https://shopify.workplace.com/groups/mlacc/permalink/968902058176200/), and saw blog posts touting that inference with a TensorRT engine was > [4x](https://beam.apache.org/documentation/ml/tensorrt-runinference/) or [9x](https://developer.nvidia.com/blog/simplifying-and-accelerating-machine-learning-predictions-in-apache-beam-with-nvidia-tensorrt/) faster than Tensorflow model.

### Glossary
##### Disclaimer: these aren't strict definitions - they are just here to convey what I mean when I use certain technical terms
- Deep Learning: the subdiscipline of Machine Learning involving deep neural networks (i.e., those with > 1 layer).
- Inference: the process of providing a machine learning model with some input and obtaining its output.
- Model: a machine learning model (not a model in the MVC sense). 
- TensorRT: an open-source deep learning framework made by Nvidia that aims to speed up deep neural networks' inference by optimizing different layers (e.g., fusing adjacent layers when possible).
- TensorRT engine: a file (or object in memory) representing a deep learning model that has been optimized with TensorRT. 
- Onnx: a machine learning framework with APIs in different languages and many backends that also allows to speed up deep neural networks' inference. Also comes with its own model format (.onnx). 
- Onnxruntime and Onnxruntime-gpu: The libraries required to run inference using an .onnx model.


#### Getting started with TensorRT
I followed [this guide](https://developer.nvidia.com/blog/simplifying-and-accelerating-machine-learning-predictions-in-apache-beam-with-nvidia-tensorrt/) to create a TensorRT engine from our ML model's .onnx file. This involved:

1. Creating a GCE instance with a Nvidia T4 GPU and CUDA toolkit and drivers
2. Creating a Firewall rule to ssh into the GCE instance
3. Uploading the .onnx file to the GCE instance
4. Building a Dockerfile based on Nvidia's official TensorRT images
5. Using the `trtexec` tool to create a TensorRT engine from the .onnx file (all from within the Nvidia/TensorRT Docker container)

##### Notes:
- The TensorRT python package allows to create a TensorRT engine from a .onnx file - I could have used that instead of the `trtexec` tool
- A TensorRT engine is 
