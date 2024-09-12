## What's this?
This repo summarizes my investigation into whether `TensorRT` could speed up a machine learning (ML) model's inference. The alternatives we considered were:

1. Running inference with the .onnx model on a GPU with the CUDAExecutionProvider
2. Running inference with the .onnx model on a GPU with the TensorrtExecutionProvider

The ML model in question is an EfficientNetV2 model that was trained by Thorn and procured during [GSD#36523](https://vault.shopify.io/gsd/projects/36523).

We first learned about TensorRT through [this WxM post](https://shopify.workplace.com/groups/mlacc/permalink/968902058176200/), and saw blog posts touting that inference with a TensorRT engine was > [4x](https://beam.apache.org/documentation/ml/tensorrt-runinference/) or [9x](https://developer.nvidia.com/blog/simplifying-and-accelerating-machine-learning-predictions-in-apache-beam-with-nvidia-tensorrt/) faster than Tensorflow model.

## TL;DR

We didn't see much improvement with converting our .onnx model to a TensorRT engine. However, we did see a ~3x speedup when we used 16-bit floating point quantization (even more when dealing with batches of 32 images!).

Inference speed for pre-trained EfficientNetV2 model:

| setup | on batches of 10 384x384x3 images | on batches of 32 384x384x3 images |
|-------|-----------------------------------|-----------------------------------|
| TensorRT engine + quantized + GPU | min=0.0209 s | min=0.0661 s |
|                                   | med=0.0342 s | med=0.0673 s |
|                                   | max=0.0374 s | max=0.101 s |
| TensorRT engine + GPU | min=0.0666 s      | min=0.206 s                       |
|                       | med=0.0684 s      | med=0.207 s                       |
|                       | max=0.699 s       | max=0.887 s                       |
| Onnxruntime-gpu + GPU + TensorrtExecutionProvider | min=0.0676 s      | min=0.207 s                       |
|                                                   | med=0.0684 s      | med=0.208 s                       |
|                                                   | max=86.057 s      | max=64.94 s                       |
| Onnxruntime-gpu + GPU + CUDAExecutionProvider | min=0.0808 s      | min=0.2496 s                       |
|                                               | med=0.0830 s      | med=0.252 s                       |
|                                               | max=4.856 s      | max=8.078 s                       |

### Glossary
##### Disclaimer: these aren't strict definitions - they are just here to convey what I mean when I use certain technical terms
- **Deep Learning**: the subdiscipline of Machine Learning involving deep neural networks (i.e., those with > 1 layer).
- **Inference**: the process of providing a machine learning model with some input and obtaining its output.
- **Model**: a machine learning model (not a model in the MVC sense). 
- **TensorRT**: an open-source deep learning framework made by Nvidia that aims to speed up deep neural networks' inference by optimizing different layers (e.g., fusing adjacent layers when possible).
- **TensorRT engine**: a file (or object in memory) representing a deep learning model that has been optimized with TensorRT. 
- **Onnx**: a machine learning framework with APIs in different languages and many backends that also allows to speed up deep neural networks' inference. Also comes with its own model format (.onnx). 
- **Onnxruntime and Onnxruntime-gpu**: The libraries required to run inference using an .onnx model without/with a Nvidia CUDA GPU.


### Getting started with TensorRT
I followed [this guide](https://developer.nvidia.com/blog/simplifying-and-accelerating-machine-learning-predictions-in-apache-beam-with-nvidia-tensorrt/) to create a TensorRT engine from our ML model's .onnx file. This involved:

1. Creating a GCE instance with a Nvidia T4 GPU and CUDA toolkit and drivers (see this video: https://videobin.shopify.io/v/e6oVbP, or [this GCE VM image](https://console.cloud.google.com/compute/machineImages/details/gsd36523-tensorrt-vm-with-gpu-image?project=shopify-commerce-trust))
2. Creating a Firewall rule to ssh into the GCE instance
3. Uploading the .onnx file to the GCE instance
4. Building a Dockerfile based on Nvidia's official TensorRT images
5. Using the `trtexec` tool to create a TensorRT engine from the .onnx file (all from within the Nvidia/TensorRT Docker container) ([link to `trtexec` documentation](https://github.com/NVIDIA/TensorRT/tree/main/samples/trtexec#tensorrt-command-line-wrapper-trtexec))


##### Contents of the Dockerfile I used instead of the suggested `tensor_rt.dockerfile`

```docker
ARG BUILD_IMAGE=nvcr.io/nvidia/tensorrt:22.09-py3

FROM ${BUILD_IMAGE}

ENV PATH="/usr/src/tensorrt/bin:${PATH}"

WORKDIR /workspace

RUN pip install --upgrade pip \
    && pip install torch>=1.7.1 \
    && pip install torchvision>=0.8.2 \
    && pip install pillow>=8.0.0 \
    && pip install transformers>=4.18.0 \
    && pip install cuda-python

ENTRYPOINT [ "bash" ]
````

##### Building the Docker container:

```bash
docker build -f tensor_rt.dockerfile -t tensor_rt .
```

##### Building the TensorRT engine

```bash
trtexec --onnx=/mnt/models/csam_model.onnx \
  --minShapes=input_1:1x384x384x3 \
  --optShapes=input_1:10x384x384x3 \
  --maxShapes=input_1:32x384x384x3 \
  --shapes=input_1:10x384x384x3 \
  --saveEngine=/mnt/tensorrt_engines/csam_model.onnx.batch1_to_32.trt \
  --useCudaGraph \
  --verbose
```

##### Notes:
- The TensorRT python package allows to create a TensorRT engine from a .onnx file - I could have used that instead of the `trtexec` tool
- Running `nvidia-smi` in the GCE instance

```bash
Thu Feb 15 23:44:18 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   63C    P0    27W /  70W |      0MiB / 15360MiB |     11%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

- Running `nvidia-smi` in the Docker container

```bash
Thu Feb 15 23:44:13 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   63C    P0    27W /  70W |      0MiB / 15360MiB |     11%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

### Benchmarking TensorRT 
I spun up my TensorRT container using:

```bash
docker run --rm -it --gpus all -v /home/samy_coulombe/:/mnt tensor_rt
```

And ran the [following Python code](https://github.com/samy-at-shopify/gsd-36523-tensorrt-investigation/blob/main/benchmarking_tensorrt_engine_gpu.py) from a Python shell.


### Benchmarking `onnxruntime-gpu`
I created a separate dockerfile to test out our .onnx model's inference speed using only `onnxruntime-gpu`, and ran the [following Python code](https://github.com/samy-at-shopify/gsd-36523-tensorrt-investigation/blob/main/benchmarking_onnxruntime_gpu.py) from a Python shell.

##### Contents of the Dockerfile:

```docker
ARG BUILD_IMAGE=nvcr.io/nvidia/tensorrt:22.09-py3

FROM ${BUILD_IMAGE} 

ENV PATH="/usr/src/tensorrt/bin:${PATH}"

WORKDIR /workspace

RUN pip install --upgrade pip \
    && pip install torch>=1.7.1 \
    && pip install torchvision>=0.8.2 \
    && pip install pillow>=8.0.0 \
    && pip install transformers>=4.18.0 \
    && pip install cuda-python \
    && pip install onnx \
    && pip install onnxruntime-gpu

ENTRYPOINT [ "bash" ]
```
