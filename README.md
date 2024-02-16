### What's this?
This repo summarizes my investigation into whether `TensorRT` could speed up a machine learning (ML) model's inference.

The ML model in question is an EfficientNetV2 model that was trained by Thorn and procured during [GSD#36523](https://vault.shopify.io/gsd/projects/36523).

We first learned about TensorRT through [this WxM post](https://shopify.workplace.com/groups/mlacc/permalink/968902058176200/), and saw blog posts touting that inference with a TensorRT engine was > [4x](https://beam.apache.org/documentation/ml/tensorrt-runinference/) or [9x](https://developer.nvidia.com/blog/simplifying-and-accelerating-machine-learning-predictions-in-apache-beam-with-nvidia-tensorrt/) faster than Tensorflow model.

### TL;DR

Inference speed for pre-trained EfficientNetV2 model:

| setup | on batches of 10 384x384x3 images | on batches of 32 384x384x3 images |
|-------|-----------------------------------|-----------------------------------|
| TensorRT engine + GPU | min=0.0666 s      | min=0.206 s                       |
|                       | med=0.0684 s      | med=0.207 s                       |
|                       | max=0.699 s       | max=0.887 s                       |
| Onnxruntime-gpu + GPU + TensorrtExecutionProvider | min=0. s      | min=0. s                       |
|                                                   | med=0. s      | med=0. s                       |
|                                                   | max=0. s      | max=0. s                       |
| Onnxruntime-gpu + GPU + CUDAExecutionProvider | min=0. s      | min=0. s                       |
|                                               | med=0. s      | med=0. s                       |
|                                               | max=0. s      | max=0. s                       |

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

1. Creating a GCE instance with a Nvidia T4 GPU and CUDA toolkit and drivers
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
docker build -f tensor_rt.dockerfile -t tensor_rt 
```

##### Building the TensorRT engine

```bash
trtexec --onnx=/mnt/models/csam_model.onnx \
  --minShapes=input:1x384x384x3 \
  --optShapes=input:10x384x384x3 \
  --maxShapes=input:32x384x384x3 \
  --shapes=input:10x384x384x3 \
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

And ran the following python code from a Python shell:

```python
import os 
from time import time

import numpy as np
import tensorrt as trt 
import pycuda.driver as cuda

cuda.init()
device = cuda.Device(0)
cuda_driver_context = device.make_context()

def load_engine(trt_runtime, engine_path):
    trt.init_libnvinfer_plugins(None, "")             
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

def create_engine_from_onnx_file( onnx_model_file_path = "/mnt/models/csam_model.onnx",
                                  images_per_batch = 1,
                                  image_shape = (384, 384, 3)):
    assert os.path.isfile(onnx_model_file_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_model_file_path)
    assert success
    with builder.create_builder_config() as config:        
        profile = builder.create_optimization_profile()     
        # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/OptimizationProfile.html#tensorrt.IOptimizationProfile.set_shape
        profile.set_shape(
            input = "input_1", 
            min = (images_per_batch, *image_shape), 
            opt = (images_per_batch, *image_shape), 
            max = (images_per_batch, *image_shape)
        )
        config.add_optimization_profile(profile)
        engine = builder.build_engine(network, config)
    assert engine is not None
    return engine

def run_inference(engine, input_image):
    # ref https://github.com/NVIDIA/TensorRT/blob/main/quickstart/SemanticSegmentation/tutorial-runtime.ipynb
    with engine.create_execution_context() as context:
        context.set_binding_shape(engine.get_binding_index("input_1"), input_image.shape)
        # Allocate host and device buffers
        bindings = []
        for binding in engine:
            binding_idx = engine.get_binding_index(binding)
            size = trt.volume(context.get_binding_shape(binding_idx))
            # dtype = trt.nptype(engine.get_binding_dtype(binding))
            dtype = np.float32
            if engine.binding_is_input(binding):
                input_buffer = np.ascontiguousarray(input_image)
                input_memory = cuda.mem_alloc(input_image.nbytes)
                bindings.append(int(input_memory))
            else:
                output_buffer = cuda.pagelocked_empty(size, dtype)
                output_memory = cuda.mem_alloc(output_buffer.nbytes)
                bindings.append(int(output_memory))
        start = time()
        stream = cuda.Stream()
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(input_memory, input_buffer, stream)
        # Run inference
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
        # Synchronize the stream
        stream.synchronize()
        print(time() - start)
    return output_buffer, time() - start

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

images_per_batch = 32
image_shape = (384, 384, 3)
np.random.seed(42)
input_image = np.random.randint(0, 255, (images_per_batch, *image_shape)).astype(np.float32)
engine = load_engine(runtime, "/mnt/tensorrt_engines/csam_model.onnx.batch1_to_32.trt")

num_runs = 10
inference_runtimes = np.zeros(num_runs)
for i in range(num_runs):
    out, runtime = run_inference(engine,input_image)
    inference_runtimes[i] = runtime

print(f"runtime on {images_per_batch} images:\nmin={inference_runtimes.min()}, max={inference_runtimes.max()}, med={np.median(inference_runtimes)}")
cuda_driver_context.pop()
exit()
```

Output: 
- `images_per_batch=32`:     runtime on 32 images:
    min=0.2056584358215332, max=0.8869142532348633, med=0.20708370208740234
-   `images_per_batch=10`:   runtime on 10 images:
    min=0.06661248207092285, max=0.6989891529083252, med=0.0684133768081665


### Benchmarking `onnxruntime-gpu`
I created a separate dockerfile to test out our .onnx model's inference speed using only `onnxruntime-gpu`.

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


