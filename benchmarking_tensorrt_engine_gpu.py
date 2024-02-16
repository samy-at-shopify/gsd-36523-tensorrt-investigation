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

images_per_batch = 10
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
