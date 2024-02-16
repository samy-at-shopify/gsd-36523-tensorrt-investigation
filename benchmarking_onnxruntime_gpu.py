from time import time 

import onnx
import numpy as np
import onnxruntime as ort

# load and check model
onnx_model_filepath = "/mnt/models/csam_model.onnx"
onnx_model = onnx.load(onnx_model_filepath)
onnx.checker.check_model(onnx_model)

# setup inference session 
ort_sess = ort.InferenceSession(onnx_model_filepath, providers=['CUDAExecutionProvider']) # change to 'TensorrtExecutionProvider'

# setup inference experiments
batch_size = 10
np.random.seed(42)
sample_images = np.random.rand(batch_size, 384, 384, 3).astype(np.float32)
payload = { "input_1": sample_images}

num_repeats = 10
inference_runtimes = np.zeros(num_repeats,)
for i in range(10):
    start = time()
    output = ort_sess.run(None, payload)
    inference_runtimes[i] = (time() - start)

print(output)
print(f"runtime on {batch_size} images:\nmin={inference_runtimes.min()}, max={inference_runtimes.max()}, med={np.median(inference_runtimes)}")
print(ort.get_device())
