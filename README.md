### What's this?
This repo summarizes my investigation into whether `TensorRT` could speed up a machine learning (ML) model's inference.

The ML model in question is an EfficientNetV2 model that was trained by Thorn and procured during [GSD#36523](https://vault.shopify.io/gsd/projects/36523).

We first learned about TensorRT through [this WxM post](https://shopify.workplace.com/groups/mlacc/permalink/968902058176200/), and saw blog posts touting that inference with a TensorRT engine was > [4x](https://beam.apache.org/documentation/ml/tensorrt-runinference/) or [9x](https://developer.nvidia.com/blog/simplifying-and-accelerating-machine-learning-predictions-in-apache-beam-with-nvidia-tensorrt/) faster than Tensorflow model.
