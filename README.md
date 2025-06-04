# MNIST Inference Engine (C++)

A minimal, from-scratch C++ inference engine for running ONNX models—demonstrated on the classic MNIST digit recognition task.
Inspired by [Build Your Own Inference Engine](https://michalpitr.substack.com/p/build-your-own-inference-engine-from).

## Requirements

Install these before building:

- [conda](https://docs.conda.io/en/latest/) (for Python environment)
- [protobuf](https://developers.google.com/protocol-buffers) (for ONNX parsing)
- `make`
- `pkg-config` or `pkgconf`

## Quick Start

#### 1. **Set up Python environment (optional, for utilities or training)**

```bash
make install
```

#### 2. **Exporting Your Own Model**

Train your model:

```bash
make trian
```

#### 3. **Generate ONNX C++ bindings**

Download ONNX proto file if not already present:

```bash
curl -O https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-ml.proto
```

Generate C++ files:

```bash
protoc --cpp_out=src/ onnx-ml.proto
```

You should already or now have `src/onnx-ml.pb.h` and `src/onnx-ml.pb.cc`.

#### 4. **Build the inference engine**

```bash
make clean && make
```

## Running Inference

You need:

- An exported ONNX model for MNIST (e.g., `models/mnist_model.onnx`)
- A test image in raw ubyte format (see below for details)

**Run:**

```bash
make run
```

or manually:

```bash
./inference_engine models/mnist_model.onnx inputs/image_0.ubyte
```

You should see output like:

```
Predicted class: 7
```

## Previewing Input Images

You can visualize `.ubyte` images using the provided Python script:

```bash
make show-image
```

This opens a window showing the digit image (should be 28x28, grayscale).

## Running Benchmark

```bash
make run-benchmark
```

### Baseline Benchmark Results

Results of running 10,000 sequential inference requests.

```
Iteration 1:
Total inference time: 5894.52 ms
Average inference time per run: 0.589452 ms

Iteration 2:
Total inference time: 5768.79 ms
Average inference time per run: 0.576879 ms

Iteration 3:
Total inference time: 5908.36 ms
Average inference time per run: 0.590836 ms
```

Averaging them, we can run about 1707 inferences per second.

## License

MIT License. See [LICENSE](LICENSE) for details.
