# 🧠 MNIST Inference Engine (C++)

A minimal, from-scratch C++ inference engine for running ONNX models—demonstrated on the classic MNIST digit recognition task.
Inspired by [Build Your Own Inference Engine](https://michalpitr.substack.com/p/build-your-own-inference-engine-from) and [vLLM](https://github.com/vllm-project/vllm).

---

## ✨ Features

- Loads ONNX models exported from PyTorch, TensorFlow, etc.
- Supports core ops: **Flatten**, **Gemm**, **ReLU**, **Add** (sufficient for fully-connected MNIST models)
- Pure C++ with only [protobuf](https://developers.google.com/protocol-buffers) as a dependency.
- Runs on ARM (Apple Silicon) and x86 Linux/macOS.

---

## 📦 Requirements

Install these before building:

- [conda](https://docs.conda.io/en/latest/) (for Python environment)
- [protobuf](https://developers.google.com/protocol-buffers) (for ONNX parsing)
- `make`
- `pkg-config` or `pkgconf`
- C++17 compiler (`clang++` or `g++`)
- `protoc` (Protocol Buffers compiler)
- [Python](https://www.python.org/) (for training/export, image preview)

---

## ⚡ Quick Start

#### 1. **Clone the repo**

```bash
git clone https://github.com/079035/CPP-Inference-Engine.git
cd CPP-Inference-Engine
```

#### 2. **Set up Python environment (optional, for utilities or training)**

```bash
conda create --name inference python=3.12
conda activate inference
pip install -r requirements.txt
```

#### 3. **Exporting Your Own Model**

Train a simple MNIST model in Python and export to ONNX (example in `train.py`):

```python
import torch
# ...train your model...
torch.onnx.export(model, dummy_input, "models/mnist_model.onnx", input_names=['onnx::Flatten_0'])
```

Ensure the input name matches your C++ code.

Train your model:

```bash
python train.py
```

#### 4. **Install Protobuf**

**macOS (Apple Silicon):**

```bash
brew install pkg-config
```

**Linux (Ubuntu):**

```bash
sudo apt-get install pkg-config
```

#### 5. **Generate ONNX C++ bindings**

Download ONNX proto file if not already present:

```bash
curl -O https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-ml.proto
```

Generate C++ files:

```bash
protoc --cpp_out=src/ onnx-ml.proto
```

You should now have `src/onnx-ml.pb.h` and `src/onnx-ml.pb.cc`.

#### 6. **Build the inference engine**

```bash
make clean && make
```

---

## 🏃‍♂️ Running Inference

You need:

- An exported ONNX model for MNIST (e.g., `models/mnist_model.onnx`)
- A test image in raw ubyte format (see below for details)

**Run:**

```bash
./inference_engine models/mnist_model.onnx inputs/image_0.ubyte
```

You should see output like:

```
Predicted class: 7
```

---

## 🖼️ Previewing Input Images (Optional)

You can visualize `.ubyte` images using the provided Python script:

```bash
python image_viewer.py inputs/image_0.ubyte
```

This opens a window showing the digit image (should be 28x28, grayscale).

---

## 🛠️ Project Structure

```
src/
  Graph.h               # Computational graph logic
  GraphUtils.h          # Topological sort
  InferenceEngine.h/cpp # Core inference engine
  main.cpp              # Main entrypoint
  Node.h                #
  onnx-ml.pb.h/cc       # Auto-generated from ONNX proto
  ONNXModelLoader.h     #
  operators.h/cpp       # Supported ONNX operators
  Tensor.h              #
inputs/
  image_0.ubyte         # Example MNIST image
  ...
  image_99.ubyte
models/
  mnist_model.onnx      # Trained ONNX model
image_viewer.py         # Python preview utility
Makefile
README.md
requirements.txt
train.py                # Python model training script
```

---

## 📚 Tips & Troubleshooting

- **"Missing input tensor" errors:**
  Make sure the input name in your `main.cpp` matches the model's expected input name (printed at runtime).

- **Output shape is `(784, 10)` instead of `(1, 10)`:**
  Ensure your `Flatten` operator reshapes the tensor as `(1, 784)`, not `(784)`.

- **Linker errors about ONNX/protobuf:**
  Make sure `onnx-ml.pb.cc` is being compiled and linked, and protobuf headers are found.

- **Protobuf version issues:**
  If you see header errors, ensure your `CXXFLAGS` includes the protobuf include directory (use `pkg-config --cflags protobuf`).

---

## 👥 Credits

Inspired by [Michal Pitr's tutorial](https://michalpitr.substack.com/p/build-your-own-inference-engine-from).

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---
