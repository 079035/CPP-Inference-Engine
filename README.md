Required Programs to Run:

- conda
- protoc
- pkgconf
- make

### Commands to Run Inference:

```
conda create --name "inference" python=3.12
conda activate inference
pip install -r requirements.txt
protoc --cpp_out=src/ onnx-ml.proto
make clean && make
./inference_engine models/mnist_model.onnx inputs/image_0.ubyte
```
