SRC_DIR = src
SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cc)
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cc=.o)
CU_OBJS = $(CU_SRCS:.cu=.o)
CXXFLAGS = -std=c++17 -O2 -Isrc -DUSE_CUDA `pkg-config --cflags protobuf`
NVCCFLAGS = -std=c++17 -O2 -Isrc -DUSE_CUDA
LDFLAGS = `pkg-config --libs protobuf` -lcudart -lcublas

INPUT_FILE ?= inputs/image_0.ubyte

TARGET = inference_engine

SHARED_OBJS = $(filter-out $(SRC_DIR)/main.o, $(OBJS) $(CU_OBJS))

all: $(TARGET) benchmark

$(TARGET): $(OBJS) $(CU_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET) benchmark benchmark.o

install:
	pip install -r requirements.txt

train:
	python train.py

run:
	./inference_engine models/mnist_model.onnx $(INPUT_FILE)

show-image:
	python image_viewer.py $(INPUT_FILE)

benchmark: $(SHARED_OBJS) benchmark.o
	$(CXX) $(CXXFLAGS) $(SHARED_OBJS) benchmark.o -o benchmark $(LDFLAGS)

run-benchmark:
	./benchmark models/mnist_model.onnx inputs
