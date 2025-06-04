SRC_DIR = src
SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cc)
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cc=.o)   # Also convert .cc to .o
CXXFLAGS = -std=c++17 -O2 -Isrc `pkg-config --cflags protobuf`
LDFLAGS = `pkg-config --libs protobuf`
INPUT_FILE ?= inputs/image_0.ubyte

TARGET = inference_engine

SHARED_OBJS = $(filter-out $(SRC_DIR)/main.o, $(OBJS))

all: $(TARGET) benchmark

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

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
