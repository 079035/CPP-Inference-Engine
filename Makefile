# Compiler
CXX = g++
CUDA_PATH = /usr/local/cuda

# Source and object files
SRCS = $(wildcard src/*.cpp) $(wildcard src/*.cc)
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cc=.o)

# Output
TARGET = inference_engine

# Flags
CXXFLAGS = -std=c++17 -O2 -Isrc `pkg-config --cflags protobuf`
LDFLAGS = `pkg-config --libs protobuf`

# Add CUDA support if USE_CUDA is defined
ifdef USE_CUDA
    CUDA_SRCS = src/operators_cuda.cu
    CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
    CXXFLAGS += -DUSE_CUDA -I$(CUDA_PATH)/include
    LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
endif

ALL_OBJS = $(OBJS) $(CUDA_OBJS)
NVCC = nvcc

install:
	pip install -r requirements.txt

train:
	python3 train.py

# Build rules
all: $(TARGET)

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/%.o: src/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/%.o: src/%.cu
	$(NVCC) -c $< -o $@ -std=c++17 -O2 -Isrc -I$(CUDA_PATH)/include -DUSE_CUDA

$(TARGET): $(ALL_OBJS)
	$(CXX) $(ALL_OBJS) -o $@ $(LDFLAGS)

benchmark.o: benchmark.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

benchmark: benchmark.o $(filter-out src/main.o,$(ALL_OBJS))
	$(CXX) benchmark.o $(filter-out src/main.o,$(ALL_OBJS)) -o benchmark $(LDFLAGS)

clean:
	rm -f src/*.o $(TARGET) *.o
