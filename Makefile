SRC_DIR = src
SRCS = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cc)
OBJS = $(SRCS:.cpp=.o)
OBJS := $(OBJS:.cc=.o)   # Also convert .cc to .o
CXXFLAGS = -std=c++17 -O2 -Isrc `pkg-config --cflags protobuf`
LDFLAGS = `pkg-config --libs protobuf`

TARGET = inference_engine

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)
