CXX = g++
CXXFLAGS = -std=c++17 -O2 -Isrc
PROTOBUF_CFLAGS = `pkg-config --cflags protobuf`
PROTOBUF_LIBS   = `pkg-config --libs protobuf`

SRC_DIR = src
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:.cpp=.o)
BIN = inference

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) -o $@ $^ $(PROTOBUF_LIBS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(PROTOBUF_CFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(BIN)
