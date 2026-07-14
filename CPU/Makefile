CXX := clang++

# 复用旧实验目录 venv 中已经安装好的 PyTorch C++ 头文件和动态库。
PYTORCH_PATH := /home/jzh/LBM/CPU/training/venv/lib/python3.12/site-packages/torch
PYTORCH_INCLUDE := $(PYTORCH_PATH)/include
PYTORCH_LIB := $(PYTORCH_PATH)/lib

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -pedantic \
	-I$(PYTORCH_INCLUDE) \
	-I$(PYTORCH_INCLUDE)/torch/csrc/api/include \
	-D_GLIBCXX_USE_CXX11_ABI=0
LDFLAGS := -L$(PYTORCH_LIB) \
	-Wl,-rpath,$(PYTORCH_LIB) \
	-ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda

TARGET := lbm_sim
SOURCES := main.cpp Inamuro.cpp InamuroSolver.cpp common.cpp
OBJECTS := $(SOURCES:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJECTS) $(TARGET)
