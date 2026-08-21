# Pure single-GPU Inamuro LBM CUDA build.

NVCC ?= nvcc
CUDA_ARCH ?= sm_120
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo -arch=$(CUDA_ARCH)
INCLUDES := -I.

GPU_SRC := InamuroCUDA.cu
CPP_SRC := Inamuro.cpp common.cpp
HEADERS := InamuroCUDA.hpp Inamuro.hpp LBMBase.hpp common.hpp

TARGET_GPU := lbm_gpu
TARGET_COMPARE := lbm_compare
PARAMS_SMALL := params_small.in
PARAMS_LAYOUT := params_layout.in

.PHONY: all gpu compare test smoke memcheck initcheck clean help

all: gpu compare

gpu: $(TARGET_GPU)

compare: $(TARGET_COMPARE)

$(TARGET_GPU): $(GPU_SRC) $(CPP_SRC) main.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) main.cpp -o $@

$(TARGET_COMPARE): $(GPU_SRC) $(CPP_SRC) compare_test.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -DLBM_TESTING $(INCLUDES) $(GPU_SRC) $(CPP_SRC) compare_test.cpp -o $@

test: $(TARGET_COMPARE)
	./$(TARGET_COMPARE) --params $(PARAMS_LAYOUT) --steps 1
	./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 1
	./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 2
	./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 10

smoke: $(TARGET_GPU)
	./$(TARGET_GPU) --params $(PARAMS_SMALL) --steps 1

memcheck: $(TARGET_COMPARE)
	compute-sanitizer --tool memcheck --error-exitcode=3 ./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 1

initcheck: $(TARGET_COMPARE)
	compute-sanitizer --tool initcheck --error-exitcode=3 ./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 1

clean:
	rm -f $(TARGET_GPU) $(TARGET_COMPARE) *.o

help:
	@echo "make [CUDA_ARCH=sm_XX] all       Build runner and comparison gate"
	@echo "make [CUDA_ARCH=sm_XX] test      Run layout and 1/2/10-step CPU/GPU gates"
	@echo "make [CUDA_ARCH=sm_XX] smoke     Run one GPU-only step"
	@echo "make [CUDA_ARCH=sm_XX] memcheck  Run CUDA memory checker"
	@echo "make [CUDA_ARCH=sm_XX] initcheck Run CUDA initialization checker"
