# Minimal GPU foundation Makefile for PINN/Poisson restart work.

NVCC ?= nvcc
CUDA_ARCH ?= sm_89

NVCCFLAGS ?= -std=c++17 -O3 -arch=$(CUDA_ARCH)
INCLUDES := -I. -I../CPU

GPU_SRC := InamuroCUDA.cu
CPP_SRC := Inamuro.cpp InamuroSolver.cpp common.cpp

TARGET_GPU := lbm_gpu
TARGET_COMPARE := lbm_compare
PARAMS_SMALL := params_small.in

.PHONY: all gpu compare test test-onepass run clean help

all: gpu compare

gpu: $(TARGET_GPU)

compare: $(TARGET_COMPARE)

$(TARGET_GPU): $(GPU_SRC) $(CPP_SRC) main.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) main.cpp -o $@

$(TARGET_COMPARE): $(GPU_SRC) $(CPP_SRC) compare_test.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) compare_test.cpp -o $@

test: $(TARGET_COMPARE)
	./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 1 --poisson split

test-onepass: $(TARGET_COMPARE)
	./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 1 --poisson onepass --pressure-boundary fused

run: $(TARGET_GPU)
	./$(TARGET_GPU) --mode gpu --params $(PARAMS_SMALL) --steps 20 --poisson onepass --pressure-boundary fused --no-roofline

clean:
	rm -rf $(TARGET_GPU) $(TARGET_COMPARE) *.o __pycache__

help:
	@echo "Targets:"
	@echo "  make gpu          Build GPU runner"
	@echo "  make compare      Build CPU/GPU comparison gate"
	@echo "  make test         Run 1-step split CPU/GPU gate"
	@echo "  make test-onepass Run 1-step onepass CPU/GPU gate"
	@echo "  make run          Run small GPU smoke case"
	@echo "  make clean        Remove local build outputs"
