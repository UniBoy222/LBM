# Production GPU solver Makefile.

NVCC ?= nvcc
CUDA_ARCH ?= sm_120

NVCCFLAGS ?= -std=c++17 -O3 -arch=$(CUDA_ARCH)
INCLUDES := -I. -I../CPU

GPU_SRC := InamuroCUDA.cu
CPP_SRC := Inamuro.cpp common.cpp

TARGET_GPU := lbm_gpu
PARAMS := params.in

.PHONY: all gpu run clean help

all: gpu

gpu: $(TARGET_GPU)

$(TARGET_GPU): $(GPU_SRC) $(CPP_SRC) main.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) main.cpp -o $@

run: $(TARGET_GPU)
	./$(TARGET_GPU) --mode gpu --params $(PARAMS) --steps 20 \
		--poisson split --pressure-boundary split \
		--poisson-dual-residual --poisson-deterministic-reductions \
		--poisson-check-interval 100 --poisson-tolerance 1e-3 \
		--poisson-iteration-limit 0 --no-roofline

clean:
	rm -rf $(TARGET_GPU) *.o __pycache__

help:
	@echo "Targets:"
	@echo "  make gpu   Build the production GPU runner"
	@echo "  make run   Run the strict 20-step split baseline"
	@echo "  make clean Remove local build outputs"
