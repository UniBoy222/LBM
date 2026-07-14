# Minimal GPU foundation Makefile for PINN/Poisson restart work.

NVCC ?= nvcc
CUDA_ARCH ?= sm_89

NVCCFLAGS ?= -std=c++17 -O3 -arch=$(CUDA_ARCH)
INCLUDES := -I. -I../CPU

GPU_SRC := InamuroCUDA.cu
CPP_SRC := Inamuro.cpp InamuroSolver.cpp common.cpp

TARGET_GPU := lbm_gpu
TARGET_COMPARE := lbm_compare
TARGET_TBOOK := tbook_stage_audit
TARGET_ORACLE := oracle_pressure_routes
PARAMS_SMALL := params_small.in

.PHONY: all gpu compare tbook-audit oracle test test-onepass run clean help

all: gpu compare

gpu: $(TARGET_GPU)

compare: $(TARGET_COMPARE)

tbook-audit: $(TARGET_TBOOK)

oracle: $(TARGET_ORACLE)

$(TARGET_GPU): $(GPU_SRC) $(CPP_SRC) main.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) main.cpp -o $@

$(TARGET_COMPARE): $(GPU_SRC) $(CPP_SRC) compare_test.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) compare_test.cpp -o $@

$(TARGET_TBOOK): $(GPU_SRC) $(CPP_SRC) tbook_stage_audit.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) tbook_stage_audit.cpp -o $@

$(TARGET_ORACLE): $(GPU_SRC) $(CPP_SRC) oracle_pressure_routes.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) oracle_pressure_routes.cpp -o $@

test: $(TARGET_COMPARE)
	./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 1 --poisson split

test-onepass: $(TARGET_COMPARE)
	./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 1 --poisson onepass --pressure-boundary fused

run: $(TARGET_GPU)
	./$(TARGET_GPU) --mode gpu --params $(PARAMS_SMALL) --steps 20 --poisson onepass --pressure-boundary fused --no-roofline

clean:
	rm -rf $(TARGET_GPU) $(TARGET_COMPARE) $(TARGET_TBOOK) $(TARGET_ORACLE) *.o __pycache__

help:
	@echo "Targets:"
	@echo "  make gpu          Build GPU runner"
	@echo "  make compare      Build CPU/GPU comparison gate"
	@echo "  make tbook-audit  Build staged Eq.(6.42)-(6.44) audit"
	@echo "  make oracle       Build strict 5/3/1 Oracle runner"
	@echo "  make test         Run 1-step split CPU/GPU gate"
	@echo "  make test-onepass Run 1-step onepass CPU/GPU gate"
	@echo "  make run          Run small GPU smoke case"
	@echo "  make clean        Remove local build outputs"
