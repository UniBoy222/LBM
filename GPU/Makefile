# Minimal GPU foundation Makefile for PINN/Poisson restart work.

NVCC ?= nvcc
CUDA_ARCH ?= sm_89
PYTHON ?= ../PINN_Poisson/.venv/bin/python

NVCCFLAGS ?= -std=c++17 -O3 -arch=$(CUDA_ARCH)
INCLUDES := -I. -I../CPU

GPU_SRC := InamuroCUDA.cu
CPP_SRC := Inamuro.cpp InamuroSolver.cpp common.cpp

TARGET_GPU := lbm_gpu
TARGET_COMPARE := lbm_compare
TARGET_BOOK_PRESSURE := book_pressure_step_test
TARGET_BOOK_REPLAY := book_pressure_replay
PARAMS_SMALL := params_small.in

.PHONY: all gpu compare test test-onepass test-book-pressure run clean help

all: gpu compare

gpu: $(TARGET_GPU)

compare: $(TARGET_COMPARE)

$(TARGET_GPU): $(GPU_SRC) $(CPP_SRC) main.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) main.cpp -o $@

$(TARGET_COMPARE): $(GPU_SRC) $(CPP_SRC) compare_test.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) compare_test.cpp -o $@

$(TARGET_BOOK_PRESSURE): $(GPU_SRC) $(CPP_SRC) book_pressure_step_test.cpp BookPressureTest.hpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) book_pressure_step_test.cpp -o $@

$(TARGET_BOOK_REPLAY): $(GPU_SRC) $(CPP_SRC) book_pressure_replay.cpp BookPressureTest.hpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) book_pressure_replay.cpp -o $@

test: $(TARGET_COMPARE)
	./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 1 --poisson split

test-onepass: $(TARGET_COMPARE)
	./$(TARGET_COMPARE) --params $(PARAMS_SMALL) --steps 1 --poisson onepass --pressure-boundary fused

test-book-pressure: $(TARGET_BOOK_PRESSURE)
	$(PYTHON) ../PINN_Poisson/scripts/test_book_pressure_operator.py --cpp-test ./$(TARGET_BOOK_PRESSURE)

run: $(TARGET_GPU)
	./$(TARGET_GPU) --mode gpu --params $(PARAMS_SMALL) --steps 20 --poisson onepass --pressure-boundary fused --no-roofline

clean:
	rm -rf $(TARGET_GPU) $(TARGET_COMPARE) $(TARGET_BOOK_PRESSURE) $(TARGET_BOOK_REPLAY) *.o __pycache__

help:
	@echo "Targets:"
	@echo "  make gpu          Build GPU runner"
	@echo "  make compare      Build CPU/GPU comparison gate"
	@echo "  make test         Run 1-step split CPU/GPU gate"
	@echo "  make test-onepass Run 1-step onepass CPU/GPU gate"
	@echo "  make test-book-pressure Run book Eq. (6.42)-(6.44) stage gate"
	@echo "  make run          Run small GPU smoke case"
	@echo "  make clean        Remove local build outputs"
