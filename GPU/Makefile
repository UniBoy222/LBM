# Production GPU solver Makefile.

NVCC ?= nvcc
CXX ?= g++
CUDA_ARCH ?= sm_120

NVCCFLAGS ?= -std=c++17 -O3 -arch=$(CUDA_ARCH)
INCLUDES := -I. -I../CPU

GPU_SRC := InamuroCUDA.cu
CPP_SRC := Inamuro.cpp common.cpp

TARGET_GPU := lbm_gpu
TARGET_TBOOK := tbook_stage_audit
TARGET_STATE_AUDIT := fixed_point_state_audit
TARGET_REAL_STATE := real_state_cross_audit
TARGET_ORACLE := oracle_pressure_routes
TARGET_TAIL_HNEQ_ORACLE := tail_hneq_warmstart_oracle
PARAMS := params.in

.PHONY: all gpu tbook-audit state-audit real-state-audit oracle tail-hneq-oracle run clean help

all: gpu

gpu: $(TARGET_GPU)

tbook-audit: $(TARGET_TBOOK)

state-audit: $(TARGET_STATE_AUDIT)

real-state-audit: $(TARGET_REAL_STATE)

oracle: $(TARGET_ORACLE)

tail-hneq-oracle: $(TARGET_TAIL_HNEQ_ORACLE)

$(TARGET_GPU): $(GPU_SRC) $(CPP_SRC) main.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) main.cpp -o $@

$(TARGET_TBOOK): $(GPU_SRC) $(CPP_SRC) tbook_stage_audit.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) tbook_stage_audit.cpp -o $@

$(TARGET_STATE_AUDIT): $(GPU_SRC) $(CPP_SRC) fixed_point_state_audit.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) fixed_point_state_audit.cpp -o $@

$(TARGET_REAL_STATE): real_state_cross_audit.cpp
	$(CXX) -std=c++17 -O3 -I. real_state_cross_audit.cpp -o $@

$(TARGET_ORACLE): $(GPU_SRC) $(CPP_SRC) oracle_pressure_routes.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) oracle_pressure_routes.cpp -o $@

$(TARGET_TAIL_HNEQ_ORACLE): $(GPU_SRC) $(CPP_SRC) tail_hneq_warmstart_oracle.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(GPU_SRC) $(CPP_SRC) tail_hneq_warmstart_oracle.cpp -o $@

run: $(TARGET_GPU)
	./$(TARGET_GPU) --mode gpu --params $(PARAMS) --steps 20 \
		--poisson split --pressure-boundary split \
		--poisson-dual-residual --poisson-deterministic-reductions \
		--poisson-check-interval 100 --poisson-tolerance 1e-3 \
		--poisson-iteration-limit 0 --no-roofline

clean:
	rm -rf $(TARGET_GPU) $(TARGET_TBOOK) $(TARGET_STATE_AUDIT) \
		$(TARGET_REAL_STATE) $(TARGET_ORACLE) $(TARGET_TAIL_HNEQ_ORACLE) *.o __pycache__

help:
	@echo "Targets:"
	@echo "  make gpu              Build the production GPU runner"
	@echo "  make tbook-audit      Build staged operator audit"
	@echo "  make state-audit      Build fixed-point state audit"
	@echo "  make real-state-audit Build CPU real-state replay audit"
	@echo "  make oracle           Build archived Oracle runner"
	@echo "  make tail-hneq-oracle Build strict p+hneq tail warm-start Oracle runner"
	@echo "  make run              Run the strict 20-step split baseline"
	@echo "  make clean            Remove local build outputs"
