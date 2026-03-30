# GPU版本Makefile - Inamuro LBM CUDA实现
# ========================================================
# 支持RTX 4060/4070 Ti (Ada Lovelace架构, sm_89)
# ========================================================

# 编译器
NVCC := nvcc

# CUDA架构 (RTX 4060/4070 Ti)
CUDA_ARCH := sm_89

# 编译选项
NVCCFLAGS := -std=c++17 -O3 -arch=$(CUDA_ARCH)
INCLUDES := -I. -I../CPU
LDFLAGS := 

# 源文件
GPU_SRC := InamuroCUDA.cu
CPP_SRC := Inamuro.cpp InamuroSolver.cpp common.cpp

# 目标程序
TARGET_GPU := lbm_gpu
TARGET_COMPARE := lbm_compare

# ========================================================
# 主要目标
# ========================================================

.PHONY: all gpu compare clean help test

all: gpu compare

# GPU版本（独立运行）
gpu: $(TARGET_GPU)

# 对比测试（GPU vs CPU）
compare: $(TARGET_COMPARE)

# ========================================================
# 编译规则
# ========================================================

# GPU独立版本
$(TARGET_GPU): $(GPU_SRC) $(CPP_SRC) main.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
		$(GPU_SRC) $(CPP_SRC) main.cpp \
		-o $@ $(LDFLAGS)
	@echo ""
	@echo "✅ GPU version compiled successfully!"
	@echo "   Executable: ./$(TARGET_GPU)"
	@echo "   GPU Architecture: $(CUDA_ARCH) (RTX 4060/4070 Ti)"
	@echo ""

# GPU vs CPU对比测试
$(TARGET_COMPARE): $(GPU_SRC) $(CPP_SRC) compare_test.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) \
		$(GPU_SRC) $(CPP_SRC) compare_test.cpp \
		-o $@ $(LDFLAGS)
	@echo ""
	@echo "✅ Comparison test compiled successfully!"
	@echo "   Executable: ./$(TARGET_COMPARE)"
	@echo ""

# ========================================================
# 测试目标
# ========================================================

# 快速测试（10步）
test: $(TARGET_COMPARE)
	@echo "=========================================="
	@echo "Running quick test (10 steps)..."
	@echo "=========================================="
	./$(TARGET_COMPARE)

# 运行GPU版本
run: $(TARGET_GPU)
	@echo "=========================================="
	@echo "Running GPU version..."
	@echo "=========================================="
	./$(TARGET_GPU)

# 性能测试（100步）
benchmark: $(TARGET_COMPARE)
	@echo "=========================================="
	@echo "Running performance benchmark (100 steps)..."
	@echo "=========================================="
	./$(TARGET_COMPARE) --benchmark

# ========================================================
# 清理
# ========================================================

clean:
	rm -f $(TARGET_GPU) $(TARGET_COMPARE) *.o
	@echo "✅ Cleaned build files"

# ========================================================
# 调试和分析
# ========================================================

# Debug版本（带调试信息）
debug: NVCCFLAGS := -std=c++17 -g -G -arch=$(CUDA_ARCH)
debug: $(TARGET_GPU)
	@echo "Debug version built with -g -G flags"

# 使用Nsight Compute分析
profile: $(TARGET_GPU)
	ncu --set full -o gpu_profile ./$(TARGET_GPU)
	@echo "Profile saved to gpu_profile.ncu-rep"

# 使用Nsight Systems分析时间线
timeline: $(TARGET_GPU)
	nsys profile -o gpu_timeline ./$(TARGET_GPU)
	@echo "Timeline saved to gpu_timeline.nsys-rep"

# 检查GPU信息
gpu-info:
	@echo "=========================================="
	@echo "GPU Information:"
	@echo "=========================================="
	nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
	@echo ""
	@echo "Current CUDA_ARCH: $(CUDA_ARCH)"
	@echo ""

# ========================================== ============
# 帮助信息
# ========================================================

help:
	@echo "GPU版本Makefile - Inamuro LBM CUDA"
	@echo "=========================================="
	@echo ""
	@echo "编译目标:"
	@echo "  make              - 编译GPU版本和对比测试"
	@echo "  make gpu          - 仅编译GPU版本"
	@echo "  make compare      - 仅编译对比测试"
	@echo "  make debug        - 编译Debug版本"
	@echo ""
	@echo "运行目标:"
	@echo "  make run          - 运行GPU版本"
	@echo "  make test         - 运行快速测试(10步)"
	@echo "  make benchmark    - 运行性能测试(100步)"
	@echo ""
	@echo "分析工具:"
	@echo "  make profile      - Nsight Compute性能分析"
	@echo "  make timeline     - Nsight Systems时间线分析"
	@echo "  make gpu-info     - 显示GPU信息"
	@echo ""
	@echo "其他:"
	@echo "  make clean        - 清理编译文件"
	@echo "  make help         - 显示此帮助"
	@echo ""
	@echo "当前配置:"
	@echo "  GPU架构: $(CUDA_ARCH) (RTX 4060/4070 Ti)"
	@echo "  编译器: $(NVCC)"
	@echo ""
