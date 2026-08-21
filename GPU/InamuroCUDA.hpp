#pragma once
#include "Inamuro.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <cassert>

/**
 * InamuroCUDA
 *  - 组合(Has-a)关系：持有对 CPU 端 Inamuro 的引用，但不拥有其生命周期
 *  - 负责 GPU 端的数据/计算与 CPU/GPU 数据交换
 */
class InamuroCUDA
{
public:
    struct PoissonDiagnostics {
        int iterations = 0;
        double relative_residual = std::numeric_limits<double>::infinity();
        bool converged = false;
        bool finite = false;
        std::vector<double> residual_trace;
    };

    // 从已经初始化好的 CPU Inamuro 构造（推荐做法）
    explicit InamuroCUDA(const Inamuro& cpuSolver);

    // 设备裸指针具有唯一所有权；禁止复制和默认移动，避免双重释放。
    InamuroCUDA(const InamuroCUDA&) = delete;
    InamuroCUDA& operator=(const InamuroCUDA&) = delete;
    InamuroCUDA(InamuroCUDA&&) = delete;
    InamuroCUDA& operator=(InamuroCUDA&&) = delete;

    ~InamuroCUDA();

    // 执行一个完整CUDA时间步。
    void performTimeStepGPU();

    // 下载完整状态（ff/gg/hh及所有宏观量），用于输出和CPU/GPU资格验证。
    void downloadFieldsToCPU(Inamuro& cpuSolver) const;
    const PoissonDiagnostics& getLastPoissonDiagnostics() const noexcept { return last_poisson; }
    void printPerformanceMetrics() const;
    void resetPerformanceMetrics();
#ifdef LBM_TESTING
    bool testFiniteGateRejectsNaN();
#endif

private:
    // ------------ 网格与布局 ------------
    const Inamuro& cpu;       // 仅引用，不拥有
    int lx = 0, ly = 0, lz = 0;
    int lz_total = 0;         // 通常 = lz + 2（z方向ghost，两端各1层）
    int N_cells  = 0;         // 物理域 cell 数 = lx*ly*lz
    int N_macro  = 0;         // 宏观场存储数 = lx*ly*lz_total
    static constexpr int Q = 15; // D3Q15
    
    // ------------ GPU参数结构体 ------------
    struct GPUParams {
        double rho_L, rho_G;     // 液相/气相密度
        double mu_L, mu_G;       // 液相/气相粘度
        double tauf, taug;       // 相场/动量松弛时间
        double k_f, k_g;         // 相场/动量表面张力系数
        double T, a, b;          // EOS状态方程参数
        double fei_L, fei_G;     // 液相/气相相场值
    } params;
    
    bool is_first_step = true;   // 标记是否是第一个时间步（与CPU一致）
    static constexpr int kPoissonMaxIterations = 1000;
    static constexpr int kPoissonCheckInterval = 100;
    static constexpr double kPoissonTolerance = 1.0e-3;
    PoissonDiagnostics last_poisson;
    
    // ------------ 性能测量 ------------
    struct PerformanceMetrics {
        double total_collision_time = 0.0;   // 碰撞kernel总时间(ms)
        double total_stream_time = 0.0;      // 迁移kernel总时间(ms)
        double total_macro_time = 0.0;       // 宏观量kernel总时间(ms)
        double total_poisson_time = 0.0;     // 压力泊松总时间(ms)
        double total_step_wall_time = 0.0;   // 完整GPU求解步墙钟(ms)
        long long total_poisson_iterations = 0;
        int time_step_count = 0;             // 时间步计数
    } perf;
    
    bool enable_timing = true;      // 是否启用性能测量
    bool enable_debug = false;      // 是否启用DEBUG输出

    // ------------ 设备端字段 ------------
    struct GPUMemory
    {
        // 分布函数 [Q * N_cells]
        double* d_ff = nullptr;
        double* d_gg = nullptr;
        double* d_hh = nullptr;
        double* d_ff_tmp = nullptr;
        double* d_gg_tmp = nullptr;
        double* d_hh_tmp = nullptr;

        // 宏观量 [N_macro]（带 z 方向 ghost）
        double* d_rho = nullptr;
        double* d_fei = nullptr;
        double* d_u   = nullptr;
        double* d_v   = nullptr;
        double* d_w   = nullptr;
        double* d_p   = nullptr;
        double* d_p_prev = nullptr;
        double* d_pressure_error = nullptr;
        int* d_finite_flag = nullptr;

        // 梯度/拉普拉斯（仅物理域，后续逐步填充）
        double* d_fei_x = nullptr; double* d_fei_y = nullptr; double* d_fei_z = nullptr;
        double* d_rho_x = nullptr; double* d_rho_y = nullptr; double* d_rho_z = nullptr;
        double* d_u_x   = nullptr; double* d_u_y   = nullptr; double* d_u_z   = nullptr;
        double* d_v_x   = nullptr; double* d_v_y   = nullptr; double* d_v_z   = nullptr;
        double* d_w_x   = nullptr; double* d_w_y   = nullptr; double* d_w_z   = nullptr;

        double* d_fei_lap = nullptr;
        double* d_u_lap   = nullptr; double* d_v_lap = nullptr; double* d_w_lap = nullptr;
    } gpu;

    // ------------ 内部帮助函数 ------------
    void allocateDeviceMemory();
    void freeDeviceMemory();

    // 从 CPU 端“扁平化”并上传到 GPU（需要 Inamuro 声明 friend class InamuroCUDA）
    void initFromCPU();

    // 下载回 CPU（仅宏观量；若需要也可扩展下载分布函数做调试）
    void downloadMacroToCPU(Inamuro& cpuSolver) const;
    void downloadDistributionsToCPU(Inamuro& cpuSolver) const;
    double computePressureResidual();
    bool validateFiniteState();

    // ------------ 单步子过程（与 CPU performTimeStep 对齐） ------------
    // 组合函数（实际实现）
    void doCollisionAndGradients();    // collision + 梯度/拉普拉斯计算
    void doStreamFF();                 // ff迁移
    void doStreamGG();                 // gg迁移
    void doBoundaryFF();               // ff边界条件
    void doBoundaryGG();               // gg边界条件
    void doMacro();                    // 宏观量计算（对应CPU的getMacro）
    void doPressurePoisson();          // 压力泊松求解（对应CPU的solvePressurePoisson）
    void doCorrectUVWAndHH();          // 速度修正+hh更新（对应CPU的correct_uvw+update_hh）
    
    // ------------ 内联索引（cell-major / Q-minor） ------------
    static __host__ __device__ inline
    int idx3D(int x, int y, int z, int lx, int ly) {
        return (z * ly + y) * lx + x;
    }
};
