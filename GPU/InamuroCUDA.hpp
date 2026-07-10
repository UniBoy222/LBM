#pragma once
#include "Inamuro.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <set>
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
    // 从已经初始化好的 CPU Inamuro 构造（推荐做法）
    explicit InamuroCUDA(const Inamuro& cpuSolver);

    // 禁止拷贝，允许移动（如需）
    InamuroCUDA(const InamuroCUDA&) = delete;
    InamuroCUDA& operator=(const InamuroCUDA&) = delete;
    InamuroCUDA(InamuroCUDA&&) noexcept = default;
    InamuroCUDA& operator=(InamuroCUDA&&) noexcept = default;

    ~InamuroCUDA();

    // 单步时间推进（GPU 版本）——基线：宏观量 + 示例 streaming，其余留空壳供后续逐步替换
    void performTimeStepGPU();
    void performTimeStepGPUWithPoissonPair(Inamuro& cpuSolver, int completed_step,
                                           const std::string& pair_output_dir,
                                           const std::string& pair_format = "tecplot",
                                           bool write_pre_pair = true,
                                           bool write_post_pair = true);

    // 将 GPU 上的宏观量下载回 CPU Inamuro（用于输出/对比）
    void downloadFieldsToCPU(Inamuro& cpuSolver) const;

    // 性能与roofline证据输出
    void setUseFusedPoisson(bool enabled);
    void setUseOnePassPoisson(bool enabled);
    void setUseScalarPoisson(bool enabled);
    void setScalarPoissonSourceScale(double scale);
    void setUseSourceAwareHHInit(bool enabled);
    void setSourceAwareHHScale(double scale);
    void setPressureRelaxScale(double scale);
    void setPoissonFixedPointRelax(double omega);
    void setUsePoissonAndersonM1(bool enabled);
    void setPoissonAndersonBetaMax(double value);
    void setUsePoissonTwoGridCorrection(bool enabled);
    void setPoissonTwoGridStrength(double value);
    void setUseFusedBoundaryPressure(bool enabled);
    void setUsePoissonGraph(bool enabled);
    void setEnablePoissonDetailTiming(bool enabled);
    void setPoissonConvergence(int check_interval, double tolerance);
    void setPoissonDiagnosticsPath(const std::string& path);
    void setUsePoissonSpatialDiagnostics(bool enabled);
    void setPressureInitializer(const std::string& path, const std::string& mode);
    void setPressureInitializerMaxIterations(int max_iterations);
    void setPressureInitializerCheckInterval(int check_interval);
    void setPressureInitializerWaitDir(const std::string& dir, int timeout_ms, int max_step);
    void setPoissonStateExport(const std::string& dir, const std::set<int>& steps, const std::string& phase);
    void printPerformanceMetrics() const;
    void printRooflineSummary() const;
    void resetPerformanceMetrics();

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
    
    // ------------ 性能测量 ------------
    struct PerformanceMetrics {
        double total_collision_time = 0.0;   // 碰撞kernel总时间(ms)
        double total_stream_time = 0.0;      // 迁移kernel总时间(ms)
        double total_macro_time = 0.0;       // 宏观量kernel总时间(ms)
        double total_poisson_time = 0.0;     // 压力泊松总时间(ms)
        double poisson_collision_time = 0.0;
        double poisson_stream_time = 0.0;
        double poisson_fused_time = 0.0;
        double poisson_onepass_time = 0.0;
        double poisson_scalar_time = 0.0;
        double poisson_init_time = 0.0;
        double poisson_boundary_time = 0.0;
        double poisson_pressure_time = 0.0;
        double poisson_boundary_pressure_time = 0.0;
        double poisson_residual_time = 0.0;
        double pressure_initializer_time = 0.0;
        int total_poisson_iterations = 0;    // 压力泊松总迭代次数
        int pressure_initializer_attempts = 0;
        int pressure_initializer_accepts = 0;
        int pressure_initializer_fallbacks = 0;
        int time_step_count = 0;             // 时间步计数
    } perf;
    
    bool enable_timing = true;      // 是否启用性能测量
    bool enable_debug = false;      // 是否启用DEBUG输出
    bool use_fused_poisson = false; // 是否启用压力碰撞+迁移融合算子
    bool use_onepass_poisson = false;
    bool use_scalar_poisson = false;
    bool use_source_aware_hh_init = false;
    bool use_fused_boundary_pressure = false;
    bool use_poisson_graph = false;
    bool use_poisson_anderson_m1 = false;
    bool use_poisson_spatial_diagnostics = false;
    bool use_poisson_two_grid_correction = false;
    bool enable_poisson_detail_timing = false;
    int poisson_check_interval = 100;
    double poisson_tolerance = 0.001;
    double scalar_poisson_source_scale = 2.0;
    double source_aware_hh_scale = 1.0;
    double pressure_relax_scale = 1.0;
    double poisson_fixed_point_relax = 1.0;
    double poisson_anderson_beta_max = 1.0;
    double poisson_two_grid_strength = 0.5;
    std::string poisson_diagnostics_path;
    std::string pressure_initializer_path;
    std::string pressure_initializer_mode = "absolute";
    std::string pressure_initializer_wait_dir;
    std::string poisson_state_export_dir;
    std::set<int> poisson_state_export_steps;
    std::string poisson_state_export_phase = "pre";
    int pressure_initializer_wait_timeout_ms = 0;
    int pressure_initializer_wait_max_step = 0;
    bool pressure_initializer_loaded = false;
    bool pressure_initializer_has_hh = false;
    int pressure_initializer_max_iterations = 0;
    int pressure_initializer_check_interval = 0;

    cudaGraph_t poisson_graph = nullptr;
    cudaGraphExec_t poisson_graph_exec = nullptr;
    int poisson_graph_check_interval = 0;

    void destroyPoissonGraph();
    void buildPoissonGraphSegment();
    void writePoissonDiagnostic(int step, int iteration, double pressure_l1_delta,
                                double pressure_l1_norm, double relative_error,
                                bool converged, double block_low_frequency_fraction,
                                int block_size, int block_count) const;
    void loadPressureInitializer();
    void loadPressureInitializerForStep(int completed_step);
    void applyPressureInitializer();
    void seedPressureResidualFromCurrentPressure();
    void performTimeStepGPUImpl(Inamuro* pair_solver, int completed_step,
                                const std::string* pair_output_dir,
                                const std::string& pair_format,
                                bool write_pre_pair,
                                bool write_post_pair);
    void writePoissonPairSnapshot(Inamuro& cpuSolver, int completed_step,
                                  const std::string& pair_output_dir,
                                  const std::string& phase) const;
    void writePoissonFeatureSnapshot(int completed_step,
                                     const std::string& pair_output_dir,
                                     const std::string& phase) const;
    void writePoissonStateSnapshot(int completed_step,
                                   const std::string& pair_output_dir,
                                   const std::string& phase) const;

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
        double* d_p_tmp = nullptr;

        // 梯度/拉普拉斯（仅物理域，后续逐步填充）
        double* d_fei_x = nullptr; double* d_fei_y = nullptr; double* d_fei_z = nullptr;
        double* d_rho_x = nullptr; double* d_rho_y = nullptr; double* d_rho_z = nullptr;
        double* d_u_x   = nullptr; double* d_u_y   = nullptr; double* d_u_z   = nullptr;
        double* d_v_x   = nullptr; double* d_v_y   = nullptr; double* d_v_z   = nullptr;
        double* d_w_x   = nullptr; double* d_w_y   = nullptr; double* d_w_z   = nullptr;

        double* d_fei_lap = nullptr;
        double* d_u_lap   = nullptr; double* d_v_lap = nullptr; double* d_w_lap = nullptr;

        double* d_p_prev = nullptr;
        double* d_pressure_error = nullptr;
        double* d_pressure_init = nullptr;
        double* d_hh_init = nullptr;
        double* d_p_backup = nullptr;
        double* d_hh_backup = nullptr;
        double* d_p_prev_backup = nullptr;
        double* d_anderson_prev_residual = nullptr;
        double* d_anderson_prev_image = nullptr;
        double* d_anderson_stats = nullptr;
        double* d_pressure_block_sums = nullptr;
        double* d_pressure_block_error = nullptr;
    } gpu;

    // ------------ 内部帮助函数 ------------
    void allocateDeviceMemory();
    void freeDeviceMemory();

    // 从 CPU 端“扁平化”并上传到 GPU（需要 Inamuro 声明 friend class InamuroCUDA）
    void initFromCPU();

    // 下载回 CPU（仅宏观量；若需要也可扩展下载分布函数做调试）
    void downloadMacroToCPU(Inamuro& cpuSolver) const;

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
    int idx3D(int x, int y, int z, int lx, int ly, int lz_tot) {
        // 注意：用于宏观量时 z 取 [0 .. lz_tot-1]，用于分布函数/梯度时 z 取物理域 [0 .. lz-1]
        return (z * ly + y) * lx + x;
    }

    static __host__ __device__ inline
    int idx4D(int q, int x, int y, int z, int lx, int ly, int lz) {
        // z 为物理域索引
        const int cell = (z * ly + y) * lx + x;
        return cell * Q + q;  // 使用Q而非Q_
    }
};
