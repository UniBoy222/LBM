#pragma once
#include "Inamuro.hpp"
#include <cuda_runtime.h>
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

    // 将 GPU 上的宏观量下载回 CPU Inamuro（用于输出/对比）
    void downloadFieldsToCPU(Inamuro& cpuSolver) const;

private:
    // ------------ 网格与布局 ------------
    const Inamuro& cpu_;       // 仅引用，不拥有
    int lx_ = 0, ly_ = 0, lz_ = 0;
    int lz_total_ = 0;         // 通常 = lz_ + 2（z方向ghost，两端各1层）
    int N_cells_  = 0;         // 物理域 cell 数 = lx_*ly_*lz_
    int N_macro_  = 0;         // 宏观场存储数 = lx_*ly_*lz_total_
    static constexpr int Q_ = 15; // D3Q15

    // ------------ 设备端字段 ------------
    struct GPUMemory
    {
        // 分布函数 [Q * N_cells_]
        double* d_ff = nullptr;
        double* d_gg = nullptr;
        double* d_hh = nullptr;

        // 宏观量 [N_macro_]（带 z 方向 ghost）
        double* d_rho = nullptr;
        double* d_fei = nullptr;
        double* d_u   = nullptr;
        double* d_v   = nullptr;
        double* d_w   = nullptr;
        double* d_p   = nullptr;

        // 梯度/拉普拉斯（仅物理域，后续逐步填充）
        double* d_fei_x = nullptr; double* d_fei_y = nullptr; double* d_fei_z = nullptr;
        double* d_rho_x = nullptr; double* d_rho_y = nullptr; double* d_rho_z = nullptr;
        double* d_u_x   = nullptr; double* d_u_y   = nullptr; double* d_u_z   = nullptr;
        double* d_v_x   = nullptr; double* d_v_y   = nullptr; double* d_v_z   = nullptr;
        double* d_w_x   = nullptr; double* d_w_y   = nullptr; double* d_w_z   = nullptr;

        double* d_fei_lap = nullptr;
        double* d_u_lap   = nullptr; double* d_v_lap = nullptr; double* d_w_lap = nullptr;
    } gpu_{};

    // ------------ 内部帮助函数 ------------
    void allocateDeviceMemory();
    void freeDeviceMemory();

    // 从 CPU 端“扁平化”并上传到 GPU（需要 Inamuro 声明 friend class InamuroCUDA）
    void initFromCPU();

    // 下载回 CPU（仅宏观量；若需要也可扩展下载分布函数做调试）
    void downloadMacroToCPU(Inamuro& cpuSolver) const;

    // ------------ 单步子过程（与 CPU performTimeStep 对齐） ------------
    void doCollisionAndGradients(); // TODO: 先留空壳，后续替换为你的 collision + (grad/lap) tiled 版本
    void doStreamFF();              // 示例：pull streaming（可换为你的版本）
    void doStreamGG();              // 同上
    void doBoundaryFF();            // TODO: 先留空壳
    void doBoundaryGG();            // TODO: 先留空壳
    void doMacro();                 // 已实现：用 ff/gg 计算 rho/fei/u/v/w（基线）
    void doPressurePoisson();       // TODO: 先留空壳（后续做全 GPU + 并行残差）
    void doCorrectUVWAndHH();       // TODO: 先留空壳

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
        return cell * Q_ + q;
    }
};