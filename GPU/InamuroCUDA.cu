#include "InamuroCUDA.hpp"
#include "BookPressureTest.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

// 简单错误检查宏
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        throw std::runtime_error("CUDA failure"); \
    } \
} while(0)
#endif

// -------------------- D3Q15 方向与权重（常量内存） --------------------
// 与 CPU 版本 D3Q15 保持完全一致
__constant__ int    c_ex[15];
__constant__ int    c_ey[15];
__constant__ int    c_ez[15];
__constant__ double c_uc[15];   // double版本方向向量（用于梯度计算）
__constant__ double c_vc[15];
__constant__ double c_wc[15];
__constant__ double c_wE[15];   // 用于 ff 的权重（等温 LBM 常用）
__constant__ double c_wF[15];   // 用于 gg 的权重（如与 c_wE 不同可分开）
__constant__ double c_Ei[15];   // LBM权重
__constant__ double c_Hi[15];   // 压力分布函数权重
__constant__ double c_Fi[15];   // 扩散函数权重

static void upload_lattice_constants()
{
    // Host 侧定义 - 与 CPU 版本 D3Q15 保持一致
    static const int ex[15] = {
        0,  1, 0, 0, -1, 0, 0,  1, -1, 1, 1, -1, 1, -1, -1
    };
    static const int ey[15] = {
        0,  0, 1, 0, 0, -1, 0,  1, 1, -1, 1, -1, -1, 1, -1
    };
    static const int ez[15] = {
        0,  0, 0, 1, 0, 0, -1,  1, 1, 1, -1, -1, -1, -1, 1
    };
    
    // double 版本方向向量（与CPU版本的uc/vc/wc一致）
    static const double uc[15] = {
        0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0,
        1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0
    };
    static const double vc[15] = {
        0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0,
        1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0
    };
    static const double wc[15] = {
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0,
        1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0
    };
    
    // LBM 权重 Ei：w0=2/9, axes=1/9, diagonals=1/72
    static const double Ei[15] = {
        2.0/9.0,
        1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0
    };
    
    // 压力分布函数权重 Hi
    static const double Hi[15] = {
        1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    
    // 扩散函数权重 Fi
    static const double Fi[15] = {
        -7.0/3.0,
        1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0,
        1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0, 1.0/24.0
    };
    
    // 先默认 gg 使用相同权重
    static const double wE[15] = {
        2.0/9.0,
        1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0
    };
    static const double wF[15] = {
        2.0/9.0,
        1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0, 1.0/72.0
    };

    // 上传到常量内存
    CUDA_CHECK(cudaMemcpyToSymbol(c_ex, ex, sizeof(ex)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ey, ey, sizeof(ey)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ez, ez, sizeof(ez)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_uc, uc, sizeof(uc)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_vc, vc, sizeof(vc)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wc, wc, sizeof(wc)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Ei, Ei, sizeof(Ei)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Hi, Hi, sizeof(Hi)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Fi, Fi, sizeof(Fi)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wE, wE, sizeof(wE)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wF, wF, sizeof(wF)));
}

// -------------------- 宏观量 kernel（对应CPU的getMacro）--------------------
__global__ void macroKernel(const double* __restrict__ ff,
                            const double* __restrict__ gg,
                            double* __restrict__ rho,
                            double* __restrict__ fei,
                            double* __restrict__ u,
                            double* __restrict__ v,
                            double* __restrict__ w,
                            int lx, int ly, int lz, int lztot,
                            double fei_L, double fei_G,
                            double rho_L, double rho_G)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int cell = (z * ly + y) * lx + x;
    
    // 1. 计算相场 φ = Σ ff[i]  (i=0到14)
    double fei_local = 0.0;
    for (int i = 0; i < 15; ++i) {
        fei_local += ff[cell * 15 + i];
    }
    
    // 2. 根据φ计算密度（三段式+正弦过渡）
    double density;
    const double fei_avg = (fei_L + fei_G) / 2.0;
    
    if (fei_local <= fei_G) {
        density = rho_G;
    } else if (fei_local >= fei_L) {
        density = rho_L;
    } else {
        double arg = (fei_local - fei_avg) / (fei_L - fei_G) * M_PI;
        density = (rho_L - rho_G) / 2.0 * (sin(arg) + 1.0) + rho_G;
    }
    
    // 3. 速度从gg计算（从i=1开始，跳过静止方向0）
    double u_local = 0.0;
    double v_local = 0.0;
    double w_local = 0.0;
    
    for (int i = 1; i < 15; ++i) {
        double g_val = gg[cell * 15 + i];
        u_local += c_uc[i] * g_val;
        v_local += c_vc[i] * g_val;
        w_local += c_wc[i] * g_val;
    }
    
    // 4. 写回到带ghost的宏观场（z+1）
    const int out = (((z + 1) * ly + y) * lx + x);
    fei[out] = fei_local;
    rho[out] = density;
    u[out] = u_local;
    v[out] = v_local;
    w[out] = w_local;
}

// -------------------- 梯度计算 kernel --------------------
/**
 * 计算一阶梯度：∇φ = (∂φ/∂x, ∂φ/∂y, ∂φ/∂z)
 * 使用 D3Q15 的 14 个非静止方向加权求和
 * 
 * @param var: 输入场（带ghost层，尺寸为lx*ly*lztot）
 * @param var_x, var_y, var_z: 输出梯度（无ghost层，尺寸为lx*ly*lz）
 * @param has_ghost: 输入是否有ghost层（宏观量有，梯度场无）
 */
__global__ void gradientKernel(
    const double* __restrict__ var,
    double* __restrict__ var_x,
    double* __restrict__ var_y,
    double* __restrict__ var_z,
    int lx, int ly, int lz, int lztot,
    bool has_ghost)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    double grad_x = 0.0, grad_y = 0.0, grad_z = 0.0;

    // 循环14个非静止方向 (k = 1..14)
    for (int k = 1; k < 15; ++k) {
        // 计算邻居索引
        int xp = x + c_ex[k];
        int yp = y + c_ey[k];
        int zp = z + c_ez[k];

        // X方向：对称边界 (slip wall)
        if (xp >= lx) xp = lx - 2;
        if (xp < 0)   xp = 1;

        // Y方向：周期边界
        if (yp >= ly) yp = 0;
        if (yp < 0)   yp = ly - 1;

        // Z方向：周期边界
        if (zp >= lz) zp = 0;
        if (zp < 0)   zp = lz - 1;

        // 获取邻居值
        int zp_idx = has_ghost ? (zp + 1) : zp;  // 有ghost层时z索引+1
        int idx = (zp_idx * ly + yp) * lx + xp;
        double var_neighbor = var[idx];

        // 累加梯度（使用uc/vc/wc作为方向权重）
        grad_x += var_neighbor * c_uc[k];
        grad_y += var_neighbor * c_vc[k];
        grad_z += var_neighbor * c_wc[k];
    }

    // 归一化（除以10.0，与CPU版本一致）
    grad_x /= 10.0;
    grad_y /= 10.0;
    grad_z /= 10.0;

    // 写回结果（梯度场无ghost层）
    const int out_idx = (z * ly + y) * lx + x;
    var_x[out_idx] = grad_x;
    var_y[out_idx] = grad_y;
    var_z[out_idx] = grad_z;
}

// -------------------- 拉普拉斯计算 kernel --------------------
/**
 * 计算二阶拉普拉斯：∇²φ
 * 公式：lap = (Σ邻居值 - 14×中心值) / 5.0
 * 
 * @param var: 输入场（带ghost层，尺寸为lx*ly*lztot）
 * @param var_lap: 输出拉普拉斯（无ghost层，尺寸为lx*ly*lz）
 * @param has_ghost: 输入是否有ghost层
 */
__global__ void laplacianKernel(
    const double* __restrict__ var,
    double* __restrict__ var_lap,
    int lx, int ly, int lz, int lztot,
    bool has_ghost)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    double neighbor_sum = 0.0;

    // 循环14个非静止方向 (k = 1..14)
    for (int k = 1; k < 15; ++k) {
        // 计算邻居索引
        int xp = x + c_ex[k];
        int yp = y + c_ey[k];
        int zp = z + c_ez[k];

        // X方向：对称边界 (slip wall)
        if (xp >= lx) xp = lx - 2;
        if (xp < 0)   xp = 1;

        // Y方向：周期边界
        if (yp >= ly) yp = 0;
        if (yp < 0)   yp = ly - 1;

        // Z方向：周期边界
        if (zp >= lz) zp = 0;
        if (zp < 0)   zp = lz - 1;

        // 获取邻居值并累加
        int zp_idx = has_ghost ? (zp + 1) : zp;
        int idx = (zp_idx * ly + yp) * lx + xp;
        neighbor_sum += var[idx];
    }

    // 获取中心值
    int z_idx = has_ghost ? (z + 1) : z;
    int center_idx = (z_idx * ly + y) * lx + x;
    double center_val = var[center_idx];

    // 计算拉普拉斯：(sum - 14*center) / 5.0
    double laplacian = (neighbor_sum - 14.0 * center_val) / 5.0;

    // 写回结果（拉普拉斯场无ghost层）
    const int out_idx = (z * ly + y) * lx + x;
    var_lap[out_idx] = laplacian;
}

// -------------------- EOS 状态方程 (device函数) --------------------
/**
 * van der Waals 状态方程
 * p = φT/(1-bφ) - aφ²
 */
__device__ inline void EOS_device(double fei, double& p, double T, double a, double b)
{
    p = fei * T / (1.0 - b * fei) - a * fei * fei;
}

// -------------------- 碰撞 kernel (核心) --------------------
/**
 * Inamuro碰撞kernel - 计算ff和gg的平衡态并执行BGK碰撞
 * 对应CPU版本的collision()函数
 * 
 * 输入：所有宏观量、梯度场、拉普拉斯场
 * 输出：更新后的ff和gg分布函数
 */
__global__ void collisionKernel(
    double* __restrict__ ff,
    double* __restrict__ gg,
    const double* __restrict__ rho,
    const double* __restrict__ fei,
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ w,
    // 梯度场（无ghost）
    const double* __restrict__ fei_x, const double* __restrict__ fei_y, const double* __restrict__ fei_z,
    const double* __restrict__ rho_x, const double* __restrict__ rho_y, const double* __restrict__ rho_z,
    const double* __restrict__ u_x,   const double* __restrict__ u_y,   const double* __restrict__ u_z,
    const double* __restrict__ v_x,   const double* __restrict__ v_y,   const double* __restrict__ v_z,
    const double* __restrict__ w_x,   const double* __restrict__ w_y,   const double* __restrict__ w_z,
    // 拉普拉斯场（无ghost）
    const double* __restrict__ fei_lap,
    const double* __restrict__ u_lap, const double* __restrict__ v_lap, const double* __restrict__ w_lap,
    // 参数
    double rho_L, double rho_G, double mu_L, double mu_G,
    double tauf, double taug, double k_f, double k_g,
    double T, double a, double b,
    int lx, int ly, int lz, int lztot,
    bool is_first_step)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    // 索引计算
    const int cell = (z * ly + y) * lx + x;           // 物理域索引（用于分布函数和梯度场）
    const int z_macro = z + 1;                         // 宏观量z索引（有ghost层）
    const int macro_idx = (z_macro * ly + y) * lx + x; // 宏观量索引

    // 预加载当前格点的宏观量到寄存器
    const double loc_rho = rho[macro_idx];
    const double loc_fei = fei[macro_idx];
    const double loc_u = u[macro_idx];
    const double loc_v = v[macro_idx];
    const double loc_w = w[macro_idx];

    // 预加载梯度和拉普拉斯
    const double loc_fei_x = fei_x[cell], loc_fei_y = fei_y[cell], loc_fei_z = fei_z[cell];
    const double loc_rho_x = rho_x[cell], loc_rho_y = rho_y[cell], loc_rho_z = rho_z[cell];
    const double loc_u_x = u_x[cell], loc_u_y = u_y[cell], loc_u_z = u_z[cell];
    const double loc_v_x = v_x[cell], loc_v_y = v_y[cell], loc_v_z = v_z[cell];
    const double loc_w_x = w_x[cell], loc_w_y = w_y[cell], loc_w_z = w_z[cell];
    const double loc_fei_lap = fei_lap[cell];
    const double loc_u_lap = u_lap[cell], loc_v_lap = v_lap[cell], loc_w_lap = w_lap[cell];

    // 计算粘度（线性插值）
    const double tmp1 = (mu_L - mu_G) / (rho_L - rho_G);
    const double mu = (loc_rho - rho_G) * tmp1 + mu_G;

    // 梯度模长平方
    const double sum_fei = loc_fei_x * loc_fei_x + loc_fei_y * loc_fei_y + loc_fei_z * loc_fei_z;
    const double sum_rho = loc_rho_x * loc_rho_x + loc_rho_y * loc_rho_y + loc_rho_z * loc_rho_z;
    const double usq = loc_u * loc_u + loc_v * loc_v + loc_w * loc_w;

    // 调用EOS计算压力
    double p0;
    EOS_device(loc_fei, p0, T, a, b);

    // 方向循环：计算平衡态并执行碰撞
    for (int i = 0; i < 15; ++i) {
        // 速度与离散速度的点积
        const double un = c_uc[i] * loc_u + c_vc[i] * loc_v + c_wc[i] * loc_w;

        // 相场梯度与离散速度的点积
        const double fei_ei = c_uc[i] * loc_fei_x + c_vc[i] * loc_fei_y + c_wc[i] * loc_fei_z;

        // 密度梯度与离散速度的点积
        const double rho_ei = c_uc[i] * loc_rho_x + c_vc[i] * loc_rho_y + c_wc[i] * loc_rho_z;

        // 离散速度模长平方
        const double ci_sq = c_uc[i] * c_uc[i] + c_vc[i] * c_vc[i] + c_wc[i] * c_wc[i];

        // 高阶项
        const double Gfei = 4.5 * (fei_ei * fei_ei) - 1.5 * sum_fei * ci_sq;
        const double Grho = 4.5 * (rho_ei * rho_ei) - 1.5 * sum_rho * ci_sq;

        // 相场分布函数平衡态 (Allen-Cahn方程)
        const double fequ = c_Hi[i] * loc_fei +
                            c_Fi[i] * (p0 - k_f * loc_fei * loc_fei_lap - k_f * sum_fei / 6.0) +
                            3.0 * c_Ei[i] * loc_fei * un +
                            c_Ei[i] * k_f * Gfei;

        // 动量分布函数平衡态 (Navier-Stokes方程)
        const double velPart = 1.0 + 3.0 * un - 1.5 * usq + 4.5 * un * un +
                               1.5 * (taug - 0.5) * 2.0 * 
                               (c_uc[i] * c_uc[i] * loc_u_x + c_uc[i] * c_vc[i] * loc_u_y + c_uc[i] * c_wc[i] * loc_u_z +
                                c_vc[i] * c_uc[i] * loc_v_x + c_vc[i] * c_vc[i] * loc_v_y + c_vc[i] * c_wc[i] * loc_v_z +
                                c_wc[i] * c_uc[i] * loc_w_x + c_wc[i] * c_vc[i] * loc_w_y + c_wc[i] * c_wc[i] * loc_w_z);

        const double gequ = c_Ei[i] * velPart +
                            c_Ei[i] * k_g / loc_rho * Grho -
                            2.0 / 3.0 * c_Fi[i] * k_g / loc_rho * sum_rho;

        // 分布函数索引
        const int id4 = cell * 15 + i;

        if (is_first_step) {
            // 第一步：初始化为平衡态
            ff[id4] = fequ;
            gg[id4] = gequ;
        } else {
            // BGK碰撞 + 粘性修正
            ff[id4] = ff[id4] - (ff[id4] - fequ) / tauf;
            gg[id4] = gg[id4] - (gg[id4] - gequ) / taug +
                      3.0 * c_Ei[i] / loc_rho * mu *
                      (c_uc[i] * loc_u_lap + c_vc[i] * loc_v_lap + c_wc[i] * loc_w_lap);
        }
    }
}

// -------------------- 边界条件 kernel (滑移反弹) --------------------
/**
 * X方向边界的滑移反弹 (slip bounce-back)
 * 对应CPU版本的slipBounceBack()函数
 * 
 * 边界规则：
 * 左边界 (x=0):  正方向 = 负方向 (1->4, 7->8等)
 * 右边界 (x=lx-1): 负方向 = 正方向 (4->1, 8->7等)
 */
__global__ void slipBounceBackKernel(
    double* __restrict__ dist,
    int lx, int ly, int lz)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= ly || z >= lz) return;

    // 反弹规则对：(正方向, 负方向)
    // 方向1<->4, 7<->8, 9<->14, 10<->13, 12<->11
    const int bounce_pairs[5][2] = {
        {1, 4}, {7, 8}, {9, 14}, {10, 13}, {12, 11}
    };

    // 左边界 (x=0)
    const int x_left = 0;
    const int cell_left = (z * ly + y) * lx + x_left;
    for (int p = 0; p < 5; ++p) {
        int pos_dir = bounce_pairs[p][0];
        int neg_dir = bounce_pairs[p][1];
        dist[cell_left * 15 + pos_dir] = dist[cell_left * 15 + neg_dir];
    }

    // 右边界 (x=lx-1)
    const int x_right = lx - 1;
    const int cell_right = (z * ly + y) * lx + x_right;
    for (int p = 0; p < 5; ++p) {
        int pos_dir = bounce_pairs[p][0];
        int neg_dir = bounce_pairs[p][1];
        dist[cell_right * 15 + neg_dir] = dist[cell_right * 15 + pos_dir];
    }
}

// -------------------- 压力泊松求解器 kernels --------------------
/**
 * 压力碰撞kernel
 * 对应CPU版本的collision_p()函数
 * 
 * 公式：τ_h = 1/ρ + 0.5
 *       h_new = h - (h - hequ)/τ_h - (Ei/3)·∇·u
 */
__global__ void collisionPressureKernel(
    double* __restrict__ hh,
    const double* __restrict__ p,
    const double* __restrict__ rho,
    const double* __restrict__ u_x,
    const double* __restrict__ v_y,
    const double* __restrict__ w_z,
    int lx, int ly, int lz, int lztot,
    double pressure_relax_scale)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int cell = (z * ly + y) * lx + x;              // 物理域索引
    const int z_macro = z + 1;
    const int macro_idx = (z_macro * ly + y) * lx + x;   // 宏观量索引

    // 预加载宏观量
    const double loc_rho = rho[macro_idx];
    const double loc_p = p[macro_idx];
    const double div_u = u_x[cell] + v_y[cell] + w_z[cell];  // ∇·u

    // 计算松弛时间 τ_h
    const double tauh = 1.0 / loc_rho + 0.5;

    // 15个方向的碰撞
    for (int k = 0; k < 15; ++k) {
        const double hequ = c_Ei[k] * loc_p;
        const int id4 = cell * 15 + k;
        
        hh[id4] = hh[id4] - pressure_relax_scale * (hh[id4] - hequ) / tauh - (c_Ei[k] / 3.0) * div_u;
    }
}

__global__ void collisionPressureStreamKernel(
    const double* __restrict__ hh,
    double* __restrict__ hh_next,
    const double* __restrict__ p,
    const double* __restrict__ rho,
    const double* __restrict__ u_x,
    const double* __restrict__ v_y,
    const double* __restrict__ w_z,
    int lx, int ly, int lz, int lztot,
    double pressure_relax_scale)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int dst_cell = (z * ly + y) * lx + x;

    for (int q = 0; q < 15; ++q) {
        int xs = x - c_ex[q];
        int ys = y - c_ey[q];
        int zs = z - c_ez[q];

        if (xs < 0) xs += lx; else if (xs >= lx) xs -= lx;
        if (ys < 0) ys += ly; else if (ys >= ly) ys -= ly;
        if (zs < 0) zs += lz; else if (zs >= lz) zs -= lz;

        const int src_cell = (zs * ly + ys) * lx + xs;
        const int src_macro = ((zs + 1) * ly + ys) * lx + xs;
        const double loc_rho = rho[src_macro];
        const double loc_p = p[src_macro];
        const double div_u = u_x[src_cell] + v_y[src_cell] + w_z[src_cell];
        const double tauh = 1.0 / loc_rho + 0.5;
        const double hequ = c_Ei[q] * loc_p;
        const double h_old = hh[src_cell * 15 + q];

        hh_next[dst_cell * 15 + q] =
            h_old - pressure_relax_scale * (h_old - hequ) / tauh - (c_Ei[q] / 3.0) * div_u;
    }
}

/**
 * 压力计算kernel
 * 对应CPU版本的getp()函数
 * 
 * 公式：p = Σ h_i
 */
__global__ void computePressureKernel(
    const double* __restrict__ hh,
    double* __restrict__ p,
    int lx, int ly, int lz, int lztot)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int cell = (z * ly + y) * lx + x;
    const int z_macro = z + 1;
    const int macro_idx = (z_macro * ly + y) * lx + x;

    double p_sum = 0.0;
    for (int i = 0; i < 15; ++i) {
        p_sum += hh[cell * 15 + i];
    }
    p[macro_idx] = p_sum;
}

__global__ void boundaryAndComputePressureKernel(
    double* __restrict__ hh,
    double* __restrict__ p,
    int lx, int ly, int lz, int lztot)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int cell = (z * ly + y) * lx + x;
    double h[15];
    for (int q = 0; q < 15; ++q) {
        h[q] = hh[cell * 15 + q];
    }

    if (x == 0) {
        h[1] = h[4];
        h[7] = h[8];
        h[9] = h[14];
        h[10] = h[13];
        h[12] = h[11];
        hh[cell * 15 + 1] = h[1];
        hh[cell * 15 + 7] = h[7];
        hh[cell * 15 + 9] = h[9];
        hh[cell * 15 + 10] = h[10];
        hh[cell * 15 + 12] = h[12];
    } else if (x == lx - 1) {
        h[4] = h[1];
        h[8] = h[7];
        h[14] = h[9];
        h[13] = h[10];
        h[11] = h[12];
        hh[cell * 15 + 4] = h[4];
        hh[cell * 15 + 8] = h[8];
        hh[cell * 15 + 14] = h[14];
        hh[cell * 15 + 13] = h[13];
        hh[cell * 15 + 11] = h[11];
    }

    double p_sum = 0.0;
    for (int q = 0; q < 15; ++q) {
        p_sum += h[q];
    }
    const int macro_idx = ((z + 1) * ly + y) * lx + x;
    p[macro_idx] = p_sum;
}

__global__ void collisionStreamBoundaryPressureKernel(
    const double* __restrict__ hh,
    double* __restrict__ hh_next,
    const double* __restrict__ p,
    double* __restrict__ p_next,
    const double* __restrict__ rho,
    const double* __restrict__ u_x,
    const double* __restrict__ v_y,
    const double* __restrict__ w_z,
    int lx, int ly, int lz, int lztot,
    double pressure_relax_scale,
    double fixed_point_relax)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    double h[15];
    for (int q = 0; q < 15; ++q) {
        int xs = x - c_ex[q];
        int ys = y - c_ey[q];
        int zs = z - c_ez[q];

        if (xs < 0) xs += lx; else if (xs >= lx) xs -= lx;
        if (ys < 0) ys += ly; else if (ys >= ly) ys -= ly;
        if (zs < 0) zs += lz; else if (zs >= lz) zs -= lz;

        const int src_cell = (zs * ly + ys) * lx + xs;
        const int src_macro = ((zs + 1) * ly + ys) * lx + xs;
        const double loc_rho = rho[src_macro];
        const double loc_p = p[src_macro];
        const double div_u = u_x[src_cell] + v_y[src_cell] + w_z[src_cell];
        const double tauh = 1.0 / loc_rho + 0.5;
        const double hequ = c_Ei[q] * loc_p;
        const double h_old = hh[src_cell * 15 + q];

        h[q] = h_old - pressure_relax_scale * (h_old - hequ) / tauh - (c_Ei[q] / 3.0) * div_u;
    }

    if (x == 0) {
        h[1] = h[4];
        h[7] = h[8];
        h[9] = h[14];
        h[10] = h[13];
        h[12] = h[11];
    } else if (x == lx - 1) {
        h[4] = h[1];
        h[8] = h[7];
        h[14] = h[9];
        h[13] = h[10];
        h[11] = h[12];
    }

    const int cell = (z * ly + y) * lx + x;
    double p_sum = 0.0;
    for (int q = 0; q < 15; ++q) {
        const double h_old = hh[cell * 15 + q];
        h[q] = h_old + fixed_point_relax * (h[q] - h_old);
        hh_next[cell * 15 + q] = h[q];
        p_sum += h[q];
    }
    const int macro_idx = ((z + 1) * ly + y) * lx + x;
    p_next[macro_idx] = p_sum;
}

__device__ inline int pressure_bounce_direction(int q, int x, int lx)
{
    if (x == 0) {
        if (q == 1) return 4;
        if (q == 7) return 8;
        if (q == 9) return 14;
        if (q == 10) return 13;
        if (q == 12) return 11;
    } else if (x == lx - 1) {
        if (q == 4) return 1;
        if (q == 8) return 7;
        if (q == 14) return 9;
        if (q == 13) return 10;
        if (q == 11) return 12;
    }
    return q;
}

__global__ void scalarPressureJacobiKernel(
    const double* __restrict__ p,
    double* __restrict__ p_next,
    const double* __restrict__ rho,
    const double* __restrict__ u_x,
    const double* __restrict__ v_y,
    const double* __restrict__ w_z,
    int lx, int ly, int lz, int lztot,
    double source_scale)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    double neighbor_sum = 0.0;
    for (int q = 1; q < 15; ++q) {
        const int qb = pressure_bounce_direction(q, x, lx);
        int xs = x - c_ex[qb];
        int ys = y - c_ey[qb];
        int zs = z - c_ez[qb];

        if (xs < 0) xs += lx; else if (xs >= lx) xs -= lx;
        if (ys < 0) ys += ly; else if (ys >= ly) ys -= ly;
        if (zs < 0) zs += lz; else if (zs >= lz) zs -= lz;

        const int src_cell = (zs * ly + ys) * lx + xs;
        const int src_macro = ((zs + 1) * ly + ys) * lx + xs;
        (void)src_cell;
        neighbor_sum += p[src_macro];
    }

    const int cell = (z * ly + y) * lx + x;
    const int macro_idx = ((z + 1) * ly + y) * lx + x;
    const double div_u = u_x[cell] + v_y[cell] + w_z[cell];
    const double rhs = source_scale * rho[macro_idx] * div_u;
    p_next[macro_idx] = (neighbor_sum - 5.0 * rhs) / 14.0;
}

__global__ void sourceAwareHHInitKernel(
    double* __restrict__ hh,
    const double* __restrict__ p,
    const double* __restrict__ rho,
    const double* __restrict__ u_x,
    const double* __restrict__ v_y,
    const double* __restrict__ w_z,
    int lx, int ly, int lz, int lztot,
    double source_scale)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int cell = (z * ly + y) * lx + x;
    const int macro_idx = ((z + 1) * ly + y) * lx + x;
    const double tauh = 1.0 / rho[macro_idx] + 0.5;
    const double div_u = u_x[cell] + v_y[cell] + w_z[cell];
    const double loc_p = p[macro_idx];

    for (int q = 0; q < 15; ++q) {
        const double hequ = c_Ei[q] * loc_p;
        const double source = (c_Ei[q] / 3.0) * div_u;
        hh[cell * 15 + q] = hequ - source_scale * tauh * source;
    }
}

__global__ void applyPressureInitializerKernel(
    double* __restrict__ p,
    const double* __restrict__ pressure_init,
    int lx, int ly, int lz, int lztot,
    int is_delta)
{
    const int global = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = lx * ly * lz;
    if (global >= n) return;

    const int x = global % lx;
    const int y = (global / lx) % ly;
    const int z = global / (lx * ly);
    const int macro_idx = ((z + 1) * ly + y) * lx + x;
    const double value = pressure_init[global];
    p[macro_idx] = is_delta ? (p[macro_idx] + value) : value;
}

__global__ void seedPressurePrevKernel(
    const double* __restrict__ p,
    double* __restrict__ p_prev,
    int lx, int ly, int lz, int lztot)
{
    const int global = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = lx * ly * lz;
    if (global >= n) return;

    const int x = global % lx;
    const int y = (global / lx) % ly;
    const int z = global / (lx * ly);
    const int macro_idx = ((z + 1) * ly + y) * lx + x;
    p_prev[global] = p[macro_idx];
}

__global__ void packPoissonFeatureSnapshotKernel(
    const double* __restrict__ u,
    const double* __restrict__ v,
    const double* __restrict__ w,
    const double* __restrict__ rho,
    const double* __restrict__ fei,
    const double* __restrict__ press,
    const double* __restrict__ upx,
    const double* __restrict__ vpy,
    const double* __restrict__ wpz,
    float* __restrict__ out,
    int lx, int ly, int lz, int lztot)
{
    const int n = lx * ly * lz;
    const int global = blockIdx.x * blockDim.x + threadIdx.x;
    if (global >= 7 * n) return;

    const int field = global / n;
    const int cell = global - field * n;
    const int x = cell % lx;
    const int y = (cell / lx) % ly;
    const int z = cell / (lx * ly);
    const int macro_idx = ((z + 1) * ly + y) * lx + x;

    double value = 0.0;
    if (field == 0) {
        value = u[macro_idx];
    } else if (field == 1) {
        value = v[macro_idx];
    } else if (field == 2) {
        value = w[macro_idx];
    } else if (field == 3) {
        value = rho[macro_idx];
    } else if (field == 4) {
        value = fei[macro_idx];
    } else if (field == 5) {
        value = press[macro_idx];
    } else {
        value = upx[cell] + vpy[cell] + wpz[cell];
    }
    out[global] = static_cast<float>(value);
}

__global__ void andersonStoreHistoryKernel(
    const double* __restrict__ h_old,
    const double* __restrict__ h_image,
    double* __restrict__ prev_residual,
    double* __restrict__ prev_image,
    int n_values)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_values) return;
    prev_residual[idx] = h_image[idx] - h_old[idx];
    prev_image[idx] = h_image[idx];
}

__global__ void andersonM1DotKernel(
    const double* __restrict__ h_old,
    const double* __restrict__ h_image,
    const double* __restrict__ prev_residual,
    double* __restrict__ stats,
    int n_values)
{
    extern __shared__ double scratch[];
    double* s_num = scratch;
    double* s_den = scratch + blockDim.x;
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double num = 0.0;
    double den = 0.0;
    if (idx < n_values) {
        const double residual = h_image[idx] - h_old[idx];
        const double delta = residual - prev_residual[idx];
        num = prev_residual[idx] * delta;
        den = delta * delta;
    }

    s_num[tid] = num;
    s_den[tid] = den;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_num[tid] += s_num[tid + stride];
            s_den[tid] += s_den[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&stats[0], s_num[0]);
        atomicAdd(&stats[1], s_den[0]);
    }
}

__global__ void andersonM1ApplyKernel(
    const double* __restrict__ h_old,
    double* __restrict__ h_image,
    const double* __restrict__ prev_image,
    double* __restrict__ prev_residual,
    double* __restrict__ prev_image_out,
    double* __restrict__ p_image,
    int lx, int ly, int lz, int lztot,
    double alpha)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int cell = (z * ly + y) * lx + x;
    double p_sum = 0.0;
    for (int q = 0; q < 15; ++q) {
        const int idx = cell * 15 + q;
        const double current_image = h_image[idx];
        const double accelerated = (1.0 - alpha) * prev_image[idx] + alpha * current_image;
        prev_residual[idx] = current_image - h_old[idx];
        prev_image_out[idx] = current_image;
        h_image[idx] = accelerated;
        p_sum += accelerated;
    }

    const int macro_idx = ((z + 1) * ly + y) * lx + x;
    p_image[macro_idx] = p_sum;
}

__global__ void pressureErrorKernel(
    const double* __restrict__ p,
    double* __restrict__ p_prev,
    double* __restrict__ err,
    int lx, int ly, int lz, int lztot)
{
    extern __shared__ double scratch[];
    double* s_err1 = scratch;
    double* s_err2 = scratch + blockDim.x;

    const int tid = threadIdx.x;
    const int global = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = lx * ly * lz;

    double local_err1 = 0.0;
    double local_err2 = 0.0;

    if (global < n) {
        const int x = global % lx;
        const int y = (global / lx) % ly;
        const int z = global / (lx * ly);
        const int macro_idx = ((z + 1) * ly + y) * lx + x;
        const double current = p[macro_idx];
        const double prev = p_prev[global];
        local_err1 = fabs(current - prev);
        local_err2 = fabs(current);
        p_prev[global] = current;
    }

    s_err1[tid] = local_err1;
    s_err2[tid] = local_err2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_err1[tid] += s_err1[tid + stride];
            s_err2[tid] += s_err2[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&err[0], s_err1[0]);
        atomicAdd(&err[1], s_err2[0]);
    }
}

__global__ void pressureErrorSpatialKernel(
    const double* __restrict__ p,
    double* __restrict__ p_prev,
    double* __restrict__ err,
    double* __restrict__ block_sums,
    int lx, int ly, int lz, int lztot,
    int block_size,
    int bx_count,
    int by_count)
{
    extern __shared__ double scratch[];
    double* s_err1 = scratch;
    double* s_err2 = scratch + blockDim.x;

    const int tid = threadIdx.x;
    const int global = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = lx * ly * lz;

    double local_err1 = 0.0;
    double local_err2 = 0.0;

    if (global < n) {
        const int x = global % lx;
        const int y = (global / lx) % ly;
        const int z = global / (lx * ly);
        const int macro_idx = ((z + 1) * ly + y) * lx + x;
        const double current = p[macro_idx];
        const double prev = p_prev[global];
        const double delta = current - prev;
        local_err1 = fabs(delta);
        local_err2 = fabs(current);

        const int bx = x / block_size;
        const int by = y / block_size;
        const int bz = z / block_size;
        const int block_id = (bz * by_count + by) * bx_count + bx;
        atomicAdd(&block_sums[block_id], delta);

        p_prev[global] = current;
    }

    s_err1[tid] = local_err1;
    s_err2[tid] = local_err2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_err1[tid] += s_err1[tid + stride];
            s_err2[tid] += s_err2[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&err[0], s_err1[0]);
        atomicAdd(&err[1], s_err2[0]);
    }
}

__global__ void blockAbsSumKernel(
    const double* __restrict__ block_sums,
    double* __restrict__ out,
    int n_blocks)
{
    extern __shared__ double scratch[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if (idx < n_blocks) {
        value = fabs(block_sums[idx]);
    }
    scratch[tid] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, scratch[0]);
    }
}

__global__ void coarsePressureCorrectionKernel(
    double* __restrict__ hh,
    double* __restrict__ p,
    double* __restrict__ p_prev,
    const double* __restrict__ block_sums,
    int lx, int ly, int lz, int lztot,
    int block_size,
    int bx_count,
    int by_count,
    double strength)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int bx = x / block_size;
    const int by = y / block_size;
    const int bz = z / block_size;
    const int block_id = (bz * by_count + by) * bx_count + bx;

    const int x0 = bx * block_size;
    const int y0 = by * block_size;
    const int z0 = bz * block_size;
    const int x1 = min(x0 + block_size, lx);
    const int y1 = min(y0 + block_size, ly);
    const int z1 = min(z0 + block_size, lz);
    const int block_cells = max(1, (x1 - x0) * (y1 - y0) * (z1 - z0));

    const double correction = strength * block_sums[block_id] / static_cast<double>(block_cells);
    const int cell = (z * ly + y) * lx + x;
    const int macro_idx = ((z + 1) * ly + y) * lx + x;
    const double corrected_p = p[macro_idx] + correction;
    p[macro_idx] = corrected_p;
    p_prev[cell] = corrected_p;

    for (int q = 0; q < 15; ++q) {
        hh[cell * 15 + q] += c_Ei[q] * correction;
    }
}

// -------------------- 速度修正和hh更新 kernels --------------------
/**
 * 速度修正kernel - 对应CPU版本的correct_uvw()
 * 公式：u -= ∇p / (2ρ)
 */
__global__ void correctVelocityKernel(
    double* __restrict__ u,
    double* __restrict__ v,
    double* __restrict__ w,
    const double* __restrict__ p,
    const double* __restrict__ rho,
    int lx, int ly, int lz, int lztot)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int z_macro = z + 1;
    const int macro_idx = (z_macro * ly + y) * lx + x;

    // X方向：对称边界
    int x_e = (x == lx - 1) ? lx - 2 : x + 1;
    int x_w = (x == 0) ? 1 : x - 1;

    // Y方向：周期边界
    int y_n = (y == ly - 1) ? 0 : y + 1;
    int y_s = (y == 0) ? ly - 1 : y - 1;

    // Z方向：周期边界
    int z_n = (z == lz - 1) ? 0 : z + 1;
    int z_s = (z == 0) ? lz - 1 : z - 1;

    // 邻居索引（宏观量带ghost层）
    int idx_xe = (z_macro * ly + y) * lx + x_e;
    int idx_xw = (z_macro * ly + y) * lx + x_w;
    int idx_yn = (z_macro * ly + y_n) * lx + x;
    int idx_ys = (z_macro * ly + y_s) * lx + x;
    int idx_zn = ((z_n + 1) * ly + y) * lx + x;
    int idx_zs = ((z_s + 1) * ly + y) * lx + x;

    // 压力梯度并修正速度
    double inv_2rho = 1.0 / (2.0 * rho[macro_idx]);
    u[macro_idx] -= (p[idx_xe] - p[idx_xw]) * inv_2rho;
    v[macro_idx] -= (p[idx_yn] - p[idx_ys]) * inv_2rho;
    w[macro_idx] -= (p[idx_zn] - p[idx_zs]) * inv_2rho;
}

/**
 * 更新hh kernel - 对应CPU版本的update_hh()
 * 公式：h_i = p * Ei
 */
__global__ void updateHHKernel(
    double* __restrict__ hh,
    const double* __restrict__ p,
    int lx, int ly, int lz, int lztot)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int cell = (z * ly + y) * lx + x;
    const int z_macro = z + 1;
    const int macro_idx = (z_macro * ly + y) * lx + x;
    
    const double loc_p = p[macro_idx];
    
    for (int i = 0; i < 15; ++i) {
        hh[cell * 15 + i] = loc_p * c_Ei[i];
    }
}

// -------------------- 简单 pull streaming（示例，可按你版本替换） --------------------
__global__ void streamKernel(const double* __restrict__ f_post,
                             double* __restrict__ f_new,
                             int lx, int ly, int lz)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int cell = (z * ly + y) * lx + x;

    // pull：从邻居拉数据（周期边界示意；你的边界条件可在 boundaryKernel 中单独做）
    for (int q = 0; q < 15; ++q) {
        int xn = x - c_ex[q], yn = y - c_ey[q], zn = z - c_ez[q];

        // 周期处理（示例；如果你的 z 有 ghost，请保持 streaming 仅在物理域）
        if (xn < 0)      xn += lx; else if (xn >= lx) xn -= lx;
        if (yn < 0)      yn += ly; else if (yn >= ly) yn -= ly;
        if (zn < 0)      zn += lz; else if (zn >= lz) zn -= lz;

        const int srcCell = (zn * ly + yn) * lx + xn;
        const int srcId4  = srcCell * 15 + q;
        const int dstId4  = cell    * 15 + q;

        f_new[dstId4] = f_post[srcId4];
    }
}

// ==================== InamuroCUDA 成员实现 ====================

InamuroCUDA::InamuroCUDA(const Inamuro& cpuSolver)
    : cpu(cpuSolver)
{
    // 从 CPU 端获取网格尺寸（与CPU命名一致）
    int nx=0, ny=0, nz=0;
    cpu.getGridSize(nx, ny, nz);
    lx = nx; ly = ny; lz = nz;

    // 你的 CPU 代码里，宏观量通常在 z 上带 2 层 ghost
    lz_total = lz + 2;

    N_cells = lx * ly * lz;
    N_macro = lx * ly * lz_total;

    // 从CPU获取参数（需要Inamuro声明friend或提供访问接口）
    params.rho_L = cpu.params.rho_L;
    params.rho_G = cpu.params.rho_G;
    params.mu_L = cpu.params.mu_L;
    params.mu_G = cpu.params.mu_G;
    params.tauf = cpu.params.tauf;
    params.taug = cpu.params.taug;
    params.k_f = cpu.params.k_f;
    params.k_g = cpu.params.k_g;
    params.T = cpu.params.T;
    params.a = cpu.params.a;
    params.b = cpu.params.b;
    params.fei_L = cpu.params.fei_L;
    params.fei_G = cpu.params.fei_G;

    upload_lattice_constants();
    allocateDeviceMemory();
    initFromCPU();
}

InamuroCUDA::~InamuroCUDA()
{
    destroyPoissonGraph();
    freeDeviceMemory();
}

void InamuroCUDA::destroyPoissonGraph()
{
    if (poisson_graph_exec) {
        cudaGraphExecDestroy(poisson_graph_exec);
        poisson_graph_exec = nullptr;
    }
    if (poisson_graph) {
        cudaGraphDestroy(poisson_graph);
        poisson_graph = nullptr;
    }
    poisson_graph_check_interval = 0;
}

void InamuroCUDA::buildPoissonGraphSegment()
{
    destroyPoissonGraph();

    if (poisson_check_interval <= 0 || (poisson_check_interval % 2) != 0) {
        throw std::runtime_error("Poisson graph requires a positive even check interval");
    }

    dim3 block(8, 8, 8);
    dim3 grid((lx+block.x-1)/block.x, (ly+block.y-1)/block.y, (lz+block.z-1)/block.z);
    dim3 onepass_block(16, 4, 4);
    dim3 onepass_grid((lx+onepass_block.x-1)/onepass_block.x,
                      (ly+onepass_block.y-1)/onepass_block.y,
                      (lz+onepass_block.z-1)/onepass_block.z);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    double* h_in = gpu.d_hh;
    double* h_out = gpu.d_hh_tmp;
    double* p_in = gpu.d_p;
    double* p_out = gpu.d_p_tmp;
    for (int iter = 0; iter < poisson_check_interval; ++iter) {
        if (use_onepass_poisson) {
            collisionStreamBoundaryPressureKernel<<<onepass_grid, onepass_block, 0, stream>>>(
                h_in, h_out, p_in, p_out, gpu.d_rho,
                gpu.d_u_x, gpu.d_v_y, gpu.d_w_z,
                lx, ly, lz, lz_total,
                pressure_relax_scale,
                poisson_fixed_point_relax
            );
            std::swap(p_in, p_out);
        } else {
            collisionPressureStreamKernel<<<grid, block, 0, stream>>>(
                h_in, h_out, gpu.d_p, gpu.d_rho,
                gpu.d_u_x, gpu.d_v_y, gpu.d_w_z,
                lx, ly, lz, lz_total,
                pressure_relax_scale
            );
            boundaryAndComputePressureKernel<<<grid, block, 0, stream>>>(
                h_out, gpu.d_p,
                lx, ly, lz, lz_total
            );
        }
        std::swap(h_in, h_out);
    }

    CUDA_CHECK(cudaStreamEndCapture(stream, &poisson_graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaGraphInstantiate(&poisson_graph_exec, poisson_graph, nullptr, nullptr, 0));
    poisson_graph_check_interval = poisson_check_interval;
}

void InamuroCUDA::allocateDeviceMemory()
{
    const size_t dist_bytes  = static_cast<size_t>(N_cells) * 15 * sizeof(double);
    const size_t macro_bytes = static_cast<size_t>(N_macro) * sizeof(double);
    const size_t cell_bytes  = static_cast<size_t>(N_cells) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&gpu.d_ff, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_gg, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_hh, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_ff_tmp, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_gg_tmp, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_hh_tmp, dist_bytes));

    CUDA_CHECK(cudaMalloc(&gpu.d_rho, macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_fei, macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_u,   macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_v,   macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_w,   macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_p,   macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_p_tmp, macro_bytes));

    // 梯度/拉普拉斯（预留，便于后续优化替换）
    CUDA_CHECK(cudaMalloc(&gpu.d_fei_x, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_fei_y, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_fei_z, cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu.d_rho_x, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_rho_y, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_rho_z, cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu.d_u_x, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_u_y, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_u_z, cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu.d_v_x, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_v_y, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_v_z, cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu.d_w_x, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_w_y, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_w_z, cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu.d_fei_lap, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_u_lap,   cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_v_lap,   cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_w_lap,   cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu.d_p_prev, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_pressure_error, 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_anderson_prev_residual, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_anderson_prev_image, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_anderson_stats, 2 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu.d_pressure_block_sums, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_pressure_block_error, sizeof(double)));
    CUDA_CHECK(cudaMemset(gpu.d_p_prev, 0, cell_bytes));
    CUDA_CHECK(cudaMemset(gpu.d_pressure_error, 0, 2 * sizeof(double)));
    CUDA_CHECK(cudaMemset(gpu.d_anderson_prev_residual, 0, dist_bytes));
    CUDA_CHECK(cudaMemset(gpu.d_anderson_prev_image, 0, dist_bytes));
    CUDA_CHECK(cudaMemset(gpu.d_anderson_stats, 0, 2 * sizeof(double)));
    CUDA_CHECK(cudaMemset(gpu.d_pressure_block_sums, 0, cell_bytes));
    CUDA_CHECK(cudaMemset(gpu.d_pressure_block_error, 0, sizeof(double)));
}

void InamuroCUDA::freeDeviceMemory()
{
    auto S = [](double*& p){ if(p){ cudaFree(p); p=nullptr; } };
    S(gpu.d_ff); S(gpu.d_gg); S(gpu.d_hh);
    S(gpu.d_ff_tmp); S(gpu.d_gg_tmp); S(gpu.d_hh_tmp);
    S(gpu.d_rho); S(gpu.d_fei); S(gpu.d_u); S(gpu.d_v); S(gpu.d_w); S(gpu.d_p); S(gpu.d_p_tmp);
    S(gpu.d_fei_x); S(gpu.d_fei_y); S(gpu.d_fei_z);
    S(gpu.d_rho_x); S(gpu.d_rho_y); S(gpu.d_rho_z);
    S(gpu.d_u_x);   S(gpu.d_u_y);   S(gpu.d_u_z);
    S(gpu.d_v_x);   S(gpu.d_v_y);   S(gpu.d_v_z);
    S(gpu.d_w_x);   S(gpu.d_w_y);   S(gpu.d_w_z);
    S(gpu.d_fei_lap); S(gpu.d_u_lap); S(gpu.d_v_lap); S(gpu.d_w_lap);
    S(gpu.d_p_prev); S(gpu.d_pressure_error); S(gpu.d_pressure_init); S(gpu.d_hh_init);
    S(gpu.d_p_backup); S(gpu.d_hh_backup); S(gpu.d_p_prev_backup);
    S(gpu.d_anderson_prev_residual); S(gpu.d_anderson_prev_image); S(gpu.d_anderson_stats);
    S(gpu.d_pressure_block_sums); S(gpu.d_pressure_block_error);
}

// 将 CPU 三/四维容器扁平化并上传（需要 Inamuro 声明 friend）
void InamuroCUDA::initFromCPU()
{
    // ------- 分布函数：Q * N_cells -------
    std::vector<double> h_ff(N_cells * 15);
    std::vector<double> h_gg(N_cells * 15);
    std::vector<double> h_hh(N_cells * 15);

    for (int z = 0; z < lz; ++z)
    for (int y = 0; y < ly; ++y)
    for (int x = 0; x < lx; ++x) {
        const int cell = (z * ly + y) * lx + x;
        for (int q = 0; q < 15; ++q) {
            const int id4 = cell * 15 + q;
            // NOTE: 依赖 friend 访问 Inamuro 的受保护成员
            h_ff[id4] = cpu.ff[q][x][y][z];
            h_gg[id4] = cpu.gg[q][x][y][z];
            h_hh[id4] = cpu.hh[q][x][y][z];
        }
    }

    CUDA_CHECK(cudaMemcpy(gpu.d_ff, h_ff.data(), h_ff.size()*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_gg, h_gg.data(), h_gg.size()*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_hh, h_hh.data(), h_hh.size()*sizeof(double), cudaMemcpyHostToDevice));

    // ------- 宏观量：N_macro（z 含 ghost，z ∈ [0..lz+1]） -------
    std::vector<double> h_rho(N_macro), h_fei(N_macro), h_u(N_macro), h_v(N_macro), h_w(N_macro), h_p(N_macro);

    for (int z = 0; z < lz_total; ++z)
    for (int y = 0; y < ly; ++y)
    for (int x = 0; x < lx; ++x) {
        const int id3 = idx3D(x, y, z, lx, ly, lz_total);
        h_rho[id3] = cpu.rho[x][y][z];
        h_fei[id3] = cpu.fei[x][y][z];
        h_u[id3]   = cpu.u[x][y][z];
        h_v[id3]   = cpu.v[x][y][z];
        h_w[id3]   = cpu.w[x][y][z];
        h_p[id3]   = cpu.p[x][y][z];
    }

    CUDA_CHECK(cudaMemcpy(gpu.d_rho, h_rho.data(), h_rho.size()*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_fei, h_fei.data(), h_fei.size()*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_u,   h_u.data(),   h_u.size()*sizeof(double),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_v,   h_v.data(),   h_v.size()*sizeof(double),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_w,   h_w.data(),   h_w.size()*sizeof(double),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_p,   h_p.data(),   h_p.size()*sizeof(double),   cudaMemcpyHostToDevice));
}

void InamuroCUDA::downloadMacroToCPU(Inamuro& cpuSolver) const
{
    std::vector<double> h_rho(N_macro), h_fei(N_macro), h_u(N_macro), h_v(N_macro), h_w(N_macro), h_p(N_macro);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), gpu.d_rho, N_macro*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fei.data(), gpu.d_fei, N_macro*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_u.data(),   gpu.d_u,   N_macro*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(),   gpu.d_v,   N_macro*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w.data(),   gpu.d_w,   N_macro*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_p.data(),   gpu.d_p,   N_macro*sizeof(double), cudaMemcpyDeviceToHost));

    for (int z = 0; z < lz_total; ++z)
    for (int y = 0; y < ly; ++y)
    for (int x = 0; x < lx; ++x) {
        const int id3 = idx3D(x, y, z, lx, ly, lz_total);
        cpuSolver.rho[x][y][z] = h_rho[id3];
        cpuSolver.fei[x][y][z] = h_fei[id3];
        cpuSolver.u[x][y][z]   = h_u[id3];
        cpuSolver.v[x][y][z]   = h_v[id3];
        cpuSolver.w[x][y][z]   = h_w[id3];
        cpuSolver.p[x][y][z]   = h_p[id3];
    }
}

void InamuroCUDA::downloadFieldsToCPU(Inamuro& cpuSolver) const
{
    downloadMacroToCPU(cpuSolver);
}

// ==================== 单步流程（基线） ====================

void InamuroCUDA::doCollisionAndGradients()
{
    dim3 block(8, 8, 8);
    dim3 grid((lx+block.x-1)/block.x, (ly+block.y-1)/block.y, (lz+block.z-1)/block.z);

    // 1. 计算梯度：∇φ, ∇ρ, ∇u, ∇v, ∇w
    gradientKernel<<<grid, block>>>(gpu.d_fei, gpu.d_fei_x, gpu.d_fei_y, gpu.d_fei_z, 
                                     lx, ly, lz, lz_total, true);
    gradientKernel<<<grid, block>>>(gpu.d_rho, gpu.d_rho_x, gpu.d_rho_y, gpu.d_rho_z, 
                                     lx, ly, lz, lz_total, true);
    gradientKernel<<<grid, block>>>(gpu.d_u, gpu.d_u_x, gpu.d_u_y, gpu.d_u_z, 
                                     lx, ly, lz, lz_total, true);
    gradientKernel<<<grid, block>>>(gpu.d_v, gpu.d_v_x, gpu.d_v_y, gpu.d_v_z, 
                                     lx, ly, lz, lz_total, true);
    gradientKernel<<<grid, block>>>(gpu.d_w, gpu.d_w_x, gpu.d_w_y, gpu.d_w_z, 
                                     lx, ly, lz, lz_total, true);
    CUDA_CHECK(cudaGetLastError());

    // 2. 计算拉普拉斯：∇²φ, ∇²u, ∇²v, ∇²w
    laplacianKernel<<<grid, block>>>(gpu.d_fei, gpu.d_fei_lap, lx, ly, lz, lz_total, true);
    laplacianKernel<<<grid, block>>>(gpu.d_u, gpu.d_u_lap, lx, ly, lz, lz_total, true);
    laplacianKernel<<<grid, block>>>(gpu.d_v, gpu.d_v_lap, lx, ly, lz, lz_total, true);
    laplacianKernel<<<grid, block>>>(gpu.d_w, gpu.d_w_lap, lx, ly, lz, lz_total, true);
    CUDA_CHECK(cudaGetLastError());

    // 3. 碰撞（计算平衡态并更新ff和gg）
    collisionKernel<<<grid, block>>>(
        gpu.d_ff, gpu.d_gg,
        gpu.d_rho, gpu.d_fei, gpu.d_u, gpu.d_v, gpu.d_w,
        gpu.d_fei_x, gpu.d_fei_y, gpu.d_fei_z,
        gpu.d_rho_x, gpu.d_rho_y, gpu.d_rho_z,
        gpu.d_u_x, gpu.d_u_y, gpu.d_u_z,
        gpu.d_v_x, gpu.d_v_y, gpu.d_v_z,
        gpu.d_w_x, gpu.d_w_y, gpu.d_w_z,
        gpu.d_fei_lap, gpu.d_u_lap, gpu.d_v_lap, gpu.d_w_lap,
        params.rho_L, params.rho_G, params.mu_L, params.mu_G,
        params.tauf, params.taug, params.k_f, params.k_g,
        params.T, params.a, params.b,
        lx, ly, lz, lz_total, is_first_step
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 第一步之后标志设为false
    is_first_step = false;
}

void InamuroCUDA::doStreamFF()
{
    dim3 block(8,8,8);
    dim3 grid((lx+block.x-1)/block.x, (ly+block.y-1)/block.y, (lz+block.z-1)/block.z);
    streamKernel<<<grid, block>>>(gpu.d_ff, gpu.d_ff_tmp, lx, ly, lz);
    CUDA_CHECK(cudaGetLastError());
    std::swap(gpu.d_ff, gpu.d_ff_tmp);
}

void InamuroCUDA::doStreamGG()
{
    dim3 block(8,8,8);
    dim3 grid((lx+block.x-1)/block.x, (ly+block.y-1)/block.y, (lz+block.z-1)/block.z);
    streamKernel<<<grid, block>>>(gpu.d_gg, gpu.d_gg_tmp, lx, ly, lz);
    CUDA_CHECK(cudaGetLastError());
    std::swap(gpu.d_gg, gpu.d_gg_tmp);
}

void InamuroCUDA::doBoundaryFF()
{
    dim3 block(16, 16);
    dim3 grid((ly+block.x-1)/block.x, (lz+block.y-1)/block.y);
    slipBounceBackKernel<<<grid, block>>>(gpu.d_ff, lx, ly, lz);
    CUDA_CHECK(cudaGetLastError());
}

void InamuroCUDA::doBoundaryGG()
{
    dim3 block(16, 16);
    dim3 grid((ly+block.x-1)/block.x, (lz+block.y-1)/block.y);
    slipBounceBackKernel<<<grid, block>>>(gpu.d_gg, lx, ly, lz);
    CUDA_CHECK(cudaGetLastError());
}

void InamuroCUDA::doMacro()
{
    dim3 block(8,8,8);
    dim3 grid((lx+block.x-1)/block.x, (ly+block.y-1)/block.y, (lz+block.z-1)/block.z);
    macroKernel<<<grid, block>>>(
        gpu.d_ff, gpu.d_gg,
        gpu.d_rho, gpu.d_fei,
        gpu.d_u, gpu.d_v, gpu.d_w,
        lx, ly, lz, lz_total,
        cpu.params.fei_L, cpu.params.fei_G,
        cpu.params.rho_L, cpu.params.rho_G
    );
    CUDA_CHECK(cudaGetLastError());
}

void InamuroCUDA::doPressurePoisson()
{
    dim3 block(8, 8, 8);
    dim3 grid((lx+block.x-1)/block.x, (ly+block.y-1)/block.y, (lz+block.z-1)/block.z);
    dim3 onepass_block(16, 4, 4);
    dim3 onepass_grid((lx+onepass_block.x-1)/onepass_block.x,
                      (ly+onepass_block.y-1)/onepass_block.y,
                      (lz+onepass_block.z-1)/onepass_block.z);
    dim3 block2d(16, 16);
    dim3 grid2d((ly+block2d.x-1)/block2d.x, (lz+block2d.y-1)/block2d.y);

    constexpr int max_iterations = 1000;
    const double tolerance = poisson_tolerance;

    cudaEvent_t phase_start{}, phase_stop{};
    const bool detail_timing = enable_timing && enable_poisson_detail_timing;
    const bool graph_eligible =
        use_poisson_graph &&
        !use_scalar_poisson &&
        !use_poisson_anderson_m1 &&
        !use_poisson_two_grid_correction &&
        (use_onepass_poisson || (use_fused_poisson && use_fused_boundary_pressure)) &&
        !detail_timing &&
        poisson_check_interval > 0 &&
        (poisson_check_interval % 2) == 0 &&
        (max_iterations % poisson_check_interval) == 0;

    if (detail_timing) {
        CUDA_CHECK(cudaEventCreate(&phase_start));
        CUDA_CHECK(cudaEventCreate(&phase_stop));
    }

    auto begin_phase = [&]() {
        if (detail_timing) {
            CUDA_CHECK(cudaEventRecord(phase_start));
        }
    };
    auto end_phase = [&](double& bucket) {
        if (detail_timing) {
            CUDA_CHECK(cudaEventRecord(phase_stop));
            CUDA_CHECK(cudaEventSynchronize(phase_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, phase_start, phase_stop));
            bucket += ms;
        }
    };

    auto check_residual = [&](int iteration) -> bool {
        auto residual_start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemset(gpu.d_pressure_error, 0, 2 * sizeof(double)));
        const int threads = 256;
        const int blocks = (N_cells + threads - 1) / threads;
        constexpr int spatial_block_size = 4;
        int spatial_block_count = 0;
        double block_low_frequency_fraction = std::numeric_limits<double>::quiet_NaN();
        const bool need_block_sums = use_poisson_spatial_diagnostics || use_poisson_two_grid_correction;
        int bx_count = 0;
        int by_count = 0;
        if (need_block_sums) {
            bx_count = (lx + spatial_block_size - 1) / spatial_block_size;
            by_count = (ly + spatial_block_size - 1) / spatial_block_size;
            const int bz_count = (lz + spatial_block_size - 1) / spatial_block_size;
            spatial_block_count = bx_count * by_count * bz_count;
            CUDA_CHECK(cudaMemset(gpu.d_pressure_block_sums, 0, spatial_block_count * sizeof(double)));
            pressureErrorSpatialKernel<<<blocks, threads, 2 * threads * sizeof(double)>>>(
                gpu.d_p, gpu.d_p_prev, gpu.d_pressure_error, gpu.d_pressure_block_sums,
                lx, ly, lz, lz_total, spatial_block_size, bx_count, by_count);
            CUDA_CHECK(cudaGetLastError());
            if (use_poisson_spatial_diagnostics) {
                CUDA_CHECK(cudaMemset(gpu.d_pressure_block_error, 0, sizeof(double)));
                const int block_sum_blocks = (spatial_block_count + threads - 1) / threads;
                blockAbsSumKernel<<<block_sum_blocks, threads, threads * sizeof(double)>>>(
                    gpu.d_pressure_block_sums, gpu.d_pressure_block_error, spatial_block_count);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            pressureErrorKernel<<<blocks, threads, 2 * threads * sizeof(double)>>>(
                gpu.d_p, gpu.d_p_prev, gpu.d_pressure_error, lx, ly, lz, lz_total);
            CUDA_CHECK(cudaGetLastError());
        }

        double h_err[2] = {0.0, 0.0};
        CUDA_CHECK(cudaMemcpy(h_err, gpu.d_pressure_error, sizeof(h_err), cudaMemcpyDeviceToHost));
        if (use_poisson_spatial_diagnostics) {
            double h_block_error = 0.0;
            CUDA_CHECK(cudaMemcpy(&h_block_error, gpu.d_pressure_block_error, sizeof(double), cudaMemcpyDeviceToHost));
            block_low_frequency_fraction = (h_err[0] > 0.0) ? (h_block_error / h_err[0]) : 0.0;
        }
        auto residual_end = std::chrono::steady_clock::now();
        if (detail_timing) {
            perf.poisson_residual_time +=
                std::chrono::duration<double, std::milli>(residual_end - residual_start).count();
        }
        const double rel_error = (h_err[1] > 0.0) ? (h_err[0] / h_err[1]) : 0.0;
        const bool converged = rel_error < tolerance;
        writePoissonDiagnostic(
            perf.time_step_count + 1,
            iteration,
            h_err[0],
            h_err[1],
            rel_error,
            converged,
            block_low_frequency_fraction,
            spatial_block_size,
            spatial_block_count
        );
        if (use_poisson_two_grid_correction && !converged && h_err[0] > 0.0) {
            coarsePressureCorrectionKernel<<<grid, block>>>(
                gpu.d_hh, gpu.d_p, gpu.d_p_prev, gpu.d_pressure_block_sums,
                lx, ly, lz, lz_total,
                spatial_block_size, bx_count, by_count,
                poisson_two_grid_strength
            );
            CUDA_CHECK(cudaGetLastError());
        }
        return converged;
    };

    auto run_poisson_iterations = [&](int iteration_limit, int requested_check_interval) -> std::pair<int, bool> {
        iteration_limit = std::max(1, std::min(iteration_limit, max_iterations));
        const int check_interval = requested_check_interval > 0 ? requested_check_interval : poisson_check_interval;
        int local_iterations_used = iteration_limit;
        bool converged = false;

        if (use_source_aware_hh_init && !use_scalar_poisson) {
            begin_phase();
            sourceAwareHHInitKernel<<<grid, block>>>(
                gpu.d_hh, gpu.d_p, gpu.d_rho,
                gpu.d_u_x, gpu.d_v_y, gpu.d_w_z,
                lx, ly, lz, lz_total,
                source_aware_hh_scale
            );
            CUDA_CHECK(cudaGetLastError());
            end_phase(perf.poisson_init_time);
        }

        const bool use_graph_for_this_run =
            graph_eligible && iteration_limit == max_iterations && check_interval == poisson_check_interval;
        if (use_graph_for_this_run) {
            if (!poisson_graph_exec || poisson_graph_check_interval != check_interval) {
                buildPoissonGraphSegment();
            }

            const int segments = iteration_limit / check_interval;
            for (int segment = 1; segment <= segments; ++segment) {
                CUDA_CHECK(cudaGraphLaunch(poisson_graph_exec, 0));
                if (check_residual(segment * check_interval)) {
                    local_iterations_used = segment * check_interval;
                    converged = true;
                    break;
                }
            }
            return {local_iterations_used, converged};
        }

        bool anderson_has_history = false;
        const int anderson_values = N_cells * 15;
        const int anderson_threads = 256;
        const int anderson_blocks = (anderson_values + anderson_threads - 1) / anderson_threads;

        // 与CPU版本一致：最多1000次，每100次检查相对压力变化。
        for (int iter = 1; iter <= iteration_limit; ++iter) {
            if (use_scalar_poisson) {
                begin_phase();
                scalarPressureJacobiKernel<<<onepass_grid, onepass_block>>>(
                    gpu.d_p, gpu.d_p_tmp, gpu.d_rho,
                    gpu.d_u_x, gpu.d_v_y, gpu.d_w_z,
                    lx, ly, lz, lz_total,
                    scalar_poisson_source_scale
                );
                CUDA_CHECK(cudaGetLastError());
                end_phase(perf.poisson_scalar_time);
                std::swap(gpu.d_p, gpu.d_p_tmp);
            } else if (use_onepass_poisson) {
                begin_phase();
                collisionStreamBoundaryPressureKernel<<<onepass_grid, onepass_block>>>(
                    gpu.d_hh, gpu.d_hh_tmp, gpu.d_p, gpu.d_p_tmp, gpu.d_rho,
                    gpu.d_u_x, gpu.d_v_y, gpu.d_w_z,
                    lx, ly, lz, lz_total,
                    pressure_relax_scale,
                    poisson_fixed_point_relax
                );
                CUDA_CHECK(cudaGetLastError());
                end_phase(perf.poisson_onepass_time);
                if (use_poisson_anderson_m1) {
                    if (!anderson_has_history) {
                        andersonStoreHistoryKernel<<<anderson_blocks, anderson_threads>>>(
                            gpu.d_hh, gpu.d_hh_tmp,
                            gpu.d_anderson_prev_residual,
                            gpu.d_anderson_prev_image,
                            anderson_values
                        );
                        CUDA_CHECK(cudaGetLastError());
                        anderson_has_history = true;
                    } else {
                        CUDA_CHECK(cudaMemset(gpu.d_anderson_stats, 0, 2 * sizeof(double)));
                        andersonM1DotKernel<<<
                            anderson_blocks,
                            anderson_threads,
                            2 * anderson_threads * sizeof(double)>>>(
                            gpu.d_hh, gpu.d_hh_tmp,
                            gpu.d_anderson_prev_residual,
                            gpu.d_anderson_stats,
                            anderson_values
                        );
                        CUDA_CHECK(cudaGetLastError());
                        double h_stats[2] = {0.0, 0.0};
                        CUDA_CHECK(cudaMemcpy(h_stats, gpu.d_anderson_stats, sizeof(h_stats), cudaMemcpyDeviceToHost));
                        double alpha = 1.0;
                        if (h_stats[1] > 1.0e-300) {
                            alpha = -h_stats[0] / h_stats[1];
                        }
                        if (!std::isfinite(alpha)) {
                            alpha = 1.0;
                        }
                        const double lo = -poisson_anderson_beta_max;
                        const double hi = 1.0 + poisson_anderson_beta_max;
                        alpha = std::max(lo, std::min(hi, alpha));
                        andersonM1ApplyKernel<<<onepass_grid, onepass_block>>>(
                            gpu.d_hh, gpu.d_hh_tmp,
                            gpu.d_anderson_prev_image,
                            gpu.d_anderson_prev_residual,
                            gpu.d_anderson_prev_image,
                            gpu.d_p_tmp,
                            lx, ly, lz, lz_total,
                            alpha
                        );
                        CUDA_CHECK(cudaGetLastError());
                    }
                }
                std::swap(gpu.d_hh, gpu.d_hh_tmp);
                std::swap(gpu.d_p, gpu.d_p_tmp);
            } else if (use_fused_poisson) {
                begin_phase();
                collisionPressureStreamKernel<<<grid, block>>>(
                    gpu.d_hh, gpu.d_hh_tmp, gpu.d_p, gpu.d_rho,
                    gpu.d_u_x, gpu.d_v_y, gpu.d_w_z,
                    lx, ly, lz, lz_total,
                    pressure_relax_scale
                );
                CUDA_CHECK(cudaGetLastError());
                end_phase(perf.poisson_fused_time);
                std::swap(gpu.d_hh, gpu.d_hh_tmp);
            } else {
                begin_phase();
                collisionPressureKernel<<<grid, block>>>(
                    gpu.d_hh, gpu.d_p, gpu.d_rho,
                    gpu.d_u_x, gpu.d_v_y, gpu.d_w_z,
                    lx, ly, lz, lz_total,
                    pressure_relax_scale
                );
                CUDA_CHECK(cudaGetLastError());
                end_phase(perf.poisson_collision_time);

                begin_phase();
                streamKernel<<<grid, block>>>(gpu.d_hh, gpu.d_hh_tmp, lx, ly, lz);
                CUDA_CHECK(cudaGetLastError());
                end_phase(perf.poisson_stream_time);
                std::swap(gpu.d_hh, gpu.d_hh_tmp);
            }

            if (!use_onepass_poisson && use_fused_boundary_pressure) {
                begin_phase();
                boundaryAndComputePressureKernel<<<grid, block>>>(
                    gpu.d_hh, gpu.d_p,
                    lx, ly, lz, lz_total
                );
                CUDA_CHECK(cudaGetLastError());
                end_phase(perf.poisson_boundary_pressure_time);
            } else if (!use_onepass_poisson) {
                // 3. hh边界
                begin_phase();
                slipBounceBackKernel<<<grid2d, block2d>>>(gpu.d_hh, lx, ly, lz);
                CUDA_CHECK(cudaGetLastError());
                end_phase(perf.poisson_boundary_time);

                // 4. 计算压力
                begin_phase();
                computePressureKernel<<<grid, block>>>(
                    gpu.d_hh, gpu.d_p,
                    lx, ly, lz, lz_total
                );
                CUDA_CHECK(cudaGetLastError());
                end_phase(perf.poisson_pressure_time);
            }

            if ((iter % check_interval == 0) || iter == iteration_limit) {
                if (check_residual(iter)) {
                    local_iterations_used = iter;
                    converged = true;
                    break;
                }
            }
        }
        return {local_iterations_used, converged};
    };

    if (pressure_initializer_loaded) {
        CUDA_CHECK(cudaMemcpy(gpu.d_p_backup, gpu.d_p, N_macro * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(gpu.d_hh_backup, gpu.d_hh, N_cells * Q * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(gpu.d_p_prev_backup, gpu.d_p_prev, N_cells * sizeof(double), cudaMemcpyDeviceToDevice));
        applyPressureInitializer();
        perf.pressure_initializer_attempts++;
    }

    int first_iteration_limit = max_iterations;
    if (pressure_initializer_loaded && pressure_initializer_max_iterations > 0) {
        first_iteration_limit = std::min(pressure_initializer_max_iterations, max_iterations);
    }
    const int first_check_interval =
        pressure_initializer_loaded && pressure_initializer_check_interval > 0
            ? pressure_initializer_check_interval
            : poisson_check_interval;
    auto [first_iterations, first_converged] =
        run_poisson_iterations(first_iteration_limit, first_check_interval);
    perf.total_poisson_iterations += first_iterations;

    if (pressure_initializer_loaded) {
        if (first_converged) {
            perf.pressure_initializer_accepts++;
        } else {
            perf.pressure_initializer_fallbacks++;
            CUDA_CHECK(cudaMemcpy(gpu.d_p, gpu.d_p_backup, N_macro * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(gpu.d_hh, gpu.d_hh_backup, N_cells * Q * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(gpu.d_p_prev, gpu.d_p_prev_backup, N_cells * sizeof(double), cudaMemcpyDeviceToDevice));
            auto [fallback_iterations, fallback_converged] =
                run_poisson_iterations(max_iterations, poisson_check_interval);
            (void)fallback_converged;
            perf.total_poisson_iterations += fallback_iterations;
        }
    }
    if (detail_timing) {
        CUDA_CHECK(cudaEventDestroy(phase_start));
        CUDA_CHECK(cudaEventDestroy(phase_stop));
    }
}

void InamuroCUDA::doCorrectUVWAndHH()
{
    dim3 block(8, 8, 8);
    dim3 grid((lx+block.x-1)/block.x, (ly+block.y-1)/block.y, (lz+block.z-1)/block.z);

    // 1. 速度修正
    correctVelocityKernel<<<grid, block>>>(
        gpu.d_u, gpu.d_v, gpu.d_w,
        gpu.d_p, gpu.d_rho,
        lx, ly, lz, lz_total
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. 更新hh
    updateHHKernel<<<grid, block>>>(
        gpu.d_hh, gpu.d_p,
        lx, ly, lz, lz_total
    );
    CUDA_CHECK(cudaGetLastError());
}

void InamuroCUDA::performTimeStepGPU()
{
    performTimeStepGPUImpl(nullptr, 0, nullptr, "tecplot", false, false);
}

void InamuroCUDA::performTimeStepGPUWithPoissonPair(
    Inamuro& cpuSolver,
    int completed_step,
    const std::string& pair_output_dir,
    const std::string& pair_format,
    bool write_pre_pair,
    bool write_post_pair)
{
    performTimeStepGPUImpl(&cpuSolver, completed_step, &pair_output_dir, pair_format, write_pre_pair, write_post_pair);
}

void InamuroCUDA::writePoissonPairSnapshot(
    Inamuro& cpuSolver,
    int completed_step,
    const std::string& pair_output_dir,
    const std::string& phase) const
{
    const std::filesystem::path dir = std::filesystem::path(pair_output_dir) / phase;
    std::filesystem::create_directories(dir);
    const std::string old_dir = cpuSolver.getOutputDirectory();
    cpuSolver.setOutputDirectory(dir.string());
    downloadMacroToCPU(cpuSolver);
    cpuSolver.writeResults(completed_step);
    cpuSolver.setOutputDirectory(old_dir);
}

void InamuroCUDA::writePoissonFeatureSnapshot(
    int completed_step,
    const std::string& pair_output_dir,
    const std::string& phase) const
{
    static constexpr char magic[8] = {'P', 'I', 'N', 'N', 'F', '2', '\0', '\0'};
    static constexpr std::int32_t nfields = 7;

    const std::filesystem::path dir = std::filesystem::path(pair_output_dir) / phase;
    std::filesystem::create_directories(dir);

    std::ostringstream name;
    name << "3D" << std::setfill('0') << std::setw(9) << completed_step << ".bin";
    const std::filesystem::path path = dir / name.str();

    const std::size_t packed_count = static_cast<std::size_t>(nfields) * static_cast<std::size_t>(N_cells);
    float* d_packed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_packed, packed_count * sizeof(float)));

    const int threads = 256;
    const int blocks = static_cast<int>((packed_count + threads - 1) / threads);
    packPoissonFeatureSnapshotKernel<<<blocks, threads>>>(
        gpu.d_u, gpu.d_v, gpu.d_w, gpu.d_rho, gpu.d_fei, gpu.d_p,
        gpu.d_u_x, gpu.d_v_y, gpu.d_w_z,
        d_packed, lx, ly, lz, lz_total);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> packed(packed_count);
    CUDA_CHECK(cudaMemcpy(packed.data(), d_packed, packed_count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_packed));

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open Poisson feature snapshot: " + path.string());
    }

    const std::int32_t header[4] = {
        static_cast<std::int32_t>(lx),
        static_cast<std::int32_t>(ly),
        static_cast<std::int32_t>(lz),
        nfields,
    };
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(packed.data()),
              static_cast<std::streamsize>(packed.size() * sizeof(float)));

    if (!out) {
        throw std::runtime_error("failed to write Poisson feature snapshot: " + path.string());
    }
}

void InamuroCUDA::writePoissonStateSnapshot(
    int completed_step,
    const std::string& pair_output_dir,
    const std::string& phase) const
{
    static constexpr char magic[8] = {'P', 'I', 'N', 'N', 'S', '1', '\0', '\0'};
    const std::filesystem::path dir = std::filesystem::path(pair_output_dir) / phase;
    std::filesystem::create_directories(dir);

    std::ostringstream name;
    name << "3D" << std::setfill('0') << std::setw(9) << completed_step << ".bin";
    const std::filesystem::path path = dir / name.str();

    std::vector<double> pressure(N_cells);
    std::vector<double> hh(static_cast<std::size_t>(N_cells) * Q);
    const std::size_t physical_offset = static_cast<std::size_t>(lx) * ly;
    CUDA_CHECK(cudaMemcpy(
        pressure.data(), gpu.d_p + physical_offset,
        pressure.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        hh.data(), gpu.d_hh,
        hh.size() * sizeof(double), cudaMemcpyDeviceToHost));

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open Poisson state snapshot: " + path.string());
    }
    const std::int32_t dims[3] = {
        static_cast<std::int32_t>(lx),
        static_cast<std::int32_t>(ly),
        static_cast<std::int32_t>(lz),
    };
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(pressure.data()),
              static_cast<std::streamsize>(pressure.size() * sizeof(double)));
    out.write(reinterpret_cast<const char*>(hh.data()),
              static_cast<std::streamsize>(hh.size() * sizeof(double)));
    if (!out) {
        throw std::runtime_error("failed to write Poisson state snapshot: " + path.string());
    }
}

void InamuroCUDA::performTimeStepGPUImpl(
    Inamuro* pair_solver,
    int completed_step,
    const std::string* pair_output_dir,
    const std::string& pair_format,
    bool write_pre_pair,
    bool write_post_pair)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    const bool write_pair = pair_output_dir != nullptr && !pair_output_dir->empty() &&
        (pair_format == "features" || pair_solver != nullptr);
    
    if (enable_timing) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    // 1) 碰撞 + 求导/拉普拉斯
    if (enable_timing) cudaEventRecord(start);
    doCollisionAndGradients();
    if (enable_timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        perf.total_collision_time += milliseconds;
    }

    // 2) 迁移
    if (enable_timing) cudaEventRecord(start);
    doStreamFF();
    doStreamGG();
    if (enable_timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        perf.total_stream_time += milliseconds;
    }

    // 3) 边界
    doBoundaryFF();
    doBoundaryGG();

    // 4) 宏观量
    if (enable_timing) cudaEventRecord(start);
    doMacro();
    if (enable_timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        perf.total_macro_time += milliseconds;
    }
    auto write_pair_snapshot = [&](const std::string& phase) {
        if (pair_format == "features") {
            writePoissonFeatureSnapshot(completed_step, *pair_output_dir, phase);
        } else if (pair_format == "state") {
            writePoissonStateSnapshot(completed_step, *pair_output_dir, phase);
        } else if (pair_format == "tecplot") {
            writePoissonPairSnapshot(*pair_solver, completed_step, *pair_output_dir, phase);
        } else {
            throw std::runtime_error("unsupported Poisson pair format: " + pair_format);
        }
    };

    if (write_pair && write_pre_pair) {
        write_pair_snapshot("pre_poisson");
    }
    if (completed_step > 0 && !poisson_state_export_dir.empty() &&
        (poisson_state_export_phase == "pre" || poisson_state_export_phase == "both") &&
        poisson_state_export_steps.count(completed_step) > 0) {
        writePoissonStateSnapshot(completed_step, poisson_state_export_dir, "pre_poisson");
    }
    if (!pressure_initializer_wait_dir.empty() && completed_step > 0 &&
        (pressure_initializer_wait_max_step <= 0 || completed_step <= pressure_initializer_wait_max_step)) {
        loadPressureInitializerForStep(completed_step);
    } else if (!pressure_initializer_wait_dir.empty()) {
        setPressureInitializer("", pressure_initializer_mode);
    }

    // 5) 压力 Poisson
    if (enable_timing) cudaEventRecord(start);
    doPressurePoisson();
    if (enable_timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        perf.total_poisson_time += milliseconds;
    }
    if (write_pair && write_post_pair) {
        write_pair_snapshot("post_poisson");
    }
    if (completed_step > 0 && !poisson_state_export_dir.empty() &&
        (poisson_state_export_phase == "post" || poisson_state_export_phase == "both") &&
        poisson_state_export_steps.count(completed_step) > 0) {
        writePoissonStateSnapshot(completed_step, poisson_state_export_dir, "post_poisson");
    }

    // 6) 速度修正 + hh 更新
    doCorrectUVWAndHH();
    
    if (enable_timing) {
        perf.time_step_count++;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    if (enable_debug && perf.time_step_count % 100 == 0) {
        std::cout << "[DEBUG] Time step " << perf.time_step_count << " completed" << std::endl;
    }
}

// ==================== 性能测量辅助函数 ====================

void InamuroCUDA::setUseFusedPoisson(bool enabled)
{
    if (use_fused_poisson != enabled) {
        destroyPoissonGraph();
    }
    use_fused_poisson = enabled;
}

void InamuroCUDA::setUseOnePassPoisson(bool enabled)
{
    if (use_onepass_poisson != enabled) {
        destroyPoissonGraph();
    }
    use_onepass_poisson = enabled;
}

void InamuroCUDA::setUseScalarPoisson(bool enabled)
{
    if (enabled) {
        throw std::invalid_argument(
            "scalar/simple Poisson is disabled; use the book D3Q15 pressure operator");
    }
    use_scalar_poisson = false;
}

void InamuroCUDA::setScalarPoissonSourceScale(double scale)
{
    scalar_poisson_source_scale = scale;
}

void InamuroCUDA::setUseSourceAwareHHInit(bool enabled)
{
    if (enabled) {
        throw std::invalid_argument(
            "source-aware guessed hh initialization is disabled; load a complete 15-channel hh state");
    }
    use_source_aware_hh_init = false;
}

void InamuroCUDA::setSourceAwareHHScale(double scale)
{
    source_aware_hh_scale = scale;
}

void InamuroCUDA::setPressureRelaxScale(double scale)
{
    if (!(scale > 0.0)) {
        throw std::invalid_argument("pressure relaxation scale must be positive");
    }
    if (pressure_relax_scale != scale) {
        destroyPoissonGraph();
    }
    pressure_relax_scale = scale;
}

void InamuroCUDA::setPoissonFixedPointRelax(double omega)
{
    if (!(omega > 0.0)) {
        throw std::invalid_argument("Poisson fixed-point relaxation must be positive");
    }
    if (poisson_fixed_point_relax != omega) {
        destroyPoissonGraph();
    }
    poisson_fixed_point_relax = omega;
}

void InamuroCUDA::setUsePoissonAndersonM1(bool enabled)
{
    if (use_poisson_anderson_m1 != enabled) {
        destroyPoissonGraph();
    }
    use_poisson_anderson_m1 = enabled;
}

void InamuroCUDA::setPoissonAndersonBetaMax(double value)
{
    if (!(value >= 0.0)) {
        throw std::invalid_argument("Poisson Anderson beta max must be non-negative");
    }
    poisson_anderson_beta_max = value;
}

void InamuroCUDA::setUsePoissonTwoGridCorrection(bool enabled)
{
    if (use_poisson_two_grid_correction != enabled) {
        destroyPoissonGraph();
    }
    use_poisson_two_grid_correction = enabled;
}

void InamuroCUDA::setPoissonTwoGridStrength(double value)
{
    if (!(value >= 0.0)) {
        throw std::invalid_argument("Poisson two-grid strength must be non-negative");
    }
    poisson_two_grid_strength = value;
}

void InamuroCUDA::setUseFusedBoundaryPressure(bool enabled)
{
    if (use_fused_boundary_pressure != enabled) {
        destroyPoissonGraph();
    }
    use_fused_boundary_pressure = enabled;
}

void InamuroCUDA::setUsePoissonGraph(bool enabled)
{
    if (use_poisson_graph != enabled) {
        destroyPoissonGraph();
    }
    use_poisson_graph = enabled;
}

void InamuroCUDA::setEnablePoissonDetailTiming(bool enabled)
{
    enable_poisson_detail_timing = enabled;
}

void InamuroCUDA::setPoissonConvergence(int check_interval, double tolerance)
{
    if (check_interval <= 0) {
        throw std::invalid_argument("poisson check interval must be positive");
    }
    if (!(tolerance > 0.0)) {
        throw std::invalid_argument("poisson tolerance must be positive");
    }
    if (poisson_check_interval != check_interval) {
        destroyPoissonGraph();
    }
    poisson_check_interval = check_interval;
    poisson_tolerance = tolerance;
}

void InamuroCUDA::setPoissonDiagnosticsPath(const std::string& path)
{
    poisson_diagnostics_path = path;
    if (poisson_diagnostics_path.empty()) {
        return;
    }

    std::ofstream out(poisson_diagnostics_path);
    if (!out) {
        throw std::runtime_error("failed to open Poisson diagnostics CSV: " + poisson_diagnostics_path);
    }
    out << "step,iteration,pressure_l1_delta,pressure_l1_norm,relative_error,converged,"
        << "block_low_frequency_fraction,block_size,block_count\n";
}

void InamuroCUDA::setUsePoissonSpatialDiagnostics(bool enabled)
{
    use_poisson_spatial_diagnostics = enabled;
}

void InamuroCUDA::setPressureInitializer(const std::string& path, const std::string& mode)
{
    if (path.empty()) {
        pressure_initializer_path.clear();
        pressure_initializer_loaded = false;
        pressure_initializer_has_hh = false;
        return;
    }
    if (mode != "absolute" && mode != "delta") {
        throw std::invalid_argument("pressure initializer mode must be absolute or delta");
    }

    pressure_initializer_path = path;
    pressure_initializer_mode = mode;
    loadPressureInitializer();
}

void InamuroCUDA::setPressureInitializerMaxIterations(int max_iterations)
{
    if (max_iterations < 0) {
        throw std::invalid_argument("pressure initializer max iterations must be non-negative");
    }
    pressure_initializer_max_iterations = max_iterations;
}

void InamuroCUDA::setPressureInitializerCheckInterval(int check_interval)
{
    if (check_interval < 0) {
        throw std::invalid_argument("pressure initializer check interval must be non-negative");
    }
    pressure_initializer_check_interval = check_interval;
}

void InamuroCUDA::setPressureInitializerWaitDir(const std::string& dir, int timeout_ms, int max_step)
{
    if (timeout_ms < 0) {
        throw std::invalid_argument("pressure initializer wait timeout must be non-negative");
    }
    if (max_step < 0) {
        throw std::invalid_argument("pressure initializer wait max step must be non-negative");
    }
    pressure_initializer_wait_dir = dir;
    pressure_initializer_wait_timeout_ms = timeout_ms;
    pressure_initializer_wait_max_step = max_step;
}

void InamuroCUDA::setPoissonStateExport(
    const std::string& dir, const std::set<int>& steps, const std::string& phase)
{
    if (phase != "pre" && phase != "post" && phase != "both") {
        throw std::invalid_argument("Poisson state export phase must be pre, post, or both");
    }
    poisson_state_export_dir = dir;
    poisson_state_export_steps = steps;
    poisson_state_export_phase = phase;
}

void InamuroCUDA::loadPressureInitializerForStep(int completed_step)
{
    auto step_stem = [](int step) {
        std::ostringstream name;
        name << "3D" << std::setfill('0') << std::setw(9) << step;
        return name.str();
    };

    const std::filesystem::path dir(pressure_initializer_wait_dir);
    const std::filesystem::path init_path = dir / (step_stem(completed_step) + ".bin");
    const std::filesystem::path skip_path = dir / (step_stem(completed_step) + ".skip");
    const auto start = std::chrono::steady_clock::now();

    while (true) {
        if (std::filesystem::exists(init_path)) {
            setPressureInitializer(init_path.string(), pressure_initializer_mode);
            return;
        }
        if (std::filesystem::exists(skip_path)) {
            setPressureInitializer("", pressure_initializer_mode);
            return;
        }
        if (pressure_initializer_wait_timeout_ms == 0) {
            setPressureInitializer("", pressure_initializer_mode);
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (elapsed_ms >= pressure_initializer_wait_timeout_ms) {
            setPressureInitializer("", pressure_initializer_mode);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void InamuroCUDA::loadPressureInitializer()
{
    static constexpr char pressure_magic[8] = {'P', 'I', 'N', 'N', 'P', '1', '\0', '\0'};
    static constexpr char state_magic[8] = {'P', 'I', 'N', 'N', 'S', '1', '\0', '\0'};
    std::ifstream in(pressure_initializer_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open pressure initializer: " + pressure_initializer_path);
    }

    char magic[8] = {};
    std::int32_t dims[3] = {};
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(dims), sizeof(dims));
    const bool has_hh = std::memcmp(magic, state_magic, sizeof(magic)) == 0;
    const bool pressure_only = std::memcmp(magic, pressure_magic, sizeof(magic)) == 0;
    if (!in || (!pressure_only && !has_hh)) {
        throw std::runtime_error("invalid pressure initializer header: " + pressure_initializer_path);
    }
    if (dims[0] != lx || dims[1] != ly || dims[2] != lz) {
        throw std::runtime_error("pressure initializer dimensions do not match solver grid");
    }

    std::vector<double> values(N_cells);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!in) {
        throw std::runtime_error("truncated pressure initializer: " + pressure_initializer_path);
    }
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("pressure initializer contains non-finite values");
        }
    }
    std::vector<double> hh_values;
    if (has_hh) {
        if (pressure_initializer_mode != "absolute") {
            throw std::runtime_error("Poisson p+hh state initializer requires absolute mode");
        }
        hh_values.resize(static_cast<std::size_t>(N_cells) * Q);
        in.read(reinterpret_cast<char*>(hh_values.data()),
                static_cast<std::streamsize>(hh_values.size() * sizeof(double)));
        if (!in) {
            throw std::runtime_error("truncated hh state initializer: " + pressure_initializer_path);
        }
        for (double value : hh_values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("hh state initializer contains non-finite values");
            }
        }
    }

    if (!gpu.d_pressure_init) {
        CUDA_CHECK(cudaMalloc(&gpu.d_pressure_init, values.size() * sizeof(double)));
    }
    if (!gpu.d_p_backup) {
        CUDA_CHECK(cudaMalloc(&gpu.d_p_backup, N_macro * sizeof(double)));
    }
    if (!gpu.d_hh_backup) {
        CUDA_CHECK(cudaMalloc(&gpu.d_hh_backup, N_cells * Q * sizeof(double)));
    }
    if (!gpu.d_p_prev_backup) {
        CUDA_CHECK(cudaMalloc(&gpu.d_p_prev_backup, N_cells * sizeof(double)));
    }
    if (has_hh && !gpu.d_hh_init) {
        CUDA_CHECK(cudaMalloc(&gpu.d_hh_init, hh_values.size() * sizeof(double)));
    }
    CUDA_CHECK(cudaMemcpy(gpu.d_pressure_init, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (has_hh) {
        CUDA_CHECK(cudaMemcpy(
            gpu.d_hh_init, hh_values.data(), hh_values.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    pressure_initializer_has_hh = has_hh;
    pressure_initializer_loaded = true;
}

void InamuroCUDA::applyPressureInitializer()
{
    if (!pressure_initializer_loaded) {
        return;
    }

    auto start = std::chrono::steady_clock::now();
    const int threads = 256;
    const int blocks = (N_cells + threads - 1) / threads;
    const int is_delta = (pressure_initializer_mode == "delta") ? 1 : 0;
    applyPressureInitializerKernel<<<blocks, threads>>>(
        gpu.d_p, gpu.d_pressure_init, lx, ly, lz, lz_total, is_delta);
    CUDA_CHECK(cudaGetLastError());

    if (pressure_initializer_has_hh) {
        CUDA_CHECK(cudaMemcpy(
            gpu.d_hh, gpu.d_hh_init,
            static_cast<std::size_t>(N_cells) * Q * sizeof(double),
            cudaMemcpyDeviceToDevice));
    } else {
        dim3 block3(8, 8, 8);
        dim3 grid3((lx + block3.x - 1) / block3.x,
                   (ly + block3.y - 1) / block3.y,
                   (lz + block3.z - 1) / block3.z);
        updateHHKernel<<<grid3, block3>>>(gpu.d_hh, gpu.d_p, lx, ly, lz, lz_total);
        CUDA_CHECK(cudaGetLastError());
    }
    seedPressureResidualFromCurrentPressure();
    auto end = std::chrono::steady_clock::now();
    perf.pressure_initializer_time +=
        std::chrono::duration<double, std::milli>(end - start).count();
}

void InamuroCUDA::seedPressureResidualFromCurrentPressure()
{
    const int threads = 256;
    const int blocks = (N_cells + threads - 1) / threads;
    seedPressurePrevKernel<<<blocks, threads>>>(gpu.d_p, gpu.d_p_prev, lx, ly, lz, lz_total);
    CUDA_CHECK(cudaGetLastError());
}

void InamuroCUDA::writePoissonDiagnostic(
    int step,
    int iteration,
    double pressure_l1_delta,
    double pressure_l1_norm,
    double relative_error,
    bool converged,
    double block_low_frequency_fraction,
    int block_size,
    int block_count) const
{
    if (poisson_diagnostics_path.empty()) {
        return;
    }

    std::ofstream out(poisson_diagnostics_path, std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to append Poisson diagnostics CSV: " + poisson_diagnostics_path);
    }
    out << step << ','
        << iteration << ','
        << std::setprecision(17) << pressure_l1_delta << ','
        << pressure_l1_norm << ','
        << relative_error << ','
        << (converged ? 1 : 0) << ',';
    if (std::isfinite(block_low_frequency_fraction)) {
        out << block_low_frequency_fraction;
    }
    out << ','
        << (block_count > 0 ? block_size : 0) << ','
        << block_count << '\n';
}

void InamuroCUDA::printPerformanceMetrics() const
{
    if (perf.time_step_count == 0) {
        std::cout << "No performance data collected yet." << std::endl;
        return;
    }
    
    const double sites = static_cast<double>(N_cells);
    std::cout << "\n========== GPU Performance Metrics ==========" << std::endl;
    std::cout << "Total time steps: " << perf.time_step_count << std::endl;
    std::cout << "Poisson mode: "
              << (use_scalar_poisson ? "scalar" : (use_onepass_poisson ? "onepass" : (use_fused_poisson ? "fused" : "split")))
              << std::endl;
    std::cout << "Boundary-pressure mode: "
              << (use_fused_boundary_pressure ? "fused" : "split") << std::endl;
    std::cout << "Poisson graph: " << (use_poisson_graph ? "enabled" : "disabled") << std::endl;
    std::cout << "Poisson convergence: check_interval=" << poisson_check_interval
              << ", tolerance=" << poisson_tolerance << std::endl;
    if (use_scalar_poisson) {
        std::cout << "Scalar Poisson source scale: " << scalar_poisson_source_scale << std::endl;
    }
    std::cout << "Source-aware hh init: "
              << (use_source_aware_hh_init ? "enabled" : "disabled")
              << ", scale=" << source_aware_hh_scale << std::endl;
    std::cout << "Pressure relaxation scale: " << pressure_relax_scale << std::endl;
    std::cout << "Poisson fixed-point relaxation: " << poisson_fixed_point_relax << std::endl;
    std::cout << "Poisson Anderson m1: "
              << (use_poisson_anderson_m1 ? "enabled" : "disabled")
              << ", beta_max=" << poisson_anderson_beta_max << std::endl;
    std::cout << "Poisson spatial diagnostics: "
              << (use_poisson_spatial_diagnostics ? "enabled" : "disabled") << std::endl;
    std::cout << "Poisson two-grid correction: "
              << (use_poisson_two_grid_correction ? "enabled" : "disabled")
              << ", strength=" << poisson_two_grid_strength << std::endl;
    const bool pressure_initializer_was_used =
        pressure_initializer_loaded || perf.pressure_initializer_attempts > 0;
    std::cout << "PINN pressure initializer: "
              << (pressure_initializer_was_used ? "enabled" : "disabled")
              << ", mode=" << pressure_initializer_mode
              << ", max_iterations="
              << (pressure_initializer_max_iterations > 0 ? pressure_initializer_max_iterations : 1000)
              << ", check_interval="
              << (pressure_initializer_check_interval > 0 ? pressure_initializer_check_interval : poisson_check_interval)
              << ", attempts=" << perf.pressure_initializer_attempts
              << ", accepts=" << perf.pressure_initializer_accepts
              << ", fallbacks=" << perf.pressure_initializer_fallbacks << std::endl;
    std::cout << "\nAverage time per kernel (ms):" << std::endl;
    std::cout << "  Collision:      " << perf.total_collision_time / perf.time_step_count << std::endl;
    std::cout << "  Stream:         " << perf.total_stream_time / perf.time_step_count << std::endl;
    std::cout << "  Macro:          " << perf.total_macro_time / perf.time_step_count << std::endl;
    std::cout << "  Poisson:        " << perf.total_poisson_time / perf.time_step_count << std::endl;
    std::cout << "  Poisson iters:  "
              << static_cast<double>(perf.total_poisson_iterations) / perf.time_step_count << std::endl;
    
    double total_avg = (perf.total_collision_time + perf.total_stream_time + 
                        perf.total_macro_time + perf.total_poisson_time) / perf.time_step_count;
    std::cout << "  Total per step: " << total_avg << std::endl;
    std::cout << "  MLUPS:          " << (sites / (total_avg * 1000.0)) << std::endl;

    const double avg_iters = static_cast<double>(perf.total_poisson_iterations) / perf.time_step_count;
    const double detail_total =
        perf.poisson_collision_time + perf.poisson_stream_time + perf.poisson_fused_time +
        perf.poisson_onepass_time +
        perf.poisson_scalar_time +
        perf.poisson_init_time +
        perf.pressure_initializer_time +
        perf.poisson_boundary_time + perf.poisson_pressure_time +
        perf.poisson_boundary_pressure_time + perf.poisson_residual_time;
    if (avg_iters > 0.0 && detail_total > 0.0) {
        std::cout << "\nAverage Poisson detail (ms/step):" << std::endl;
        std::cout << "  pressure collision: " << perf.poisson_collision_time / perf.time_step_count << std::endl;
        std::cout << "  pressure stream:    " << perf.poisson_stream_time / perf.time_step_count << std::endl;
        std::cout << "  pressure fused:     " << perf.poisson_fused_time / perf.time_step_count << std::endl;
        std::cout << "  pressure onepass:   " << perf.poisson_onepass_time / perf.time_step_count << std::endl;
        std::cout << "  pressure scalar:    " << perf.poisson_scalar_time / perf.time_step_count << std::endl;
        std::cout << "  hh init:            " << perf.poisson_init_time / perf.time_step_count << std::endl;
        std::cout << "  PINN p init:        " << perf.pressure_initializer_time / perf.time_step_count << std::endl;
        std::cout << "  pressure boundary:  " << perf.poisson_boundary_time / perf.time_step_count << std::endl;
        std::cout << "  pressure sum:       " << perf.poisson_pressure_time / perf.time_step_count << std::endl;
        std::cout << "  boundary+sum fused: " << perf.poisson_boundary_pressure_time / perf.time_step_count << std::endl;
        std::cout << "  residual check:     " << perf.poisson_residual_time / perf.time_step_count << std::endl;
        std::cout << "Average Poisson detail (us/iter):" << std::endl;
        std::cout << "  pressure collision: " << perf.poisson_collision_time * 1000.0 / perf.total_poisson_iterations << std::endl;
        std::cout << "  pressure stream:    " << perf.poisson_stream_time * 1000.0 / perf.total_poisson_iterations << std::endl;
        std::cout << "  pressure fused:     " << perf.poisson_fused_time * 1000.0 / perf.total_poisson_iterations << std::endl;
        std::cout << "  pressure onepass:   " << perf.poisson_onepass_time * 1000.0 / perf.total_poisson_iterations << std::endl;
        std::cout << "  pressure scalar:    " << perf.poisson_scalar_time * 1000.0 / perf.total_poisson_iterations << std::endl;
        std::cout << "  hh init:            " << perf.poisson_init_time * 1000.0 / perf.total_poisson_iterations << std::endl;
        std::cout << "  PINN p init:        " << perf.pressure_initializer_time * 1000.0 / perf.total_poisson_iterations << std::endl;
        std::cout << "  pressure boundary:  " << perf.poisson_boundary_time * 1000.0 / perf.total_poisson_iterations << std::endl;
        std::cout << "  pressure sum:       " << perf.poisson_pressure_time * 1000.0 / perf.total_poisson_iterations << std::endl;
        std::cout << "  boundary+sum fused: " << perf.poisson_boundary_pressure_time * 1000.0 / perf.total_poisson_iterations << std::endl;
    }
    
    std::cout << "\nKernel time distribution:" << std::endl;
    double total = perf.total_collision_time + perf.total_stream_time + 
                   perf.total_macro_time + perf.total_poisson_time;
    std::cout << "  Collision:      " << (perf.total_collision_time / total * 100) << "%" << std::endl;
    std::cout << "  Stream:         " << (perf.total_stream_time / total * 100) << "%" << std::endl;
    std::cout << "  Macro:          " << (perf.total_macro_time / total * 100) << "%" << std::endl;
    std::cout << "  Poisson:        " << (perf.total_poisson_time / total * 100) << "%" << std::endl;
    std::cout << "=============================================\n" << std::endl;
}

void InamuroCUDA::printRooflineSummary() const
{
    const double q = 15.0;
    const double bytes_per_double = 8.0;
    const double cells = static_cast<double>(N_cells);

    const double stream_bytes_per_site = 2.0 * q * bytes_per_double;
    const double macro_bytes_per_site = (2.0 * q + 5.0) * bytes_per_double;
    const double gradient_lap_bytes_per_site =
        (5.0 * 14.0 + 4.0 * 15.0 + 15.0 + 15.0 + 18.0 + 4.0) * bytes_per_double;
    const double poisson_iter_bytes_per_site =
        (15.0 + 3.0 + 15.0 + 15.0 + 15.0 + 1.0) * bytes_per_double;

    std::cout << "\n========== Static Roofline Model ==========" << std::endl;
    std::cout << "Grid cells: " << N_cells << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Estimated bytes/site/update:" << std::endl;
    std::cout << "  stream(ff+gg):       " << (2.0 * stream_bytes_per_site) << std::endl;
    std::cout << "  macro:               " << macro_bytes_per_site << std::endl;
    std::cout << "  gradient+lap+coll:   " << gradient_lap_bytes_per_site << std::endl;
    std::cout << "  poisson / iteration: " << poisson_iter_bytes_per_site << std::endl;
    std::cout << "Estimated total bytes for one non-Poisson step part: "
              << cells * (2.0 * stream_bytes_per_site + macro_bytes_per_site + gradient_lap_bytes_per_site) / 1.0e9
              << " GB" << std::endl;
    std::cout << "Use measured Poisson iteration count to add: cells * "
              << poisson_iter_bytes_per_site << " bytes/iter." << std::endl;
    std::cout << "===========================================\n" << std::endl;
}

void InamuroCUDA::resetPerformanceMetrics()
{
    perf.total_collision_time = 0.0;
    perf.total_stream_time = 0.0;
    perf.total_macro_time = 0.0;
    perf.total_poisson_time = 0.0;
    perf.poisson_collision_time = 0.0;
    perf.poisson_stream_time = 0.0;
    perf.poisson_fused_time = 0.0;
    perf.poisson_onepass_time = 0.0;
    perf.poisson_scalar_time = 0.0;
    perf.poisson_init_time = 0.0;
    perf.poisson_boundary_time = 0.0;
    perf.poisson_pressure_time = 0.0;
    perf.poisson_boundary_pressure_time = 0.0;
    perf.poisson_residual_time = 0.0;
    perf.pressure_initializer_time = 0.0;
    perf.total_poisson_iterations = 0;
    perf.pressure_initializer_attempts = 0;
    perf.pressure_initializer_accepts = 0;
    perf.pressure_initializer_fallbacks = 0;
    perf.time_step_count = 0;
}

BookPressureStages runBookPressureStagesGPU(
    int lx, int ly, int lz,
    const std::vector<double>& hh,
    const std::vector<double>& pressure,
    const std::vector<double>& rho,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w)
{
    const int n = lx * ly * lz;
    const int lztot = lz + 2;
    const int nmacro = lx * ly * lztot;
    if (static_cast<int>(hh.size()) != n * 15 ||
        static_cast<int>(pressure.size()) != n || static_cast<int>(rho.size()) != n ||
        static_cast<int>(u.size()) != n || static_cast<int>(v.size()) != n ||
        static_cast<int>(w.size()) != n) {
        throw std::invalid_argument("book pressure test input size mismatch");
    }

    upload_lattice_constants();
    std::vector<double> macro_p(nmacro, 0.0), macro_rho(nmacro, 0.0);
    std::vector<double> macro_u(nmacro, 0.0), macro_v(nmacro, 0.0), macro_w(nmacro, 0.0);
    const int offset = lx * ly;
    std::copy(pressure.begin(), pressure.end(), macro_p.begin() + offset);
    std::copy(rho.begin(), rho.end(), macro_rho.begin() + offset);
    std::copy(u.begin(), u.end(), macro_u.begin() + offset);
    std::copy(v.begin(), v.end(), macro_v.begin() + offset);
    std::copy(w.begin(), w.end(), macro_w.begin() + offset);

    double *d_hh = nullptr, *d_tmp = nullptr, *d_p = nullptr, *d_rho = nullptr;
    double *d_u = nullptr, *d_v = nullptr, *d_w = nullptr;
    double *d_ux = nullptr, *d_uy = nullptr, *d_uz = nullptr;
    double *d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
    double *d_wx = nullptr, *d_wy = nullptr, *d_wz = nullptr;
    auto free_all = [&]() {
        cudaFree(d_hh); cudaFree(d_tmp); cudaFree(d_p); cudaFree(d_rho);
        cudaFree(d_u); cudaFree(d_v); cudaFree(d_w);
        cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_wx); cudaFree(d_wy); cudaFree(d_wz);
    };

    try {
        const std::size_t dist_bytes = static_cast<std::size_t>(n) * 15 * sizeof(double);
        const std::size_t cell_bytes = static_cast<std::size_t>(n) * sizeof(double);
        const std::size_t macro_bytes = static_cast<std::size_t>(nmacro) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_hh, dist_bytes));
        CUDA_CHECK(cudaMalloc(&d_tmp, dist_bytes));
        CUDA_CHECK(cudaMalloc(&d_p, macro_bytes));
        CUDA_CHECK(cudaMalloc(&d_rho, macro_bytes));
        CUDA_CHECK(cudaMalloc(&d_u, macro_bytes));
        CUDA_CHECK(cudaMalloc(&d_v, macro_bytes));
        CUDA_CHECK(cudaMalloc(&d_w, macro_bytes));
        CUDA_CHECK(cudaMalloc(&d_ux, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_uy, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_uz, cell_bytes));
        CUDA_CHECK(cudaMalloc(&d_vx, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_vy, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_vz, cell_bytes));
        CUDA_CHECK(cudaMalloc(&d_wx, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_wy, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_wz, cell_bytes));
        CUDA_CHECK(cudaMemcpy(d_hh, hh.data(), dist_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p, macro_p.data(), macro_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rho, macro_rho.data(), macro_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u, macro_u.data(), macro_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, macro_v.data(), macro_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, macro_w.data(), macro_bytes, cudaMemcpyHostToDevice));

        dim3 block(8, 8, 8);
        dim3 grid((lx + block.x - 1) / block.x,
                  (ly + block.y - 1) / block.y,
                  (lz + block.z - 1) / block.z);
        dim3 block2d(16, 16);
        dim3 grid2d((ly + block2d.x - 1) / block2d.x,
                    (lz + block2d.y - 1) / block2d.y);
        gradientKernel<<<grid, block>>>(d_u, d_ux, d_uy, d_uz, lx, ly, lz, lztot, true);
        gradientKernel<<<grid, block>>>(d_v, d_vx, d_vy, d_vz, lx, ly, lz, lztot, true);
        gradientKernel<<<grid, block>>>(d_w, d_wx, d_wy, d_wz, lx, ly, lz, lztot, true);
        CUDA_CHECK(cudaGetLastError());

        BookPressureStages result;
        std::vector<double> ux(n), vy(n), wz(n);
        CUDA_CHECK(cudaMemcpy(ux.data(), d_ux, cell_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vy.data(), d_vy, cell_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(wz.data(), d_wz, cell_bytes, cudaMemcpyDeviceToHost));
        result.divergence.resize(n);
        for (int i = 0; i < n; ++i) result.divergence[i] = ux[i] + vy[i] + wz[i];

        collisionPressureKernel<<<grid, block>>>(
            d_hh, d_p, d_rho, d_ux, d_vy, d_wz, lx, ly, lz, lztot, 1.0);
        CUDA_CHECK(cudaGetLastError());
        result.collision.resize(static_cast<std::size_t>(n) * 15);
        CUDA_CHECK(cudaMemcpy(result.collision.data(), d_hh, dist_bytes, cudaMemcpyDeviceToHost));

        streamKernel<<<grid, block>>>(d_hh, d_tmp, lx, ly, lz);
        CUDA_CHECK(cudaGetLastError());
        result.streamed.resize(static_cast<std::size_t>(n) * 15);
        CUDA_CHECK(cudaMemcpy(result.streamed.data(), d_tmp, dist_bytes, cudaMemcpyDeviceToHost));

        slipBounceBackKernel<<<grid2d, block2d>>>(d_tmp, lx, ly, lz);
        CUDA_CHECK(cudaGetLastError());
        result.bounced.resize(static_cast<std::size_t>(n) * 15);
        CUDA_CHECK(cudaMemcpy(result.bounced.data(), d_tmp, dist_bytes, cudaMemcpyDeviceToHost));

        computePressureKernel<<<grid, block>>>(d_tmp, d_p, lx, ly, lz, lztot);
        CUDA_CHECK(cudaGetLastError());
        result.pressure.resize(n);
        CUDA_CHECK(cudaMemcpy(
            result.pressure.data(), d_p + offset, cell_bytes, cudaMemcpyDeviceToHost));
        free_all();
        return result;
    } catch (...) {
        free_all();
        throw;
    }
}

BookPressureSolution solveBookPressureGPU(
    int lx, int ly, int lz,
    const std::vector<double>& hh,
    const std::vector<double>& pressure,
    const std::vector<double>& rho,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    const std::vector<double>& divergence,
    int max_iterations,
    int check_interval,
    double tolerance)
{
    const int n = lx * ly * lz;
    const int lztot = lz + 2;
    const int nmacro = lx * ly * lztot;
    if (static_cast<int>(hh.size()) != n * 15 || static_cast<int>(pressure.size()) != n ||
        static_cast<int>(rho.size()) != n || static_cast<int>(u.size()) != n ||
        static_cast<int>(v.size()) != n || static_cast<int>(w.size()) != n ||
        static_cast<int>(divergence.size()) != n ||
        max_iterations <= 0 || check_interval <= 0 || tolerance <= 0.0) {
        throw std::invalid_argument("invalid book pressure replay input");
    }
    upload_lattice_constants();
    const int offset = lx * ly;
    std::vector<double> macro_p(nmacro, 0.0), macro_rho(nmacro, 0.0);
    std::vector<double> macro_u(nmacro, 0.0), macro_v(nmacro, 0.0), macro_w(nmacro, 0.0);
    std::copy(pressure.begin(), pressure.end(), macro_p.begin() + offset);
    std::copy(rho.begin(), rho.end(), macro_rho.begin() + offset);
    std::copy(u.begin(), u.end(), macro_u.begin() + offset);
    std::copy(v.begin(), v.end(), macro_v.begin() + offset);
    std::copy(w.begin(), w.end(), macro_w.begin() + offset);

    double *d_ha = nullptr, *d_hb = nullptr, *d_p = nullptr, *d_rho = nullptr;
    double *d_u = nullptr, *d_v = nullptr, *d_w = nullptr;
    double *d_ux = nullptr, *d_uy = nullptr, *d_uz = nullptr;
    double *d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
    double *d_wx = nullptr, *d_wy = nullptr, *d_wz = nullptr;
    auto cleanup = [&]() {
        cudaFree(d_ha); cudaFree(d_hb); cudaFree(d_p); cudaFree(d_rho);
        cudaFree(d_u); cudaFree(d_v); cudaFree(d_w);
        cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
        cudaFree(d_wx); cudaFree(d_wy); cudaFree(d_wz);
    };
    try {
        const std::size_t dist_bytes = static_cast<std::size_t>(n) * 15 * sizeof(double);
        const std::size_t cell_bytes = static_cast<std::size_t>(n) * sizeof(double);
        const std::size_t macro_bytes = static_cast<std::size_t>(nmacro) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_ha, dist_bytes)); CUDA_CHECK(cudaMalloc(&d_hb, dist_bytes));
        CUDA_CHECK(cudaMalloc(&d_p, macro_bytes)); CUDA_CHECK(cudaMalloc(&d_rho, macro_bytes));
        CUDA_CHECK(cudaMalloc(&d_u, macro_bytes)); CUDA_CHECK(cudaMalloc(&d_v, macro_bytes)); CUDA_CHECK(cudaMalloc(&d_w, macro_bytes));
        CUDA_CHECK(cudaMalloc(&d_ux, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_uy, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_uz, cell_bytes));
        CUDA_CHECK(cudaMalloc(&d_vx, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_vy, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_vz, cell_bytes));
        CUDA_CHECK(cudaMalloc(&d_wx, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_wy, cell_bytes)); CUDA_CHECK(cudaMalloc(&d_wz, cell_bytes));
        CUDA_CHECK(cudaMemcpy(d_ha, hh.data(), dist_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_p, macro_p.data(), macro_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rho, macro_rho.data(), macro_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_u, macro_u.data(), macro_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, macro_v.data(), macro_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_w, macro_w.data(), macro_bytes, cudaMemcpyHostToDevice));
        dim3 block(8, 8, 8);
        dim3 grid((lx + block.x - 1) / block.x, (ly + block.y - 1) / block.y, (lz + block.z - 1) / block.z);
        dim3 block2d(16, 16);
        dim3 grid2d((ly + block2d.x - 1) / block2d.x, (lz + block2d.y - 1) / block2d.y);
        CUDA_CHECK(cudaMemcpy(d_ux, divergence.data(), cell_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_vy, 0, cell_bytes));
        CUDA_CHECK(cudaMemset(d_wz, 0, cell_bytes));

        BookPressureSolution solution;
        std::vector<double> previous = pressure;
        std::vector<double> current(n);
        for (int iteration = 1; iteration <= max_iterations; ++iteration) {
            collisionPressureKernel<<<grid, block>>>(d_ha, d_p, d_rho, d_ux, d_vy, d_wz, lx, ly, lz, lztot, 1.0);
            streamKernel<<<grid, block>>>(d_ha, d_hb, lx, ly, lz);
            std::swap(d_ha, d_hb);
            slipBounceBackKernel<<<grid2d, block2d>>>(d_ha, lx, ly, lz);
            computePressureKernel<<<grid, block>>>(d_ha, d_p, lx, ly, lz, lztot);
            CUDA_CHECK(cudaGetLastError());
            if (iteration % check_interval == 0 || iteration == max_iterations) {
                CUDA_CHECK(cudaMemcpy(current.data(), d_p + offset, cell_bytes, cudaMemcpyDeviceToHost));
                double err1 = 0.0, err2 = 0.0;
                for (int i = 0; i < n; ++i) {
                    err1 += std::abs(current[i] - previous[i]);
                    err2 += std::abs(current[i]);
                }
                previous = current;
                solution.relative_error = err2 > 0.0 ? err1 / err2 : 0.0;
                if (solution.relative_error < tolerance) {
                    solution.iterations = iteration;
                    solution.converged = true;
                    break;
                }
            }
            solution.iterations = iteration;
        }
        solution.pressure.resize(n);
        solution.hh.resize(static_cast<std::size_t>(n) * 15);
        CUDA_CHECK(cudaMemcpy(solution.pressure.data(), d_p + offset, cell_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(solution.hh.data(), d_ha, dist_bytes, cudaMemcpyDeviceToHost));
        cleanup();
        return solution;
    } catch (...) {
        cleanup();
        throw;
    }
}
