#include "InamuroCUDA.hpp"
#include <cmath>
#include <algorithm>

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
    int lx, int ly, int lz, int lztot)
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
        
        hh[id4] = hh[id4] - (hh[id4] - hequ) / tauh - (c_Ei[k] / 3.0) * div_u;
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
    freeDeviceMemory();
}

void InamuroCUDA::allocateDeviceMemory()
{
    const size_t dist_bytes  = static_cast<size_t>(N_cells) * 15 * sizeof(double);
    const size_t macro_bytes = static_cast<size_t>(N_macro) * sizeof(double);
    const size_t cell_bytes  = static_cast<size_t>(N_cells) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&gpu.d_ff, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_gg, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_hh, dist_bytes));

    CUDA_CHECK(cudaMalloc(&gpu.d_rho, macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_fei, macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_u,   macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_v,   macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_w,   macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu.d_p,   macro_bytes));

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
}

void InamuroCUDA::freeDeviceMemory()
{
    auto S = [](double*& p){ if(p){ cudaFree(p); p=nullptr; } };
    S(gpu.d_ff); S(gpu.d_gg); S(gpu.d_hh);
    S(gpu.d_rho); S(gpu.d_fei); S(gpu.d_u); S(gpu.d_v); S(gpu.d_w); S(gpu.d_p);
    S(gpu.d_fei_x); S(gpu.d_fei_y); S(gpu.d_fei_z);
    S(gpu.d_rho_x); S(gpu.d_rho_y); S(gpu.d_rho_z);
    S(gpu.d_u_x);   S(gpu.d_u_y);   S(gpu.d_u_z);
    S(gpu.d_v_x);   S(gpu.d_v_y);   S(gpu.d_v_z);
    S(gpu.d_w_x);   S(gpu.d_w_y);   S(gpu.d_w_z);
    S(gpu.d_fei_lap); S(gpu.d_u_lap); S(gpu.d_v_lap); S(gpu.d_w_lap);
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
    // 简单用 ff 本地做一次 pull streaming 示例（现实中应使用 ping-pong 或 AA/奇偶步）
    // 这里用 d_hh 作为临时缓冲，避免覆盖
    streamKernel<<<grid, block>>>(gpu.d_ff, gpu.d_hh, lx, ly, lz);
    CUDA_CHECK(cudaGetLastError());
    std::swap(gpu.d_ff, gpu.d_hh);
}

void InamuroCUDA::doStreamGG()
{
    dim3 block(8,8,8);
    dim3 grid((lx+block.x-1)/block.x, (ly+block.y-1)/block.y, (lz+block.z-1)/block.z);
    streamKernel<<<grid, block>>>(gpu.d_gg, gpu.d_hh, lx, ly, lz);
    CUDA_CHECK(cudaGetLastError());
    std::swap(gpu.d_gg, gpu.d_hh);
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
    dim3 block2d(16, 16);
    dim3 grid2d((ly+block2d.x-1)/block2d.x, (lz+block2d.y-1)/block2d.y);

    // 压力泊松迭代求解（简化版本，最多1000次迭代）
    for (int iter = 0; iter < 1000; ++iter) {
        // 1. 压力碰撞
        collisionPressureKernel<<<grid, block>>>(
            gpu.d_hh, gpu.d_p, gpu.d_rho,
            gpu.d_u_x, gpu.d_v_y, gpu.d_w_z,
            lx, ly, lz, lz_total
        );
        CUDA_CHECK(cudaGetLastError());

        // 2. hh迁移
        streamKernel<<<grid, block>>>(gpu.d_hh, gpu.d_ff, lx, ly, lz);
        std::swap(gpu.d_hh, gpu.d_ff);
        
        // 3. hh边界
        slipBounceBackKernel<<<grid2d, block2d>>>(gpu.d_hh, lx, ly, lz);
        CUDA_CHECK(cudaGetLastError());

        // 4. 计算压力
        computePressureKernel<<<grid, block>>>(
            gpu.d_hh, gpu.d_p,
            lx, ly, lz, lz_total
        );
        CUDA_CHECK(cudaGetLastError());

        // TODO: 每100步检查收敛性（需要实现残差计算）
        // 简化版本：固定迭代次数
        if (iter >= 100) break;  // 暂时迭代100次
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
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
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

    // 5) 压力 Poisson
    if (enable_timing) cudaEventRecord(start);
    doPressurePoisson();
    if (enable_timing) {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        perf.total_poisson_time += milliseconds;
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

void InamuroCUDA::printPerformanceMetrics() const
{
    if (perf.time_step_count == 0) {
        std::cout << "No performance data collected yet." << std::endl;
        return;
    }
    
    std::cout << "\n========== GPU Performance Metrics ==========" << std::endl;
    std::cout << "Total time steps: " << perf.time_step_count << std::endl;
    std::cout << "\nAverage time per kernel (ms):" << std::endl;
    std::cout << "  Collision:      " << perf.total_collision_time / perf.time_step_count << std::endl;
    std::cout << "  Stream:         " << perf.total_stream_time / perf.time_step_count << std::endl;
    std::cout << "  Macro:          " << perf.total_macro_time / perf.time_step_count << std::endl;
    std::cout << "  Poisson:        " << perf.total_poisson_time / perf.time_step_count << std::endl;
    
    double total_avg = (perf.total_collision_time + perf.total_stream_time + 
                        perf.total_macro_time + perf.total_poisson_time) / perf.time_step_count;
    std::cout << "  Total per step: " << total_avg << std::endl;
    
    std::cout << "\nKernel time distribution:" << std::endl;
    double total = perf.total_collision_time + perf.total_stream_time + 
                   perf.total_macro_time + perf.total_poisson_time;
    std::cout << "  Collision:      " << (perf.total_collision_time / total * 100) << "%" << std::endl;
    std::cout << "  Stream:         " << (perf.total_stream_time / total * 100) << "%" << std::endl;
    std::cout << "  Macro:          " << (perf.total_macro_time / total * 100) << "%" << std::endl;
    std::cout << "  Poisson:        " << (perf.total_poisson_time / total * 100) << "%" << std::endl;
    std::cout << "=============================================\n" << std::endl;
}

void InamuroCUDA::resetPerformanceMetrics()
{
    perf.total_collision_time = 0.0;
    perf.total_stream_time = 0.0;
    perf.total_macro_time = 0.0;
    perf.total_poisson_time = 0.0;
    perf.time_step_count = 0;
}