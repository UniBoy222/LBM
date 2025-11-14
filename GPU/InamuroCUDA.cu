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
// 你可以按你的 Inamuro 定义替换这些值；此处给出标准 D3Q15（rest, 6 axes, 8 diagonals）
__constant__ int    c_ex[InamuroCUDA::Q_];
__constant__ int    c_ey[InamuroCUDA::Q_];
__constant__ int    c_ez[InamuroCUDA::Q_];
__constant__ double c_wE[InamuroCUDA::Q_];   // 用于 ff 的权重（等温 LBM 常用）
__constant__ double c_wF[InamuroCUDA::Q_];   // 用于 gg 的权重（如与 c_wE 不同可分开）

static void upload_lattice_constants()
{
    // Host 侧定义
    static const int ex[InamuroCUDA::Q_] = {
        0,  1,-1, 0, 0, 0, 0,  1, 1, 1, 1,-1,-1,-1,-1
    };
    static const int ey[InamuroCUDA::Q_] = {
        0,  0, 0, 1,-1, 0, 0,  1,-1, 1,-1, 1,-1, 1,-1
    };
    static const int ez[InamuroCUDA::Q_] = {
        0,  0, 0, 0, 0, 1,-1,  1, 1,-1,-1, 1, 1,-1,-1
    };
    // D3Q15 权重：w0=2/9, axes=1/9, diagonals=1/72
    static const double wE[InamuroCUDA::Q_] = {
        2.0/9.0,
        1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,
        1.0/72.0,1.0/72.0,1.0/72.0,1.0/72.0,1.0/72.0,1.0/72.0,1.0/72.0,1.0/72.0
    };
    // 先默认 gg 使用相同权重；若你的 Inamuro 定义不同，可改成独立数组
    static const double wF[InamuroCUDA::Q_] = {
        2.0/9.0,
        1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,1.0/9.0,
        1.0/72.0,1.0/72.0,1.0/72.0,1.0/72.0,1.0/72.0,1.0/72.0,1.0/72.0,1.0/72.0
    };

    CUDA_CHECK(cudaMemcpyToSymbol(c_ex, ex, sizeof(ex)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ey, ey, sizeof(ey)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_ez, ez, sizeof(ez)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wE, wE, sizeof(wE)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_wF, wF, sizeof(wF)));
}

// -------------------- 宏观量 kernel（基线实现） --------------------
__global__ void macroKernel(const double* __restrict__ ff,
                            const double* __restrict__ gg,
                            double* __restrict__ rho,
                            double* __restrict__ fei,
                            double* __restrict__ u,
                            double* __restrict__ v,
                            double* __restrict__ w,
                            int lx, int ly, int lz, int lz_tot)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= lx || y >= ly || z >= lz) return;

    const int cell = (z * ly + y) * lx + x;
    // 累加 ff/gg 的 0 阶与 1 阶矩
    double rho_loc = 0.0;
    double fei_loc = 0.0;
    double ux = 0.0, vy = 0.0, wz = 0.0;

    // 注意：此处求和用最简单的二阶近似（与等温 LBM 一致）
    // 对 ff：ρ = Σ f_k；ρ u = Σ f_k * c_k
    // 对 gg：φ = Σ g_k（若你模型中 φ 的定义是 Σ g_k * 权 wF_k，请替换下一行）
    for (int q = 0; q < InamuroCUDA::Q_; ++q) {
        const int id4 = cell * InamuroCUDA::Q_ + q;
        const double fk = ff[id4];
        const double gk = gg[id4];
        rho_loc += fk;
        fei_loc += gk;  // 如果你的相场需要 wF 权重，这里可改为 fei_loc += gk; 或 fei_loc += gk;（按你的实现一致即可）
        ux += fk * c_ex[q];
        vy += fk * c_ey[q];
        wz += fk * c_ez[q];
    }

    // 速度 = 动量 / ρ（避免除零）
    if (rho_loc != 0.0) {
        ux /= rho_loc;
        vy /= rho_loc;
        wz /= rho_loc;
    }

    // 写回到带 ghost 的宏观场（z+1）
    const int out = InamuroCUDA::idx3D(x, y, z + 1, lx, ly, lz_tot);
    rho[out] = rho_loc;
    fei[out] = fei_loc;
    u[out]   = ux;
    v[out]   = vy;
    w[out]   = wz;
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
    for (int q = 0; q < InamuroCUDA::Q_; ++q) {
        int xn = x - c_ex[q], yn = y - c_ey[q], zn = z - c_ez[q];

        // 周期处理（示例；如果你的 z 有 ghost，请保持 streaming 仅在物理域）
        if (xn < 0)      xn += lx; else if (xn >= lx) xn -= lx;
        if (yn < 0)      yn += ly; else if (yn >= ly) yn -= ly;
        if (zn < 0)      zn += lz; else if (zn >= lz) zn -= lz;

        const int srcCell = (zn * ly + yn) * lx + xn;
        const int srcId4  = srcCell * InamuroCUDA::Q_ + q;
        const int dstId4  = cell    * InamuroCUDA::Q_ + q;

        f_new[dstId4] = f_post[srcId4];
    }
}

// ==================== InamuroCUDA 成员实现 ====================

InamuroCUDA::InamuroCUDA(const Inamuro& cpuSolver)
    : cpu_(cpuSolver)
{
    // 从 CPU 端获取网格尺寸
    int nx=0, ny=0, nz=0;
    cpu_.getGridSize(nx, ny, nz);
    lx_ = nx; ly_ = ny; lz_ = nz;

    // 你的 CPU 代码里，宏观量通常在 z 上带 2 层 ghost
    lz_total_ = lz_ + 2;

    N_cells_ = lx_ * ly_ * lz_;
    N_macro_ = lx_ * ly_ * lz_total_;

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
    const size_t dist_bytes  = static_cast<size_t>(N_cells_) * Q_ * sizeof(double);
    const size_t macro_bytes = static_cast<size_t>(N_macro_) * sizeof(double);
    const size_t cell_bytes  = static_cast<size_t>(N_cells_) * sizeof(double);

    CUDA_CHECK(cudaMalloc(&gpu_.d_ff, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_gg, dist_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_hh, dist_bytes));

    CUDA_CHECK(cudaMalloc(&gpu_.d_rho, macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_fei, macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_u,   macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_v,   macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_w,   macro_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_p,   macro_bytes));

    // 梯度/拉普拉斯（预留，便于后续优化替换）
    CUDA_CHECK(cudaMalloc(&gpu_.d_fei_x, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_fei_y, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_fei_z, cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu_.d_rho_x, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_rho_y, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_rho_z, cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu_.d_u_x, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_u_y, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_u_z, cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu_.d_v_x, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_v_y, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_v_z, cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu_.d_w_x, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_w_y, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_w_z, cell_bytes));

    CUDA_CHECK(cudaMalloc(&gpu_.d_fei_lap, cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_u_lap,   cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_v_lap,   cell_bytes));
    CUDA_CHECK(cudaMalloc(&gpu_.d_w_lap,   cell_bytes));
}

void InamuroCUDA::freeDeviceMemory()
{
    auto S = [](double*& p){ if(p){ cudaFree(p); p=nullptr; } };
    S(gpu_.d_ff); S(gpu_.d_gg); S(gpu_.d_hh);
    S(gpu_.d_rho); S(gpu_.d_fei); S(gpu_.d_u); S(gpu_.d_v); S(gpu_.d_w); S(gpu_.d_p);
    S(gpu_.d_fei_x); S(gpu_.d_fei_y); S(gpu_.d_fei_z);
    S(gpu_.d_rho_x); S(gpu_.d_rho_y); S(gpu_.d_rho_z);
    S(gpu_.d_u_x);   S(gpu_.d_u_y);   S(gpu_.d_u_z);
    S(gpu_.d_v_x);   S(gpu_.d_v_y);   S(gpu_.d_v_z);
    S(gpu_.d_w_x);   S(gpu_.d_w_y);   S(gpu_.d_w_z);
    S(gpu_.d_fei_lap); S(gpu_.d_u_lap); S(gpu_.d_v_lap); S(gpu_.d_w_lap);
}

// 将 CPU 三/四维容器扁平化并上传（需要 Inamuro 声明 friend）
void InamuroCUDA::initFromCPU()
{
    // ------- 分布函数：Q * N_cells_ -------
    std::vector<double> h_ff(N_cells_ * Q_);
    std::vector<double> h_gg(N_cells_ * Q_);
    std::vector<double> h_hh(N_cells_ * Q_);

    for (int z = 0; z < lz_; ++z)
    for (int y = 0; y < ly_; ++y)
    for (int x = 0; x < lx_; ++x) {
        const int cell = (z * ly_ + y) * lx_ + x;
        for (int q = 0; q < Q_; ++q) {
            const int id4 = cell * Q_ + q;
            // NOTE: 依赖 friend 访问 Inamuro 的受保护成员
            h_ff[id4] = cpu_.ff[q][x][y][z];
            h_gg[id4] = cpu_.gg[q][x][y][z];
            h_hh[id4] = cpu_.hh[q][x][y][z];
        }
    }

    CUDA_CHECK(cudaMemcpy(gpu_.d_ff, h_ff.data(), h_ff.size()*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_.d_gg, h_gg.data(), h_gg.size()*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_.d_hh, h_hh.data(), h_hh.size()*sizeof(double), cudaMemcpyHostToDevice));

    // ------- 宏观量：N_macro_（z 含 ghost，z ∈ [0..lz_+1]） -------
    std::vector<double> h_rho(N_macro_), h_fei(N_macro_), h_u(N_macro_), h_v(N_macro_), h_w(N_macro_), h_p(N_macro_);

    for (int z = 0; z < lz_total_; ++z)
    for (int y = 0; y < ly_; ++y)
    for (int x = 0; x < lx_; ++x) {
        const int id3 = idx3D(x, y, z, lx_, ly_, lz_total_);
        h_rho[id3] = cpu_.rho[x][y][z];
        h_fei[id3] = cpu_.fei[x][y][z];
        h_u[id3]   = cpu_.u[x][y][z];
        h_v[id3]   = cpu_.v[x][y][z];
        h_w[id3]   = cpu_.w[x][y][z];
        h_p[id3]   = cpu_.p[x][y][z];
    }

    CUDA_CHECK(cudaMemcpy(gpu_.d_rho, h_rho.data(), h_rho.size()*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_.d_fei, h_fei.data(), h_fei.size()*sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_.d_u,   h_u.data(),   h_u.size()*sizeof(double),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_.d_v,   h_v.data(),   h_v.size()*sizeof(double),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_.d_w,   h_w.data(),   h_w.size()*sizeof(double),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_.d_p,   h_p.data(),   h_p.size()*sizeof(double),   cudaMemcpyHostToDevice));
}

void InamuroCUDA::downloadMacroToCPU(Inamuro& cpuSolver) const
{
    std::vector<double> h_rho(N_macro_), h_fei(N_macro_), h_u(N_macro_), h_v(N_macro_), h_w(N_macro_), h_p(N_macro_);
    CUDA_CHECK(cudaMemcpy(h_rho.data(), gpu_.d_rho, N_macro_*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_fei.data(), gpu_.d_fei, N_macro_*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_u.data(),   gpu_.d_u,   N_macro_*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(),   gpu_.d_v,   N_macro_*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w.data(),   gpu_.d_w,   N_macro_*sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_p.data(),   gpu_.d_p,   N_macro_*sizeof(double), cudaMemcpyDeviceToHost));

    for (int z = 0; z < lz_total_; ++z)
    for (int y = 0; y < ly_; ++y)
    for (int x = 0; x < lx_; ++x) {
        const int id3 = idx3D(x, y, z, lx_, ly_, lz_total_);
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
    // TODO: 在此处调用你的 tiled 梯度/Laplacian + collision 内核
    // 先留空壳以便基线跑通（宏观量由 ff/gg 的当前状态计算）
}

void InamuroCUDA::doStreamFF()
{
    dim3 block(8,8,8);
    dim3 grid((lx_+block.x-1)/block.x, (ly_+block.y-1)/block.y, (lz_+block.z-1)/block.z);
    // 简单用 ff 本地做一次 pull streaming 示例（现实中应使用 ping-pong 或 AA/奇偶步）
    // 这里用 d_hh 作为临时缓冲，避免覆盖
    streamKernel<<<grid, block>>>(gpu_.d_ff, gpu_.d_hh, lx_, ly_, lz_);
    CUDA_CHECK(cudaGetLastError());
    std::swap(gpu_.d_ff, gpu_.d_hh);
}

void InamuroCUDA::doStreamGG()
{
    dim3 block(8,8,8);
    dim3 grid((lx_+block.x-1)/block.x, (ly_+block.y-1)/block.y, (lz_+block.z-1)/block.z);
    streamKernel<<<grid, block>>>(gpu_.d_gg, gpu_.d_hh, lx_, ly_, lz_);
    CUDA_CHECK(cudaGetLastError());
    std::swap(gpu_.d_gg, gpu_.d_hh);
}

void InamuroCUDA::doBoundaryFF()
{
    // TODO: 依据你 CPU 版的边界实现补齐（此处留空，避免与真实 BC 假定冲突）
}

void InamuroCUDA::doBoundaryGG()
{
    // TODO: 同上
}

void InamuroCUDA::doMacro()
{
    dim3 block(8,8,8);
    dim3 grid((lx_+block.x-1)/block.x, (ly_+block.y-1)/block.y, (lz_+block.z-1)/block.z);
    macroKernel<<<grid, block>>>(
        gpu_.d_ff, gpu_.d_gg,
        gpu_.d_rho, gpu_.d_fei,
        gpu_.d_u, gpu_.d_v, gpu_.d_w,
        lx_, ly_, lz_, lz_total_
    );
    CUDA_CHECK(cudaGetLastError());
}

void InamuroCUDA::doPressurePoisson()
{
    // TODO: 后续替换为你的“全 GPU + 并行残差”Poisson 求解器
}

void InamuroCUDA::doCorrectUVWAndHH()
{
    // TODO: 速度修正 + 更新 hh（按你 CPU 版的逻辑一一对应到 GPU）
}

void InamuroCUDA::performTimeStepGPU()
{
    // 1) 碰撞 + 求导/拉普拉斯（先留空壳）
    doCollisionAndGradients();

    // 2) 迁移（示例 pull stream）
    doStreamFF();
    doStreamGG();

    // 3) 边界
    doBoundaryFF();
    doBoundaryGG();

    // 4) 宏观量
    doMacro();

    // 5) 压力 Poisson（后续替换）
    doPressurePoisson();

    // 6) 速度修正 + hh 更新（后续替换）
    doCorrectUVWAndHH();
}