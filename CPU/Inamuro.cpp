#include "Inamuro.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

// =========================================
// === Inamuro::Parameters 类实现 ===
// ========================================

Inamuro::Parameters::Parameters()
{
    std::cout << "使用默认Inamuro参数" << std::endl;
    update_gam();
}

Inamuro::Parameters::Parameters(const std::string& filename)
{
    std::cout << "从文件读取Inamuro参数: " << filename << std::endl;

    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "警告: 无法打开参数文件，使用默认参数" << std::endl;
        update_gam();
        return;
    }
    try
    {
        // 跳过求解器参数（前4行：网格尺寸、周期边界、时间步、输出频率）
        int lx_file, ly_file, lz_file, t_max, Nwri;
        file >> lx_file >> ly_file >> lz_file; // 第1行: 网格尺寸（跳过，不使用）
        file >> period;                        // 第2行: 周期边界
        file >> t_max;                         // 第3行: 最大时间步（跳过）
        file >> Nwri;                          // 第4行: 输出频率（跳过）

        // 读取物理参数（从第5行开始）
        file >> rho_L >> rho_G;     // 第5行: 密度
        file >> tauf >> taug;       // 第6行: 松弛时间
        file >> mu_L >> mu_G;       // 第7行: 粘度
        file >> k_f >> k_g;         // 第8行: 系数
        file >> T;                  // 第9行: 温度
        file >> a;                  // 第10行: EOS参数a
        file >> b;                  // 第11行: EOS参数b
        file >> fei_max >> fei_min; // 第12行: 相场边界
        file >> fei_L >> fei_G;     // 第13行: 相场值
        file >> DD;                 // 第14行: 液滴直径

        file.close();
        std::cout << "Inamuro参数读取成功!" << std::endl;
        update_gam();
    }
    catch (const std::exception& e)
    {
        std::cerr << "错误: Inamuro参数读取失败 - " << e.what() << std::endl;
        file.close();
        update_gam();
    }
}

void Inamuro::Parameters::update_gam()
{
    gam_l = (tauf - 0.5) / 3.0;
    gam_g = (taug - 0.5) / 3.0;
}

void Inamuro::Parameters::print() const
{
    std::cout << "\n=== Inamuro 物理参数 ===" << std::endl;
    std::cout << "流体物性:" << std::endl;
    std::cout << "  密度: rho_L=" << rho_L << ", rho_G=" << rho_G << std::endl;
    std::cout << "  粘度: mu_L=" << mu_L << ", mu_G=" << mu_G << std::endl;
    std::cout << "LBM参数:" << std::endl;
    std::cout << "  松弛: tauf=" << tauf << ", taug=" << taug << std::endl;
    std::cout << "  运动粘性: gam_l=" << gam_l << ", gam_g=" << gam_g
              << " (由松弛时间计算)" << std::endl;
    std::cout << "相场参数:" << std::endl;
    std::cout << "  相场值: fei_L=" << fei_L << ", fei_G=" << fei_G << std::endl;
    std::cout << "  边界值: fei_max=" << fei_max << ", fei_min=" << fei_min << std::endl;
    std::cout << "  表面张力系数: k_f=" << k_f << ", k_g=" << k_g << std::endl;
    std::cout << "物理几何:" << std::endl;
    std::cout << "  液滴直径: DD=" << DD << std::endl;
    std::cout << "状态方程:" << std::endl;
    std::cout << "  EOS参数: T=" << T << ", a=" << a << ", b=" << b << std::endl;
    std::cout << "========================" << std::endl;
}

// =========================================
// === Inamuro 类实现 ===
// =========================================

Inamuro::Inamuro(int nx, int ny, int nz) : LBMBase(), lx(nx), ly(ny), lz(nz), params()
{
    initializeArrays();
    initializeDroplets();
    std::cout << "Inamuro算法核心初始化完成（默认参数）: " << lx << "x" << ly << "x" << lz << std::endl;
}

Inamuro::Inamuro(const std::string& filename) : LBMBase(), params(filename)
{
    std::cout << "从文件读取所有参数: " << filename << std::endl;

    // 第一步：从文件读取网格尺寸
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "错误: 无法打开参数文件 " << filename << "，使用默认网格尺寸" << std::endl;
        lx = 48;
        ly = 96;
        lz = 128;
    }
    else
    {
        file >> lx >> ly >> lz; // 读取第一行的网格尺寸
        file.close();
        std::cout << "从文件读取网格尺寸: " << lx << "x" << ly << "x" << lz << std::endl;
    }

    // 第二步：初始化数组
    initializeArrays();
    // 第三步：显示加载的参数
    params.print();       // 显示加载的参数
    initializeDroplets(); // 初始化两个液滴

    std::cout << "Inamuro算法核心初始化完成（全部从文件）: " << lx << "x" << ly << "x" << lz << std::endl;
}

void Inamuro::collision()
{
    // 预计算常量
    const double tmp1 = (params.mu_L - params.mu_G) / (params.rho_L - params.rho_G);

    // ---- 计算梯度 & 拉普拉斯 ----
    firstord(u, u_x, u_y, u_z);         // ∇u
    firstord(v, v_x, v_y, v_z);         // ∇v
    firstord(w, w_x, w_y, w_z);         // ∇w
    firstord(rho, rho_x, rho_y, rho_z); // ∇ρ
    firstord(fei, fei_x, fei_y, fei_z); // ∇φ
    secondord(u, u_lap);                // ∇²u
    secondord(v, v_lap);                // ∇²v
    secondord(w, w_lap);                // ∇²w
    secondord(fei, fei_lap);            // ∇²φ

    static bool is_first_step = true;
    // ---- 主循环 ----
    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                // 当前格点的宏观量（考虑虚拟层偏移）
                int z_macro = z + 1;

                // 计算当前格点的粘度（线性插值）
                double mu = (rho[x][y][z_macro] - params.rho_G) * tmp1 + params.mu_G;

                // 梯度模长的平方
                double sum_fei = fei_x[x][y][z] * fei_x[x][y][z] +
                                 fei_y[x][y][z] * fei_y[x][y][z] +
                                 fei_z[x][y][z] * fei_z[x][y][z];

                double sum_rho = rho_x[x][y][z] * rho_x[x][y][z] +
                                 rho_y[x][y][z] * rho_y[x][y][z] +
                                 rho_z[x][y][z] * rho_z[x][y][z];

                double fei2sum = fei_lap[x][y][z]; // 相场拉普拉斯
                double usq = u[x][y][z_macro] * u[x][y][z_macro] +
                             v[x][y][z_macro] * v[x][y][z_macro] +
                             w[x][y][z_macro] * w[x][y][z_macro];

                // 状态方程计算压力
                double p0;
                EOS(fei[x][y][z_macro], p0);

                // --- 方向循环 ---
                std::array<double, D3Q15::Q> fequ, gequ;
                for (int i = 0; i < D3Q15::Q; ++i)
                {
                    // 速度与离散速度的点积
                    double un = D3Q15::uc[i] * u[x][y][z_macro] +
                                D3Q15::vc[i] * v[x][y][z_macro] +
                                D3Q15::wc[i] * w[x][y][z_macro];

                    // 相场梯度与离散速度的点积
                    double fei_ei = D3Q15::uc[i] * fei_x[x][y][z] +
                                    D3Q15::vc[i] * fei_y[x][y][z] +
                                    D3Q15::wc[i] * fei_z[x][y][z];

                    // 密度梯度与离散速度的点积
                    double rho_ei = D3Q15::uc[i] * rho_x[x][y][z] +
                                    D3Q15::vc[i] * rho_y[x][y][z] +
                                    D3Q15::wc[i] * rho_z[x][y][z];

                    // 高阶项计算
                    double ci_sq = D3Q15::uc[i] * D3Q15::uc[i] +
                                   D3Q15::vc[i] * D3Q15::vc[i] +
                                   D3Q15::wc[i] * D3Q15::wc[i];

                    double Gfei = 4.5 * (fei_ei * fei_ei) - 1.5 * sum_fei * ci_sq;
                    double Grho = 4.5 * (rho_ei * rho_ei) - 1.5 * sum_rho * ci_sq;

                    // 相场分布函数平衡态 (Allen-Cahn方程)
                    fequ[i] = D3Q15::Hi[i] * fei[x][y][z_macro] +
                              D3Q15::Fi[i] * (p0 - params.k_f * fei[x][y][z_macro] * fei2sum - params.k_f * sum_fei / 6.0) +
                              3.0 * D3Q15::Ei[i] * fei[x][y][z_macro] * un +
                              D3Q15::Ei[i] * params.k_f * Gfei;

                    // 动量分布函数平衡态 (Navier-Stokes方程)
                    double velPart = 1.0 + 3.0 * un - 1.5 * usq + 4.5 * un * un +
                                     1.5 * (params.taug - 0.5) * 2.0 * (D3Q15::uc[i] * D3Q15::uc[i] * u_x[x][y][z] + D3Q15::uc[i] * D3Q15::vc[i] * u_y[x][y][z] + D3Q15::uc[i] * D3Q15::wc[i] * u_z[x][y][z] + D3Q15::vc[i] * D3Q15::uc[i] * v_x[x][y][z] + D3Q15::vc[i] * D3Q15::vc[i] * v_y[x][y][z] + D3Q15::vc[i] * D3Q15::wc[i] * v_z[x][y][z] + D3Q15::wc[i] * D3Q15::uc[i] * w_x[x][y][z] + D3Q15::wc[i] * D3Q15::vc[i] * w_y[x][y][z] + D3Q15::wc[i] * D3Q15::wc[i] * w_z[x][y][z]);

                    gequ[i] = D3Q15::Ei[i] * velPart +
                              D3Q15::Ei[i] * params.k_g / rho[x][y][z_macro] * Grho -
                              2.0 / 3.0 * D3Q15::Fi[i] * params.k_g / rho[x][y][z_macro] * sum_rho;
                }

                if (is_first_step)
                {
                    // 第一次：初始化
                    for (int k = 0; k < D3Q15::Q; ++k)
                    {
                        ff[k][x][y][z] = fequ[k];
                        gg[k][x][y][z] = gequ[k];
                    }
                }
                else
                {
                    // 以后：正常碰撞
                    for (int k = 0; k < D3Q15::Q; ++k)
                    {
                        ff[k][x][y][z] = ff[k][x][y][z] - (ff[k][x][y][z] - fequ[k]) / params.tauf;
                        gg[k][x][y][z] = gg[k][x][y][z] - (gg[k][x][y][z] - gequ[k]) / params.taug +
                                         3.0 * D3Q15::Ei[k] / rho[x][y][z_macro] * mu *
                                             (D3Q15::uc[k] * u_lap[x][y][z] +
                                              D3Q15::vc[k] * v_lap[x][y][z] +
                                              D3Q15::wc[k] * w_lap[x][y][z]);
                    }
                }
            }
        }
    }
    is_first_step = false;
}

void Inamuro::stream(Vector4D& dist) // distribution 分布
{
    static Vector4D temp;
    resize4D(temp, D3Q15::Q, lx, ly, lz, 0.0);

    // 计算周期边界下的新坐标
    auto calc_new_coord = [](int current, int step, int max_size) -> int
    {
        return (current + step + max_size) % max_size;
    };

    // 执行单个粒子的迁移
    auto migrate_particle = [&](int dir, int x, int y, int z) -> void
    {
        int new_x = calc_new_coord(x, D3Q15::ex[dir], lx);
        int new_y = calc_new_coord(y, D3Q15::ey[dir], ly);
        int new_z = calc_new_coord(z, D3Q15::ez[dir], lz);
        temp[dir][new_x][new_y][new_z] = dist[dir][x][y][z];
    };

    // 迁移所有粒子
    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                for (int dir = 0; dir < D3Q15::Q; ++dir)
                {
                    migrate_particle(dir, x, y, z);
                }
            }
        }
    }
    dist = std::move(temp); // 移动赋值
}

void Inamuro::applyBoundaryConditions(Vector4D& dist)
{
    slipBounceBack(dist);
}

void Inamuro::getMacro()
{
    const double fei_avg = (params.fei_L + params.fei_G) / 2.0;

    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {

                // 1. 计算相场 φ = Σ ff_i
                double fei_local = 0.0;
                for (int i = 0; i < D3Q15::Q; ++i)
                {
                    fei_local += ff[i][x][y][z];
                }
                fei[x][y][z + 1] = fei_local; // 修复：直接使用z+1

                // 2. 根据φ计算密度 (三段式 + 正弦过渡)
                double density;
                if (fei_local <= params.fei_G)
                {
                    density = params.rho_G;
                }
                else if (fei_local >= params.fei_L)
                {
                    density = params.rho_L;
                }
                else
                {
                    double arg = (fei_local - fei_avg) / (params.fei_L - params.fei_G) * M_PI;
                    density = (params.rho_L - params.rho_G) / 2.0 * (std::sin(arg) + 1.0) + params.rho_G;
                }
                rho[x][y][z + 1] = density; // 有虚拟层，使用z+1

                // 3. 计算速度 (初始化为零)
                u[x][y][z + 1] = 0.0; // 有虚拟层，使用z+1
                v[x][y][z + 1] = 0.0; // 有虚拟层，使用z+1
                w[x][y][z + 1] = 0.0; // 有虚拟层，使用z+1

                // 4. 按 gg 求宏观速度
                for (int i = 1; i < D3Q15::Q; ++i)
                { // 从1开始，跳过静止方向
                    double g_val = gg[i][x][y][z];
                    u[x][y][z + 1] += D3Q15::uc[i] * g_val; // 有虚拟层，使用z+1
                    v[x][y][z + 1] += D3Q15::vc[i] * g_val; // 有虚拟层，使用z+1
                    w[x][y][z + 1] += D3Q15::wc[i] * g_val; // 有虚拟层，使用z+1
                }
            }
        }
    }
}
void Inamuro::performTimeStep()
{
    // 1. 基础LBM步骤：推进 ff / gg，并从分布函数恢复宏观量
    collision(); // 碰撞
    stream(ff);
    stream(gg);                  // 流动
    applyBoundaryConditions(ff); // x边界反弹
    applyBoundaryConditions(gg);
    getMacro(); // 计算宏观量（对应getuv()）

    // 2. 压力泊松方程迭代求解
    solvePressurePoisson();

    // 3. 速度修正和下一步准备
    correct_uvw();
    update_hh();
}

void Inamuro::writeResults(int timeStep)
{
    if (!output_config.output_dir.empty())
    {
        std::filesystem::create_directories(output_config.output_dir);
    }

    if (output_config.enable_tecplot)
    {
        writeTecplotBinary(timeStep);
    }

    if (output_config.enable_debug_field_csv)
    {
        writeDebugOutput(timeStep);
    }
}

std::string Inamuro::getAlgorithmName() const
{
    return "Inamuro";
}

// === 数据访问接口 ===

void Inamuro::getGridSize(int& nx, int& ny, int& nz) const
{
    nx = lx;
    ny = ly;
    nz = lz;
}

void Inamuro::setOutputConfig(const OutputConfig& config)
{
    output_config = config;
}

const Inamuro::OutputConfig& Inamuro::getOutputConfig() const
{
    return output_config;
}

void Inamuro::loadNeuralNetwork(const std::string& model_path)
{
    try
    {
        if (use_gpu_inference && !torch::cuda::is_available())
        {
            throw std::runtime_error("配置要求 GPU 推理，但当前 libtorch CUDA 不可用");
        }

        model = torch::jit::load(model_path);
        model.eval();
        if (use_gpu_inference)
        {
            model.to(torch::kCUDA);
        }
        else
        {
            model.to(torch::kCPU);
        }
        model_loaded = true;
        std::cout << "TorchScript模型加载成功: " << model_path
                  << " | 推理设备: " << (use_gpu_inference ? "GPU" : "CPU")
                  << std::endl;
    }
    catch (const std::exception& e)
    {
        model_loaded = false;
        use_neural_network = false;
        std::cerr << "警告: TorchScript模型加载失败，已自动禁用NN加速: "
                  << e.what() << std::endl;
    }
    catch (const c10::Error& e)
    {
        model_loaded = false;
        use_neural_network = false;
        std::cerr << "警告: TorchScript模型加载失败，已自动禁用NN加速: "
                  << e.what_without_backtrace() << std::endl;
    }
}

void Inamuro::setUseNeuralNetwork(bool enable)
{
    use_neural_network = enable;
}

void Inamuro::setUseGpuInference(bool enable)
{
    use_gpu_inference = enable && torch::cuda::is_available();
}

bool Inamuro::isModelLoaded() const
{
    return model_loaded;
}

bool Inamuro::isUsingNeuralNetwork() const
{
    return use_neural_network;
}

bool Inamuro::isUsingGpuInference() const
{
    return use_gpu_inference;
}

const Inamuro::PressureSolveDiagnostics& Inamuro::getLastPressureSolveDiagnostics() const
{
    return last_pressure_solve_diagnostics;
}

Inamuro::DiagnosticsSnapshot Inamuro::collectDiagnostics() const
{
    DiagnosticsSnapshot snapshot;

    double rho_sum = 0.0;
    double fei_sum = 0.0;
    double u_sum = 0.0;
    double v_sum = 0.0;
    double w_sum = 0.0;
    double vel_mag_sum = 0.0;
    double p_sum = 0.0;

    std::size_t rho_valid = 0;
    std::size_t fei_valid = 0;
    std::size_t u_valid = 0;
    std::size_t v_valid = 0;
    std::size_t w_valid = 0;
    std::size_t vel_mag_valid = 0;
    std::size_t p_valid = 0;

    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 1; z <= lz; ++z)
            {
                const double rho_value = rho[x][y][z];
                const double fei_value = fei[x][y][z];
                const double u_value = u[x][y][z];
                const double v_value = v[x][y][z];
                const double w_value = w[x][y][z];
                const double p_value = p[x][y][z];
                const double vel_mag = std::sqrt(u_value * u_value + v_value * v_value + w_value * w_value);

                updateFieldStatistics(snapshot.rho_stats, rho_value);
                if (std::isfinite(rho_value))
                {
                    rho_sum += rho_value;
                    ++rho_valid;
                }

                updateFieldStatistics(snapshot.fei_stats, fei_value);
                if (std::isfinite(fei_value))
                {
                    fei_sum += fei_value;
                    ++fei_valid;
                }

                updateFieldStatistics(snapshot.u_stats, u_value);
                if (std::isfinite(u_value))
                {
                    u_sum += u_value;
                    ++u_valid;
                }

                updateFieldStatistics(snapshot.v_stats, v_value);
                if (std::isfinite(v_value))
                {
                    v_sum += v_value;
                    ++v_valid;
                }

                updateFieldStatistics(snapshot.w_stats, w_value);
                if (std::isfinite(w_value))
                {
                    w_sum += w_value;
                    ++w_valid;
                }

                updateFieldStatistics(snapshot.velocity_magnitude_stats, vel_mag);
                if (std::isfinite(vel_mag))
                {
                    vel_mag_sum += vel_mag;
                    ++vel_mag_valid;
                }

                updateFieldStatistics(snapshot.p_stats, p_value);
                if (std::isfinite(p_value))
                {
                    p_sum += p_value;
                    ++p_valid;
                }
            }
        }
    }

    finalizeFieldStatistics(snapshot.rho_stats, rho_valid, rho_sum);
    finalizeFieldStatistics(snapshot.fei_stats, fei_valid, fei_sum);
    finalizeFieldStatistics(snapshot.u_stats, u_valid, u_sum);
    finalizeFieldStatistics(snapshot.v_stats, v_valid, v_sum);
    finalizeFieldStatistics(snapshot.w_stats, w_valid, w_sum);
    finalizeFieldStatistics(snapshot.velocity_magnitude_stats, vel_mag_valid, vel_mag_sum);
    finalizeFieldStatistics(snapshot.p_stats, p_valid, p_sum);

    return snapshot;
}

Inamuro::Vector3D Inamuro::predictPressureDelta(const Vector3D& pressure_prev)
{
    Vector3D delta_p;
    resize3D(delta_p, lx, ly, lz, 0.0);

    if (!model_loaded)
    {
        return delta_p;
    }

    std::vector<float> input_data(static_cast<std::size_t>(6) * lx * ly * lz, 0.0f);

    auto write_channel = [&](int channel, int x, int y, int z, double value)
    {
        const std::size_t spatial_idx =
            static_cast<std::size_t>(x) * ly * lz +
            static_cast<std::size_t>(y) * lz +
            static_cast<std::size_t>(z);
        const std::size_t idx = static_cast<std::size_t>(channel) * lx * ly * lz + spatial_idx;
        input_data[idx] = static_cast<float>((value - input_means[channel]) / input_stds[channel]);
    };

    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                write_channel(0, x, y, z, rho[x][y][z + 1]);
                write_channel(1, x, y, z, u[x][y][z + 1]);
                write_channel(2, x, y, z, v[x][y][z + 1]);
                write_channel(3, x, y, z, w[x][y][z + 1]);
                write_channel(4, x, y, z, fei[x][y][z + 1]);
                write_channel(5, x, y, z, pressure_prev[x][y][z]);
            }
        }
    }

    torch::NoGradGuard no_grad;
    auto input_tensor = torch::from_blob(input_data.data(), {1, 6, lx, ly, lz}, torch::kFloat32).clone();
    if (use_gpu_inference)
    {
        input_tensor = input_tensor.to(torch::kCUDA);
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input_tensor);

    auto output_tensor = model.forward(inputs).toTensor();
    if (use_gpu_inference)
    {
        if (!output_tensor.is_cuda())
        {
            throw std::runtime_error("配置要求 GPU 推理，但模型输出不在 CUDA 上");
        }
        if (!input_tensor.is_cuda())
        {
            throw std::runtime_error("配置要求 GPU 推理，但输入张量不在 CUDA 上");
        }
        std::cout << "[NNInference] input_device=" << input_tensor.device()
                  << " output_device=" << output_tensor.device() << std::endl;
        output_tensor = output_tensor.to(torch::kCPU);
    }
    auto output = output_tensor.contiguous();
    const float* output_ptr = output.data_ptr<float>();

    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                const std::size_t idx =
                    static_cast<std::size_t>(x) * ly * lz +
                    static_cast<std::size_t>(y) * lz +
                    static_cast<std::size_t>(z);
                delta_p[x][y][z] = static_cast<double>(output_ptr[idx]) * output_std + output_mean;
            }
        }
    }

    return delta_p;
}

void Inamuro::initializePressureFromPrediction(const Vector3D& predicted_pressure)
{
    // 最小恢复版先使用最直接的初始化方式：
    // 1. 将预测压力写入宏观压力数组 p
    // 2. 用 hh = Ei * p 初始化压力分布函数
    // 这样可以先验证 NN 初值是否有效，再单独检查是否需要更严格的一致性重建。
    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                p[x][y][z + 1] = predicted_pressure[x][y][z];
                for (int i = 0; i < D3Q15::Q; ++i)
                {
                    hh[i][x][y][z] = D3Q15::Ei[i] * p[x][y][z + 1];
                }
            }
        }
    }
}

// === 算法实现方法 ===

void Inamuro::initializeArrays()
{
    // 4D数组初始化（分布函数）
    resize4D(ff, D3Q15::Q, lx, ly, lz);
    resize4D(gg, D3Q15::Q, lx, ly, lz);
    resize4D(hh, D3Q15::Q, lx, ly, lz);

    // 3D数组初始化（宏观量，z方向+2层）
    resize3D(rho, lx, ly, lz + 2, params.rho_G);
    resize3D(fei, lx, ly, lz + 2, params.fei_min);
    resize3D(u, lx, ly, lz + 2);
    resize3D(v, lx, ly, lz + 2);
    resize3D(w, lx, ly, lz + 2);
    resize3D(p, lx, ly, lz + 2);

    // 梯度场初始化
    resize3D(fei_x, lx, ly, lz);
    resize3D(fei_y, lx, ly, lz);
    resize3D(fei_z, lx, ly, lz);
    resize3D(rho_x, lx, ly, lz);
    resize3D(rho_y, lx, ly, lz);
    resize3D(rho_z, lx, ly, lz);
    resize3D(u_x, lx, ly, lz);
    resize3D(u_y, lx, ly, lz);
    resize3D(u_z, lx, ly, lz);
    resize3D(v_x, lx, ly, lz);
    resize3D(v_y, lx, ly, lz);
    resize3D(v_z, lx, ly, lz);
    resize3D(w_x, lx, ly, lz);
    resize3D(w_y, lx, ly, lz);
    resize3D(w_z, lx, ly, lz);
    resize3D(fei_lap, lx, ly, lz);

    // 拉普拉斯场初始化
    resize3D(u_lap, lx, ly, lz);
    resize3D(v_lap, lx, ly, lz);
    resize3D(w_lap, lx, ly, lz);
}

void Inamuro::initializeDroplets()
{
    double radius = params.DD / 2.0;   // 液滴半径
    double cy = ly / 2.0;              // Y中心
    double cz1 = lz / 2.0 + params.DD; // 液滴1的Z位置
    double cz2 = lz / 2.0 - params.DD; // 液滴2的Z位置
    double velocity = 0.035;           // 初始速度

    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 1; z < lz + 1; ++z) // 不初始化虚拟层 1~lz
            {
                double dx = x, dy = y - cy;
                double dz1 = (z - 1) - cz1, dz2 = (z - 1) - cz2;
                double r1 = std::sqrt(dx * dx + dy * dy + dz1 * dz1);
                double r2 = std::sqrt(dx * dx + dy * dy + dz2 * dz2);

                // 液滴1（向上运动）
                if (r1 < radius)
                {
                    fei[x][y][z] = params.fei_max; // 液相饱和值
                    rho[x][y][z] = params.rho_L;   // 液相密度
                    w[x][y][z] = -velocity;        // 向上速度
                }

                // 液滴2（向下运动）
                if (r2 < radius)
                {
                    fei[x][y][z] = params.fei_max; // 液相饱和值
                    rho[x][y][z] = params.rho_L;   // 液相密度
                    w[x][y][z] = velocity;         // 向下速度
                }
            }
        }
    }
}

void Inamuro::slipBounceBack(Vector4D& dist)
{
    // 使用std::pair来存储正方向和负方向
    const std::vector<std::pair<int, int>> x_bounce_rules = {
        {1, 4}, {7, 8}, {9, 14}, {10, 13}, {12, 11}};
    // 使用lambda函数来实现滑移反弹
    auto bounce_x_wall = [&](int wall_x, bool is_left)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                // 使用for循环来遍历正方向和负方向
                for (const auto& rule : x_bounce_rules)
                {
                    int pos_dir = rule.first;  // 正方向
                    int neg_dir = rule.second; // 负方向

                    if (is_left)
                    {
                        dist[pos_dir][wall_x][y][z] = dist[neg_dir][wall_x][y][z];
                    }
                    else
                    {
                        dist[neg_dir][wall_x][y][z] = dist[pos_dir][wall_x][y][z];
                    }
                }
            }
        }
    };

    // 使用bounce_x_wall函数来实现滑移反弹
    bounce_x_wall(0, true);
    bounce_x_wall(lx - 1, false);
}
void Inamuro::solvePressurePoisson()
{
    // 压力误差缓存（对应pp数组）
    static Vector3D pressure_prev;
    static bool first_call = true;
    ++pressure_solve_call_count;

    // 第一次调用时初始化压力缓存
    if (first_call)
    {
        resize3D(pressure_prev, lx, ly, lz, 0.0);
        first_call = false;
    }

    const auto pressure_solve_start = std::chrono::steady_clock::now();

    // v3 训练数据从 Step 11 开始，前 10 步属于极端动态阶段，直接禁用 NN。
    // 从第 11 次压力求解开始，再按“上一步泊松迭代次数”决定是否启用 NN。
    const bool passed_warmup_stage = pressure_solve_call_count > NN_WARMUP_STEPS;
    const int prev_iteration_count = last_iteration_count;
    const bool nn_temporarily_disabled = nn_disable_countdown > 0;
    const bool use_nn_this_step =
        passed_warmup_stage &&
        use_neural_network &&
        model_loaded &&
        !nn_temporarily_disabled &&
        (prev_iteration_count > NN_ITER_THRESHOLD_STRICT);

    last_pressure_solve_diagnostics.pressure_solve_call_count = pressure_solve_call_count;
    last_pressure_solve_diagnostics.passed_warmup_stage = passed_warmup_stage;
    last_pressure_solve_diagnostics.nn_used = use_nn_this_step;
    last_pressure_solve_diagnostics.prev_iteration_count = prev_iteration_count;
    last_pressure_solve_diagnostics.iteration_count = 0;
    last_pressure_solve_diagnostics.hit_max_iterations = false;
    last_pressure_solve_diagnostics.pressure_solve_time_ms = 0.0;
    last_pressure_solve_diagnostics.nn_prediction_time_ms = 0.0;

    std::cout << "[PressureSolve] 调用=" << pressure_solve_call_count
              << " 上一步迭代数=" << prev_iteration_count
              << " 已过前10步=" << (passed_warmup_stage ? "是" : "否")
              << " 冷却中=" << (nn_temporarily_disabled ? "是" : "否")
              << " 使用NN=" << (use_nn_this_step ? "是" : "否") << std::endl;

    if (use_nn_this_step)
    {
        const auto nn_start = std::chrono::steady_clock::now();
        const Vector3D delta_p = predictPressureDelta(pressure_prev);
        Vector3D predicted_pressure;
        resize3D(predicted_pressure, lx, ly, lz, 0.0);

        for (int x = 0; x < lx; ++x)
        {
            for (int y = 0; y < ly; ++y)
            {
                for (int z = 0; z < lz; ++z)
                {
                    predicted_pressure[x][y][z] = pressure_prev[x][y][z] + delta_p[x][y][z];
                }
            }
        }

        initializePressureFromPrediction(predicted_pressure);
        const auto nn_end = std::chrono::steady_clock::now();
        last_pressure_solve_diagnostics.nn_prediction_time_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(nn_end - nn_start).count() / 1000.0;
    }

    // 压力泊松方程迭代求解
    int iteration_count = 0;
    const int convergence_check_interval = use_nn_this_step ? 50 : 100;
    for (int i_iter = 1; i_iter <= 1000; ++i_iter)
    {
        iteration_count = i_iter;
        // 压力修正（对应correction()）
        collision_p();

        // 压力分布函数迁移
        stream(hh);
        applyBoundaryConditions(hh);

        // 计算压力场（对应getp()）
        getp();

        // 每固定步数检查收敛性
        if (i_iter % convergence_check_interval == 0)
        {
            double error = getError(pressure_prev);
            if (error < 0.001)
            {
                // 只在输出时间步显示收敛信息，避免过多输出
                // std::cout << "压力迭代收敛！" << std::endl;
                break;
            }
        }
    }

    last_iteration_count = iteration_count;
    if (nn_disable_countdown > 0)
    {
        --nn_disable_countdown;
    }
    if (use_nn_this_step && NN_DISABLE_COOLDOWN_STEPS > 0 && iteration_count >= prev_iteration_count)
    {
        nn_disable_countdown = NN_DISABLE_COOLDOWN_STEPS;
    }
    last_pressure_solve_diagnostics.iteration_count = iteration_count;
    last_pressure_solve_diagnostics.hit_max_iterations = (iteration_count >= 1000);
    const auto pressure_solve_end = std::chrono::steady_clock::now();
    last_pressure_solve_diagnostics.pressure_solve_time_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(pressure_solve_end - pressure_solve_start).count() / 1000.0;
}

double Inamuro::getError(Vector3D& pressure_prev)
{
    double err1 = 0.0, err2 = 0.0;

    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                double p_current = p[x][y][z + 1]; // p有虚拟层
                double p_prev = pressure_prev[x][y][z];

                err1 += std::abs(p_current - p_prev);
                err2 += std::abs(p_current);

                // 更新压力缓存
                pressure_prev[x][y][z] = p_current;
            }
        }
    }

    return (err2 > 0.0) ? (err1 / err2) : 0.0;
}

void Inamuro::collision_p()
{
    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                // τ_h：Eq.(6.43) - 注意z+1是因为虚拟层偏移
                double tauh = 1.0 / rho[x][y][z + 1] + 0.5;

                // 碰撞：Eq.(6.42)
                for (int k = 0; k < D3Q15::Q; ++k)
                {
                    double hequ = D3Q15::Ei[k] * p[x][y][z + 1]; // z+1虚拟层偏移

                    hh[k][x][y][z] = hh[k][x][y][z] -
                                     (hh[k][x][y][z] - hequ) / tauh -
                                     (D3Q15::Ei[k] / 3.0) * (u_x[x][y][z] + v_y[x][y][z] + w_z[x][y][z]);
                }
            }
        }
    }
}
void Inamuro::getp()
{
    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                p[x][y][z + 1] = 0.0; // 初始化为0
                for (int i = 0; i < D3Q15::Q; ++i)
                    p[x][y][z + 1] += hh[i][x][y][z]; // 直接累加
            }
        }
    }
}
void Inamuro::correct_uvw()
{
    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                // X方向：对称边界
                int x_e = (x == lx - 1) ? lx - 2 : x + 1;
                int x_w = (x == 0) ? 1 : x - 1;

                // Y方向：周期边界
                int y_n = (y == ly - 1) ? 0 : y + 1;
                int y_s = (y == 0) ? ly - 1 : y - 1;

                // Z方向：周期边界
                int z_n = (z == lz - 1) ? 0 : z + 1;
                int z_s = (z == 0) ? lz - 1 : z - 1;

                // 压力梯度校正速度（所有数组都有虚拟层，用z+1）
                double inv_2rho = 1.0 / (2.0 * rho[x][y][z + 1]);

                u[x][y][z + 1] -= (p[x_e][y][z + 1] - p[x_w][y][z + 1]) * inv_2rho;
                v[x][y][z + 1] -= (p[x][y_n][z + 1] - p[x][y_s][z + 1]) * inv_2rho;
                w[x][y][z + 1] -= (p[x][y][z_n + 1] - p[x][y][z_s + 1]) * inv_2rho;
            }
        }
    }
}

void Inamuro::update_hh()
{
    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                for (int i = 0; i < D3Q15::Q; ++i)
                {
                    hh[i][x][y][z] = p[x][y][z + 1] * D3Q15::Ei[i]; // p有虚拟层，用z+1
                }
            }
        }
    }
}
// === 数值方法辅助函数实现 ===

void Inamuro::EOS(double fei, double& p)
{
    // (6.10): p = φT/(1-bφ) - aφ²
    p = fei * params.T / (1.0 - params.b * fei) - params.a * fei * fei;
}

// === 辅助函数：获取变量值（处理不同的索引模式） ===

double Inamuro::getValue(const Vector3D& var, int x, int y, int z) const
{
    // 对于有虚拟层的宏观量 (z方向 +2)
    if (&var == &u || &var == &v || &var == &w || &var == &rho || &var == &fei || &var == &p)
    {
        return var[x][y][z + 1]; // z方向索引偏移
    }
    // 对于没有虚拟层的梯度场
    else
    {
        return var[x][y][z];
    }
}

void Inamuro::writeTecplotBinary(int timeStep)
{
    // 使用C++字符串流和现代文件操作
    std::ostringstream filename;
    filename << output_config.output_dir << "3D" << std::setfill('0') << std::setw(9) << timeStep << ".plt";

    std::ofstream file(filename.str(), std::ios::binary);
    if (!file)
    {
        std::cerr << "无法打开文件: " << filename.str() << std::endl;
        return;
    }

    // Tecplot二进制文件头常量
    constexpr float ZONEMARKER = 299.0f;
    constexpr float EOHMARKER = 357.0f;

    // 辅助lambda函数
    auto writeInt = [&file](int value)
    {
        file.write(reinterpret_cast<const char*>(&value), sizeof(int));
    };

    auto writeFloat = [&file](float value)
    {
        file.write(reinterpret_cast<const char*>(&value), sizeof(float));
    };

    auto writeString = [&file](const std::string& str)
    {
        file.write(str.c_str(), str.length());
    };

    // 文件标识和版本
    writeString("#!TDV101");
    writeInt(1);

    // 标题
    dumpString("Inamuro", file);

    // 变量定义
    const std::vector<std::string> variables = {
        "X", "Y", "Z", "u", "v", "w", "rho", "fei", "press"};

    writeInt(static_cast<int>(variables.size()));
    for (const auto& var : variables)
    {
        dumpString(var, file);
    }

    // Zone信息
    writeFloat(ZONEMARKER);
    dumpString("ZONE 001", file);

    // Zone配置
    writeInt(-1); // Zone Color
    writeInt(0);  // ZoneType
    writeInt(1);  // DataPacking (Point)
    writeInt(0);  // Var Location
    writeInt(0);  // Face connections

    // 网格尺寸
    writeInt(lx);
    writeInt(ly);
    writeInt(lz);

    writeInt(0); // No auxiliary data
    writeFloat(EOHMARKER);

    // Zone数据格式
    writeFloat(ZONEMARKER);

    // 数据格式定义（使用现代C++的数组初始化）
    const std::array<int, 9> data_formats = {3, 3, 3, 1, 1, 1, 1, 1, 1};
    for (int format : data_formats)
    {
        writeInt(format);
    }

    writeInt(0);  // No variable sharing
    writeInt(-1); // No connectivity sharing

    // 数据写入（使用现代C++的范围循环）
    for (int z = 0; z < lz; ++z)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int x = 0; x < lx; ++x)
            {
                // 坐标转换（转为1基索引）
                int coord_x = x + 1;
                int coord_y = y + 1;
                int coord_z = z + 1;

                // 检查障碍物（如果存在）
                bool is_obstacle = false; // TODO: 实现障碍物检查

                // 数据提取
                float press, rho_val, fei_val, u_val, v_val;
                if (is_obstacle)
                {
                    press = 0.0f;
                    rho_val = -1.0f;
                    fei_val = -1.0f;
                    u_val = -1.0f;
                    v_val = -1.0f;
                }
                else
                {
                    press = static_cast<float>(p[x][y][z + 1]);
                    rho_val = static_cast<float>(rho[x][y][z + 1]);
                    fei_val = static_cast<float>(fei[x][y][z + 1]);
                    u_val = static_cast<float>(u[x][y][z + 1]);
                    v_val = static_cast<float>(v[x][y][z + 1]);
                }
                float w_val = is_obstacle ? -1.0f : static_cast<float>(w[x][y][z + 1]);

                // 写入数据点（使用现代C++的批量操作）
                const std::array<int, 3> coords = {coord_x, coord_y, coord_z};
                const std::array<float, 6> values = {u_val, v_val, w_val, rho_val, fei_val, press};

                for (int coord : coords)
                {
                    writeInt(coord);
                }
                for (float value : values)
                {
                    writeFloat(value);
                }
            }
        }
    }

    file.close();
    std::cout << "Tecplot文件已写入: " << filename.str() << std::endl;
}

void Inamuro::dumpString(const std::string& str, std::ofstream& file)
{
    // 使用现代C++的字符串处理
    for (char c : str)
    {
        int char_code = static_cast<int>(c);
        file.write(reinterpret_cast<const char*>(&char_code), sizeof(int));
    }

    constexpr int null_terminator = 0;
    file.write(reinterpret_cast<const char*>(&null_terminator), sizeof(int));
}

void Inamuro::writeDebugOutput(int timeStep)
{
    std::ostringstream filename;
    filename << output_config.output_dir << "debug_" << std::setfill('0') << std::setw(9) << timeStep << ".csv";

    std::ofstream file(filename.str());
    if (!file)
    {
        std::cerr << "无法打开调试文件: " << filename.str() << std::endl;
        return;
    }

    // 使用现代C++的流操作
    file << "x,y,z,u,v,w,rho,fei,p\n";

    // 使用现代C++的格式化输出
    file << std::scientific << std::setprecision(6);

    for (int z = 0; z < lz; ++z)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int x = 0; x < lx; ++x)
            {
                file << (x + 1) << "," << (y + 1) << "," << (z + 1) << ","
                     << u[x][y][z + 1] << "," << v[x][y][z + 1] << "," << w[x][y][z + 1] << ","
                     << rho[x][y][z + 1] << "," << fei[x][y][z + 1] << "," << p[x][y][z + 1] << "\n";
            }
        }
    }

    file.close();
    std::cout << "调试CSV文件已写入: " << filename.str() << std::endl;
}

void Inamuro::updateFieldStatistics(FieldStatistics& stats, double value, double)
{
    if (std::isnan(value))
    {
        ++stats.nan_count;
        return;
    }

    if (!std::isfinite(value))
    {
        ++stats.inf_count;
        return;
    }

    if (value < stats.min)
    {
        stats.min = value;
    }
    if (value > stats.max)
    {
        stats.max = value;
    }

    const double abs_value = std::abs(value);
    if (abs_value > stats.abs_max)
    {
        stats.abs_max = abs_value;
    }
}

void Inamuro::finalizeFieldStatistics(FieldStatistics& stats, std::size_t valid_count, double accumulator)
{
    if (valid_count == 0)
    {
        stats.min = std::numeric_limits<double>::quiet_NaN();
        stats.max = std::numeric_limits<double>::quiet_NaN();
        stats.mean = std::numeric_limits<double>::quiet_NaN();
        stats.abs_max = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    stats.mean = accumulator / static_cast<double>(valid_count);
}
