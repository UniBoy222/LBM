#pragma once

#include "LBMBase.hpp"
#include "common.hpp"
#include <string>
#include <vector>

/**
 * Inamuro算法核心类 - 继承LBM标准基类，实现Inamuro两相流算法
 */
class Inamuro : public LBMBase
{
    friend class InamuroCUDA;
protected:
    // === 网格尺寸 ===
    int lx, ly, lz;

    // === 分布函数 [Q][lx][ly][lz] ===
    Vector4D ff, gg, hh;

    // === 宏观量 （z方向多两层虚拟层）[lx][ly][lz+2] ===
    Vector3D rho;
    Vector3D fei;
    Vector3D u, v, w; // 速度
    Vector3D p;

    // === 梯度场（计算用） [lx][ly][lz] ===
    Vector3D fei_x, fei_y, fei_z; // 相场梯度
    Vector3D rho_x, rho_y, rho_z; // 密度梯度
    Vector3D u_x, u_y, u_z;       // u速度的三个方向梯度 (∂u/∂x, ∂u/∂y, ∂u/∂z)
    Vector3D v_x, v_y, v_z;       // v速度的三个方向梯度 (∂v/∂x, ∂v/∂y, ∂v/∂z)
    Vector3D w_x, w_y, w_z;       // w速度的三个方向梯度 (∂w/∂x, ∂w/∂y, ∂w/∂z)
    Vector3D fei_lap;             // 相场拉普拉斯 (∇²φ)
    Vector3D u_lap, v_lap, w_lap; // 速度拉普拉斯 (∇²u, ∇²v, ∇²w)
    // 参数结构体
    struct Parameters
    {
        int period = 1;
        // === 流体物性参数 ===
        double rho_L = 50.0, rho_G = 1.0;    // 液相/气相密度
        double mu_L = 0.008, mu_G = 0.00016; // 液相/气相粘度

        // === LBM参数 ===
        double tauf = 1.0, taug = 1.0; // 松弛时间
        double gam_l, gam_g;           // 运动粘性系数（由松弛时间计算）

        // === 相场参数 ===
        double k_f = 0.5, k_g = 0.0005;              // 相场系数/表面张力系数
        double fei_L = 0.092, fei_G = 0.015;         // 液相/气相相场值
        double fei_max = 0.09714, fei_min = 0.01134; // 相场边界值

        // === 物理几何参数 ===
        double DD = 32.0; // 液滴直径

        // === 初始条件参数 ===
        std::string init_mode = "two_droplets"; // two_droplets 或 single_droplet
        std::string init_profile = "sharp";     // sharp 或 tanh
        double init_velocity = 0.035;
        double interface_width = 2.0;
        double init_center_x = -1.0;
        double init_center_y = -1.0;
        double init_center_z = -1.0;
        double init_separation = -1.0;
        double init_offset_x = 0.0;

        // === 状态方程参数 ===
        double T = 0.035, a = 1.0, b = 6.7; // van der Waals EOS参数

        // === 构造函数重载 ===
        Parameters();
        Parameters(const std::string& filename);

        // === 方法 ===
        void update_gam();
        void print() const;
    };
    Parameters params;

public:
    // === 构造函数重载 ===
    explicit Inamuro(int nx = 48, int ny = 96, int nz = 128); // 默认网格尺寸构造函数
    explicit Inamuro(const std::string& filename);   // 从文件读取所有参数构造函数
    virtual ~Inamuro() = default;

    // === 实现LBMBase的纯虚函数 ===
    void collision() override;
    void stream(Vector4D& dist) override;
    void applyBoundaryConditions(Vector4D& dist) override;
    void getMacro() override;
    void performTimeStep() override;
    void writeResults(int timeStep) override;
    std::string getAlgorithmName() const override;

    // === 数据访问接口 ===
    void getGridSize(int& nx, int& ny, int& nz) const override;
    void setOutputDirectory(const std::string& output_dir);
    std::string getOutputDirectory() const;

private:
    template <typename T>
    void resize3D(std::vector<std::vector<std::vector<T>>>& vec,
                  int nx, int ny, int nz, const T& value = T{})
    {
        vec.resize(nx);
        for (auto& vec_y : vec)
        {
            vec_y.resize(ny);
            for (auto& vec_z : vec_y)
            {
                vec_z.resize(nz, value);
            }
        }
    }

    template <typename T>
    void resize4D(std::vector<std::vector<std::vector<std::vector<T>>>>& vec,
                  int nq, int nx, int ny, int nz, const T& value = T{})
    {
        vec.resize(nq);
        for (auto& vec_x : vec)
        {
            resize3D(vec_x, nx, ny, nz, value);
        }
    }
    // === 算法实现方法 ===
    void initializeArrays();           // 初始化数组
    virtual void initializeDroplets(); // 初始化两个液滴
    void slipBounceBack(Vector4D& dist); // 滑移反弹

    // 压力泊松方程
    void solvePressurePoisson(); // 求解压力泊松方程
    double getError(Vector3D& pressure_prev); // 获取压力误差

    void collision_p(); // 碰撞压力
    void getp(); // 获取压力
    void correct_uvw(); // 修正速度
    void update_hh(); // 更新压力分布函数

    // === 数值方法辅助函数 ===
    void EOS(double, double&); // 状态方程

    // === 求导模板函数 ===
    template <typename T>
    void firstord(const T&, Vector3D&, Vector3D&, Vector3D&);

    template <typename T>
    void secondord(const T&, Vector3D&);

    // === 辅助函数 ===
    double getValue(const Vector3D&, int, int, int) const;

    // === 文件输出相关方法 ===
    void writeTecplotBinary(int timeStep);                        // 写入Tecplot二进制文件
    void writeDebugOutput(int timeStep);                          // 写入调试文件
    void dumpString(const std::string& str, std::ofstream& file); // 写入字符串

    // === 输出配置 ===
    struct OutputConfig
    {
        bool enable_tecplot = true;      // 是否启用Tecplot二进制文件
        bool enable_debug = false;       // 是否启用调试文件
        std::string output_dir = "out/"; // 输出目录
    } output_config;
};
// 求导模板函数实现
template <typename T>
inline void Inamuro::firstord(const T& var, Vector3D& var_x, Vector3D& var_y, Vector3D& var_z) // 一阶导数
{
    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                var_x[x][y][z] = 0.0;
                var_y[x][y][z] = 0.0;
                var_z[x][y][z] = 0.0;

                // 使用 D3Q15 的 1-14 方向 (排除静止方向 0)
                for (int k = 1; k < D3Q15::Q; ++k)
                {
                    // 计算相邻格点索引，处理边界条件
                    int xp = x + D3Q15::ex[k];
                    int yp = y + D3Q15::ey[k];
                    int zp = z + D3Q15::ez[k];

                    // x方向：对称边界条件 (slip wall)
                    if (xp >= lx)
                        xp = lx - 2;
                    if (xp < 0)
                        xp = 1;

                    // y, z方向：周期边界条件
                    if (yp >= ly)
                        yp = 0;
                    if (yp < 0)
                        yp = ly - 1;
                    if (zp >= lz)
                        zp = 0;
                    if (zp < 0)
                        zp = lz - 1;

                    // 获取变量值（处理可能的索引偏移）
                    double var_value = getValue(var, xp, yp, zp);

                    // 累加梯度贡献
                    var_x[x][y][z] += var_value * D3Q15::uc[k];
                    var_y[x][y][z] += var_value * D3Q15::vc[k];
                    var_z[x][y][z] += var_value * D3Q15::wc[k];
                }

                // 标准化 (除以 10.0)
                var_x[x][y][z] /= 10.0;
                var_y[x][y][z] /= 10.0;
                var_z[x][y][z] /= 10.0;
            }
        }
    }
}
// 二阶导数模板函数实现
template <typename T>
inline void Inamuro::secondord(const T& var, Vector3D& var_lap) // 二阶导数
{
    // 计算任意变量的拉普拉斯

    for (int x = 0; x < lx; ++x)
    {
        for (int y = 0; y < ly; ++y)
        {
            for (int z = 0; z < lz; ++z)
            {
                var_lap[x][y][z] = 0.0;

                // 使用 D3Q15 的 1-14 方向 (排除静止方向 0)
                for (int k = 1; k < D3Q15::Q; ++k)
                {
                    int xp = x + D3Q15::ex[k];
                    int yp = y + D3Q15::ey[k];
                    int zp = z + D3Q15::ez[k];

                    // 边界条件处理
                    if (xp >= lx)
                        xp = lx - 2;
                    if (xp < 0)
                        xp = 1;
                    if (yp >= ly)
                        yp = 0;
                    if (yp < 0)
                        yp = ly - 1;
                    if (zp >= lz)
                        zp = 0;
                    if (zp < 0)
                        zp = lz - 1;

                    // 获取变量值
                    double var_value = getValue(var, xp, yp, zp);

                    var_lap[x][y][z] += var_value;
                }

                // 拉普拉斯算子：(sum - 14*center) / 5.0
                double center_value = getValue(var, x, y, z);
                var_lap[x][y][z] = (var_lap[x][y][z] - 14.0 * center_value) / 5.0;
            }
        }
    }
}
