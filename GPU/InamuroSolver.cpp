#include "InamuroSolver.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

// 构造函数
InamuroSolver::InamuroSolver(const std::string& filename)
    : current_time_step(0)
{
    // 记录初始化开始时间
    auto init_start = std::chrono::steady_clock::now();

    // 创建Inamuro对象（数组分配、液滴初始化）
    inamuro = std::make_unique<Inamuro>(filename);

    // 计算初始化耗时
    auto init_end = std::chrono::steady_clock::now();
    init_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();

    loadConfiguration(filename);
}

void InamuroSolver::run() // 运行求解器
{
    printSimulationHeader();

    std::cout << ">>> 初始化完成，耗时: " << std::fixed << std::setprecision(2)
              << init_time_ms << " ms" << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    // 主时间循环：0 到 max_time_steps-1（共max_time_steps步）
    for (current_time_step = 0; current_time_step < max_time_steps; ++current_time_step)
    {
        // 输出控制：每output_frequency步输出一次
        if (current_time_step % output_frequency == 0)
        {
            // 输出时间步编号
            std::cout << current_time_step << std::endl;

            // 输出当前时间
            printCurrentTime();

            // 输出进度信息
            updateProgress();

            // 写入结果文件
            inamuro->writeResults(current_time_step);
        }

        // 记录本步计算开始时间
        auto step_start = std::chrono::steady_clock::now();

        // 执行一个时间步
        inamuro->performTimeStep();

        // 计算本步耗时
        auto step_end = std::chrono::steady_clock::now();
        double step_ms = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count() / 1000.0;

        // 累加时间（所有计算步骤都计入平均）
        total_step_time_ms += step_ms;

        // 计算步骤编号：current_time_step + 1
        int computation_step = current_time_step + 1;
        double avg_ms = total_step_time_ms / computation_step;

        std::cout << "  → 第 " << std::setw(3) << computation_step
                  << " 步计算完成 | 耗时: " << std::fixed << std::setprecision(2)
                  << step_ms << " ms | 平均: " << avg_ms << " ms/步" << std::endl;
        std::cout << std::flush;
    }

    printElapsedTime();
    std::cout << "------   仿真完成   ---------" << std::endl;
}

void InamuroSolver::loadConfiguration(const std::string& filename) // 加载配置
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "警告: 无法打开参数文件，使用默认配置" << std::endl;
        return;
    }

    // 跳过网格尺寸行（已被Inamuro读取）
    std::string line;
    std::getline(file, line);

    // 跳过周期边界行
    std::getline(file, line);

    // 读取时间步和输出频率
    file >> max_time_steps >> output_frequency;

    std::cout << "求解器配置: max_time_steps=" << max_time_steps
              << ", output_frequency=" << output_frequency << std::endl;
}

void InamuroSolver::printSimulationHeader() const // 打印模拟头
{
    std::cout << "\n"
              << std::string(50, '=') << std::endl;
    std::cout << "    Inamuro LBM D3Q15 仿真开始" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    int lx, ly, lz;
    inamuro->getGridSize(lx, ly, lz);
    std::cout << "网格尺寸: " << lx << " × " << ly << " × " << lz << std::endl;
    std::cout << "最大时间步: " << max_time_steps << std::endl;
    std::cout << "输出频率: " << output_frequency << std::endl;
    std::cout << std::string(50, '-') << std::endl;
}

void InamuroSolver::updateProgress() // 更新进度
{
    if (enable_progress)
    {
        double progress = static_cast<double>(current_time_step) / max_time_steps;

        // 简单的进度显示
        std::cout << "时间步 " << std::setw(6) << current_time_step
                  << "/" << max_time_steps
                  << " (" << std::fixed << std::setprecision(1) << progress * 100 << "%)";

        // 估算剩余时间（基于已完成的计算步骤的平均）
        if (enable_timing && current_time_step > 0 && total_step_time_ms > 0)
        {
            double avg_ms_per_step = total_step_time_ms / current_time_step;
            double remaining_steps = max_time_steps - current_time_step;
            double remaining_s = (avg_ms_per_step * remaining_steps) / 1000.0;

            std::cout << " - 剩余: " << std::setprecision(0) << remaining_s << "s";
        }
        std::cout << std::endl;
    }
}

void InamuroSolver::printCurrentTime() const // 显示当前时间
{
    // 使用现代C++方法显示当前时间
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    struct tm* tm_info = localtime(&time_t_now);

    // 使用C++ iostream和格式化输出，保存原始格式设置
    auto old_fill = std::cout.fill('0');
    std::cout << std::setw(2) << tm_info->tm_hour << ":"
              << std::setw(2) << tm_info->tm_min << ":"
              << std::setw(2) << tm_info->tm_sec << std::endl;
    std::cout.fill(old_fill); // 恢复原始填充字符
}

void InamuroSolver::printElapsedTime() const // 打印耗时
{
    if (enable_timing)
    {
        double elapsed_s = total_step_time_ms / 1000.0;
        std::cout << "\n总计算时间: " << std::fixed << std::setprecision(2)
                  << elapsed_s << " 秒 (" << total_step_time_ms << " ms)" << std::endl;
        std::cout << "平均时间/步: " << std::fixed << std::setprecision(2)
                  << (total_step_time_ms / max_time_steps) << " ms" << std::endl;
    }
}

double InamuroSolver::getElapsedSeconds() const // 获取耗时（已废弃，保留接口兼容性）
{
    return total_step_time_ms / 1000.0;
}
