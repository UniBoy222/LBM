#pragma once

#include "Inamuro.hpp"
#include <chrono>
#include <memory>

class InamuroSolver
{
public:
    // 构造函数
    explicit InamuroSolver(const std::string& filename);

    // 析构函数
    ~InamuroSolver() = default;

    // 禁用拷贝（防止意外复制大对象）
    InamuroSolver(const InamuroSolver&) = delete;
    InamuroSolver& operator=(const InamuroSolver&) = delete;

    // 主要接口
    void run();

private:
    // 核心算法对象（智能指针管理）
    std::unique_ptr<Inamuro> inamuro;

    // 配置参数
    int max_time_steps = 1000;   // 最大时间步
    int output_frequency = 10;   // 输出频率
    bool enable_timing = true;   // 是否启用计时
    bool enable_progress = true; // 是否显示进度

    // 运行时状态
    int current_time_step;           // 当前时间步
    double init_time_ms;             // 初始化耗时（毫秒）
    double total_step_time_ms = 0.0; // 累计步骤时间（毫秒）

    // 现代C++时间管理
    std::chrono::steady_clock::time_point start_time; // 开始时间

    // 私有方法
    void loadConfiguration(const std::string& filename); // 加载配置
    void printSimulationHeader() const;                  // 打印模拟头
    void updateProgress();                               // 更新进度
    void printCurrentTime() const;                       // 显示当前时间
    void printElapsedTime() const;                       // 打印耗时
    double getElapsedSeconds() const;                    // 获取耗时
};