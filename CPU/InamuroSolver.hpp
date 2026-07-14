#pragma once

#include "Inamuro.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>

class InamuroSolver
{
public:
    // 求解器级配置：
    // 1. 控制台显示相关开关
    // 2. 各类输出文件开关
    // 3. 输出目录和诊断文件名
    // 这些配置既可以保留默认值，也可以在 params.in 末尾通过 key=value 覆盖。
    struct SolverConfig
    {
        bool enable_timing = true;           // 是否在终端输出总耗时和平均步耗时
        bool enable_progress = true;         // 是否显示进度条/剩余时间
        bool enable_tecplot_output = true;   // 是否按 output_frequency 写 Tecplot 二进制结果
        bool enable_debug_field_csv = false; // 是否按 output_frequency 写整场 CSV
        bool enable_step_summary_csv = true; // 是否每一步写 step_diagnostics.csv
        bool enable_basic_warnings = true;   // 是否在终端输出最基础的数值稳定性告警
        bool enable_nn_pressure_init = true; // 是否启用 NN 压力初始化
        bool enable_gpu_inference = false;   // 是否用 GPU 做 NN 推理
        std::string output_dir = "out/";     // 所有输出文件写入这个目录
        std::string step_summary_filename = "step_diagnostics.csv"; // 每步全场统计摘要
    };

    // 构造函数
    explicit InamuroSolver(const std::string& filename);

    // 析构函数
    ~InamuroSolver();

    // 禁用拷贝（防止意外复制大对象）
    InamuroSolver(const InamuroSolver&) = delete;
    InamuroSolver& operator=(const InamuroSolver&) = delete;

    // 主要接口
    void run();
    void setSolverConfig(const SolverConfig& config);

private:
    // 核心算法对象（智能指针管理）
    std::unique_ptr<Inamuro> inamuro;
    SolverConfig solver_config; // 当前生效的求解器配置

    // 配置参数
    int max_time_steps = 1000;   // 最大时间步
    int output_frequency = 10;   // 输出频率

    // 运行时状态
    int current_time_step;           // 当前时间步
    double init_time_ms;             // 初始化耗时（毫秒）
    double total_step_time_ms = 0.0; // 累计步骤时间（毫秒）
    double last_step_time_ms = 0.0;  // 最近一步总耗时（毫秒）
    bool warned_negative_rho = false;
    bool warned_non_finite = false;
    int nn_used_steps = 0;
    int nn_used_steps_first_200 = 0;
    int nn_used_steps_after_200 = 0;
    int first_nn_used_step = -1;
    int last_nn_used_step = -1;
    double nn_used_iteration_sum = 0.0;
    double nn_not_used_iteration_sum = 0.0;
    int nn_used_iteration_samples = 0;
    int nn_not_used_iteration_samples = 0;

    // 现代C++时间管理
    std::chrono::steady_clock::time_point start_time; // 开始时间
    std::ofstream diagnostics_stream; // step_diagnostics.csv 文件流
    std::filesystem::path diagnostics_lock_dir; // 防止两个进程同时写同一个诊断文件

    // 私有方法
    void loadConfiguration(const std::string& filename); // 加载配置
    void parseOptionalSetting(const std::string& key, const std::string& value); // 解析 params.in 中的 key=value 可选配置
    void applyOutputConfiguration();                     // 把求解器配置同步给 Inamuro
    void openDiagnosticsFile();                         // 打开每步摘要 CSV
    void writeDiagnosticsHeader();                      // 写入每步摘要 CSV 表头
    void recordStepDiagnostics(int timeStep);           // 记录每步诊断摘要
    void checkDiagnosticsWarnings(int timeStep,
                                  const Inamuro::DiagnosticsSnapshot& snapshot); // 只检查负密度和 NaN/Inf
    void printSimulationHeader() const;                  // 打印模拟头
    void updateProgress();                               // 更新进度
    void printCurrentTime() const;                       // 显示当前时间
    void printElapsedTime() const;                       // 打印耗时
    double getElapsedSeconds() const;                    // 获取耗时
    void releaseDiagnosticsLock();                       // 释放诊断文件锁
};
