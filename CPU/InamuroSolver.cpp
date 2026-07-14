#include "InamuroSolver.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace
{
std::string trimCopy(const std::string& text)
{
    const std::size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
    {
        return "";
    }

    const std::size_t last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}
} // namespace

// 构造函数：
// 1. 先构造 Inamuro，完成数组和初始液滴初始化
// 2. 再读取求解器自己的配置（步数、输出频率、日志开关）
// 3. 最后打开诊断文件，准备在 run() 中持续写入
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
    applyOutputConfiguration();
    openDiagnosticsFile();
}

InamuroSolver::~InamuroSolver()
{
    releaseDiagnosticsLock();
}

void InamuroSolver::setSolverConfig(const SolverConfig& config)
{
    solver_config = config;
    applyOutputConfiguration();
}

void InamuroSolver::run() // 运行求解器
{
    printSimulationHeader();
    // step 0 对应“初始场”，先写一条诊断，后面可以和 step 1 之后对比。
    recordStepDiagnostics(0);

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
        last_step_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(step_end - step_start).count() / 1000.0;

        // 一步结束后，把整个场的统计量落盘。
        // 这个文件是排查数值爆炸最核心的日志之一。
        recordStepDiagnostics(current_time_step + 1);

        // 累加时间（所有计算步骤都计入平均）
        total_step_time_ms += last_step_time_ms;

        // 计算步骤编号：current_time_step + 1
        int computation_step = current_time_step + 1;
        double avg_ms = total_step_time_ms / computation_step;

        std::cout << "  → 第 " << std::setw(3) << computation_step
                  << " 步计算完成 | 耗时: " << std::fixed << std::setprecision(2)
                  << last_step_time_ms << " ms | 平均: " << avg_ms << " ms/步" << std::endl;
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

    // 读取求解器主配置：
    // 第3行是总步数，第4行是输出频率。
    file >> max_time_steps >> output_frequency;

    std::cout << "求解器配置: max_time_steps=" << max_time_steps
              << ", output_frequency=" << output_frequency << std::endl;

    // 从这里开始允许写可选配置，格式是 key=value。
    // 这样不影响前面原有的物理参数读取逻辑。
    std::getline(file, line); // 读掉当前行的剩余内容

    while (std::getline(file, line))
    {
        const std::size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos)
        {
            line = line.substr(0, comment_pos);
        }

        line = trimCopy(line);
        if (line.empty())
        {
            continue;
        }

        const std::size_t pos = line.find('=');
        if (pos == std::string::npos)
        {
            continue;
        }

        std::string key = trimCopy(line.substr(0, pos));
        std::string value = line.substr(pos + 1);

        // 去除 value 中的行内注释（'#' 及其后面的内容）
        const std::size_t value_comment_pos = value.find('#');
        if (value_comment_pos != std::string::npos)
        {
            value = value.substr(0, value_comment_pos);
        }
        value = trimCopy(value);

        parseOptionalSetting(key, value);
    }
}

void InamuroSolver::parseOptionalSetting(const std::string& key, const std::string& value)
{
    // 这里只做最简单的布尔解析，便于手工改 params.in：
    // 1 / true / on 都视为开启，其余视为关闭。
    auto to_bool = [](const std::string& text) -> bool
    {
        return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
    };

    if (key == "enable_timing")
    {
        solver_config.enable_timing = to_bool(value);
    }
    else if (key == "enable_progress")
    {
        solver_config.enable_progress = to_bool(value);
    }
    else if (key == "enable_tecplot_output")
    {
        solver_config.enable_tecplot_output = to_bool(value);
    }
    else if (key == "enable_debug_field_csv")
    {
        solver_config.enable_debug_field_csv = to_bool(value);
    }
    else if (key == "enable_step_summary_csv")
    {
        solver_config.enable_step_summary_csv = to_bool(value);
    }
    else if (key == "enable_basic_warnings")
    {
        solver_config.enable_basic_warnings = to_bool(value);
    }
    else if (key == "output_dir")
    {
        solver_config.output_dir = value;
    }
    else if (key == "step_summary_filename")
    {
        solver_config.step_summary_filename = value;
    }
}

void InamuroSolver::applyOutputConfiguration()
{
    // Inamuro 只关心“实际怎么输出”，
    // 所以这里把求解器层的配置翻译成算法对象能理解的 OutputConfig。
    Inamuro::OutputConfig output_config;
    output_config.enable_tecplot = solver_config.enable_tecplot_output;
    output_config.enable_debug_field_csv = solver_config.enable_debug_field_csv;
    output_config.enable_step_summary_csv = solver_config.enable_step_summary_csv;
    output_config.output_dir = solver_config.output_dir;
    output_config.summary_filename = solver_config.step_summary_filename;
    inamuro->setOutputConfig(output_config);
}

void InamuroSolver::openDiagnosticsFile()
{
    if (!solver_config.enable_step_summary_csv)
    {
        return;
    }

    std::filesystem::create_directories(solver_config.output_dir);

    const std::filesystem::path diagnostics_path =
        std::filesystem::path(solver_config.output_dir) / solver_config.step_summary_filename;

    diagnostics_lock_dir = diagnostics_path;
    diagnostics_lock_dir += ".lock";

    std::error_code ec;
    if (!std::filesystem::create_directory(diagnostics_lock_dir, ec))
    {
        std::cerr << "警告: 诊断文件 " << diagnostics_path
                  << " 已被另一个仿真进程占用，当前进程将关闭 step_diagnostics.csv 写入。" << std::endl;
        diagnostics_lock_dir.clear();
        return;
    }

    diagnostics_stream.open(diagnostics_path, std::ios::out | std::ios::trunc);
    if (!diagnostics_stream)
    {
        std::cerr << "警告: 无法打开诊断文件 " << diagnostics_path << std::endl;
        releaseDiagnosticsLock();
        return;
    }

    diagnostics_stream << std::scientific << std::setprecision(10);
    writeDiagnosticsHeader();
}

void InamuroSolver::writeDiagnosticsHeader()
{
    if (!diagnostics_stream)
    {
        return;
    }

    diagnostics_stream
        << "time_step"
        << ",step_time_ms,pressure_solve_call_count,prev_pressure_iterations,pressure_iterations,pressure_hit_max_iter,pressure_solve_ms"
        << ",rho_min,rho_max,rho_mean,rho_abs_max,rho_nan_count,rho_inf_count"
        << ",fei_min,fei_max,fei_mean,fei_abs_max,fei_nan_count,fei_inf_count"
        << ",u_min,u_max,u_mean,u_abs_max,u_nan_count,u_inf_count"
        << ",v_min,v_max,v_mean,v_abs_max,v_nan_count,v_inf_count"
        << ",w_min,w_max,w_mean,w_abs_max,w_nan_count,w_inf_count"
        << ",vel_mag_min,vel_mag_max,vel_mag_mean,vel_mag_abs_max,vel_mag_nan_count,vel_mag_inf_count"
        << ",p_min,p_max,p_mean,p_abs_max,p_nan_count,p_inf_count"
        << "\n";
}

void InamuroSolver::recordStepDiagnostics(int timeStep)
{
    if (!diagnostics_stream)
    {
        return;
    }

    const Inamuro::DiagnosticsSnapshot snapshot = inamuro->collectDiagnostics();
    const Inamuro::PressureSolveDiagnostics pressure_diag = inamuro->getLastPressureSolveDiagnostics();
    const auto write_stats = [this](const Inamuro::FieldStatistics& stats)
    {
        diagnostics_stream
            << "," << stats.min
            << "," << stats.max
            << "," << stats.mean
            << "," << stats.abs_max
            << "," << stats.nan_count
            << "," << stats.inf_count;
    };

    diagnostics_stream << timeStep;
    if (timeStep == 0)
    {
        diagnostics_stream
            << ",0,0,0,0,0,0";
    }
    else
    {
        diagnostics_stream
            << "," << last_step_time_ms
            << "," << pressure_diag.pressure_solve_call_count
            << "," << pressure_diag.prev_iteration_count
            << "," << pressure_diag.iteration_count
            << "," << (pressure_diag.hit_max_iterations ? 1 : 0)
            << "," << pressure_diag.pressure_solve_time_ms;
    }
    write_stats(snapshot.rho_stats);
    write_stats(snapshot.fei_stats);
    write_stats(snapshot.u_stats);
    write_stats(snapshot.v_stats);
    write_stats(snapshot.w_stats);
    write_stats(snapshot.velocity_magnitude_stats);
    write_stats(snapshot.p_stats);
    diagnostics_stream << "\n";
    diagnostics_stream.flush();

    checkDiagnosticsWarnings(timeStep, snapshot);
}

void InamuroSolver::checkDiagnosticsWarnings(
    int timeStep,
    const Inamuro::DiagnosticsSnapshot& snapshot)
{
    if (!solver_config.enable_basic_warnings)
    {
        return;
    }

    const bool has_non_finite =
        snapshot.rho_stats.nan_count > 0 || snapshot.rho_stats.inf_count > 0 ||
        snapshot.fei_stats.nan_count > 0 || snapshot.fei_stats.inf_count > 0 ||
        snapshot.u_stats.nan_count > 0 || snapshot.u_stats.inf_count > 0 ||
        snapshot.v_stats.nan_count > 0 || snapshot.v_stats.inf_count > 0 ||
        snapshot.w_stats.nan_count > 0 || snapshot.w_stats.inf_count > 0 ||
        snapshot.velocity_magnitude_stats.nan_count > 0 || snapshot.velocity_magnitude_stats.inf_count > 0 ||
        snapshot.p_stats.nan_count > 0 || snapshot.p_stats.inf_count > 0;

    if (has_non_finite && !warned_non_finite)
    {
        warned_non_finite = true;
        std::cerr << "\n[告警] 第 " << timeStep
                  << " 步首次出现 NaN/Inf，请立即检查 step_diagnostics.csv" << std::endl;
    }

    if (snapshot.rho_stats.min < 0.0 && !warned_negative_rho)
    {
        warned_negative_rho = true;
        std::cerr << "\n[告警] 第 " << timeStep
                  << " 步首次出现负密度: rho_min=" << snapshot.rho_stats.min << std::endl;
    }
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
    std::cout << "Tecplot输出: " << (solver_config.enable_tecplot_output ? "开启" : "关闭") << std::endl;
    std::cout << "整场CSV输出: " << (solver_config.enable_debug_field_csv ? "开启" : "关闭") << std::endl;
    std::cout << "每步诊断摘要: " << (solver_config.enable_step_summary_csv ? "开启" : "关闭") << std::endl;
    std::cout << "基础告警: " << (solver_config.enable_basic_warnings ? "开启" : "关闭") << std::endl;
    std::cout << "压力Poisson: 严格无上限（每100次检查，tol=1e-3）" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
}

void InamuroSolver::updateProgress() // 更新进度
{
    if (solver_config.enable_progress)
    {
        double progress = static_cast<double>(current_time_step) / max_time_steps;

        // 简单的进度显示
        std::cout << "时间步 " << std::setw(6) << current_time_step
                  << "/" << max_time_steps
                  << " (" << std::fixed << std::setprecision(1) << progress * 100 << "%)";

        // 估算剩余时间（基于已完成的计算步骤的平均）
        if (solver_config.enable_timing && current_time_step > 0 && total_step_time_ms > 0)
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
    if (solver_config.enable_timing)
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

void InamuroSolver::releaseDiagnosticsLock()
{
    if (!diagnostics_lock_dir.empty())
    {
        std::error_code ec;
        std::filesystem::remove(diagnostics_lock_dir, ec);
        diagnostics_lock_dir.clear();
    }
}
