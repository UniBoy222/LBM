#include "Inamuro.hpp"
#include "InamuroCUDA.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct RunConfig {
    std::string params = "params.in";
    std::string mode = "gpu";
    std::string poisson = "split";
    std::string pressure_boundary = "split";
    int steps = -1;
    int poisson_check_interval = 100;
    double poisson_tolerance = 0.001;
    double scalar_source_scale = 2.0;
    bool source_aware_hh_init = false;
    double source_aware_hh_scale = 1.0;
    double pressure_relax_scale = 1.0;
    double poisson_fixed_point_relax = 1.0;
    bool poisson_anderson_m1 = false;
    double poisson_anderson_beta_max = 1.0;
    bool poisson_two_grid_correction = false;
    double poisson_two_grid_strength = 0.5;
    int output_frequency = 100;
    std::set<int> output_steps;
    bool write_output = false;
    bool print_roofline = true;
    bool poisson_detail = false;
    bool poisson_graph = false;
    std::string poisson_diagnostics;
    bool poisson_spatial_diagnostics = false;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [--mode cpu|gpu] [--params FILE] [--steps N]\n"
        << "       [--poisson split|fused|onepass|scalar] [--pressure-boundary split|fused]\n"
        << "       [--poisson-check-interval N] [--poisson-tolerance X]\n"
        << "       [--scalar-source-scale X]\n"
        << "       [--source-aware-hh-init] [--source-aware-hh-scale X]\n"
        << "       [--pressure-relax-scale X] [--poisson-fixed-point-relax X]\n"
        << "       [--poisson-anderson-m1] [--poisson-anderson-beta-max X]\n"
        << "       [--poisson-two-grid-correction] [--poisson-two-grid-strength X]\n"
        << "       [--output-frequency N] [--output-steps A,B,C]\n"
        << "       [--poisson-graph] [--poisson-detail] [--poisson-diagnostics CSV]\n"
        << "       [--poisson-spatial-diagnostics]\n"
        << "       [--write-output] [--no-roofline]\n";
}

std::set<int> parse_output_steps(const std::string& text)
{
    std::set<int> steps;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        const int step = std::stoi(item);
        if (step < 0) {
            throw std::runtime_error("--output-steps values must be non-negative");
        }
        steps.insert(step);
    }
    return steps;
}

bool should_write_step(const RunConfig& cfg, int completed_step)
{
    if (!cfg.write_output) {
        return false;
    }
    if (!cfg.output_steps.empty()) {
        return cfg.output_steps.count(completed_step) > 0;
    }
    return completed_step % cfg.output_frequency == 0;
}

void load_run_config_from_params(const std::string& filename, RunConfig& cfg,
                                 bool steps_set, bool output_frequency_set)
{
    std::ifstream file(filename);
    if (!file) {
        return;
    }

    int lx = 0, ly = 0, lz = 0;
    int period = 0;
    int steps = 0;
    int output_frequency = 0;
    if (file >> lx >> ly >> lz >> period >> steps >> output_frequency) {
        if (!steps_set && cfg.steps < 0) {
            cfg.steps = steps;
        }
        if (!output_frequency_set) {
            cfg.output_frequency = output_frequency;
        }
    }
}

RunConfig parse_args(int argc, char** argv)
{
    RunConfig cfg;
    bool steps_set = false;
    bool output_frequency_set = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--mode") {
            cfg.mode = require_value(arg);
        } else if (arg == "--poisson") {
            cfg.poisson = require_value(arg);
        } else if (arg == "--pressure-boundary") {
            cfg.pressure_boundary = require_value(arg);
        } else if (arg == "--params") {
            cfg.params = require_value(arg);
        } else if (arg == "--steps") {
            cfg.steps = std::stoi(require_value(arg));
            steps_set = true;
        } else if (arg == "--poisson-check-interval") {
            cfg.poisson_check_interval = std::stoi(require_value(arg));
        } else if (arg == "--poisson-tolerance") {
            cfg.poisson_tolerance = std::stod(require_value(arg));
        } else if (arg == "--scalar-source-scale") {
            cfg.scalar_source_scale = std::stod(require_value(arg));
        } else if (arg == "--source-aware-hh-init") {
            cfg.source_aware_hh_init = true;
        } else if (arg == "--source-aware-hh-scale") {
            cfg.source_aware_hh_scale = std::stod(require_value(arg));
        } else if (arg == "--pressure-relax-scale") {
            cfg.pressure_relax_scale = std::stod(require_value(arg));
        } else if (arg == "--poisson-fixed-point-relax") {
            cfg.poisson_fixed_point_relax = std::stod(require_value(arg));
        } else if (arg == "--poisson-anderson-m1") {
            cfg.poisson_anderson_m1 = true;
        } else if (arg == "--poisson-anderson-beta-max") {
            cfg.poisson_anderson_beta_max = std::stod(require_value(arg));
        } else if (arg == "--poisson-two-grid-correction") {
            cfg.poisson_two_grid_correction = true;
        } else if (arg == "--poisson-two-grid-strength") {
            cfg.poisson_two_grid_strength = std::stod(require_value(arg));
        } else if (arg == "--output-frequency") {
            cfg.output_frequency = std::stoi(require_value(arg));
            output_frequency_set = true;
        } else if (arg == "--output-steps") {
            cfg.output_steps = parse_output_steps(require_value(arg));
        } else if (arg == "--write-output") {
            cfg.write_output = true;
        } else if (arg == "--poisson-detail") {
            cfg.poisson_detail = true;
        } else if (arg == "--poisson-graph") {
            cfg.poisson_graph = true;
        } else if (arg == "--poisson-diagnostics") {
            cfg.poisson_diagnostics = require_value(arg);
        } else if (arg == "--poisson-spatial-diagnostics") {
            cfg.poisson_spatial_diagnostics = true;
        } else if (arg == "--no-roofline") {
            cfg.print_roofline = false;
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (cfg.mode != "cpu" && cfg.mode != "gpu") {
        throw std::runtime_error("--mode must be cpu or gpu");
    }
    if (cfg.poisson != "split" && cfg.poisson != "fused" && cfg.poisson != "onepass" && cfg.poisson != "scalar") {
        throw std::runtime_error("--poisson must be split, fused, onepass, or scalar");
    }
    if (cfg.pressure_boundary != "split" && cfg.pressure_boundary != "fused") {
        throw std::runtime_error("--pressure-boundary must be split or fused");
    }
    if (cfg.poisson_check_interval <= 0) {
        throw std::runtime_error("--poisson-check-interval must be positive");
    }
    if (!(cfg.poisson_tolerance > 0.0)) {
        throw std::runtime_error("--poisson-tolerance must be positive");
    }
    if (!(cfg.poisson_fixed_point_relax > 0.0)) {
        throw std::runtime_error("--poisson-fixed-point-relax must be positive");
    }
    if (!(cfg.poisson_anderson_beta_max >= 0.0)) {
        throw std::runtime_error("--poisson-anderson-beta-max must be non-negative");
    }
    if (!(cfg.poisson_two_grid_strength >= 0.0)) {
        throw std::runtime_error("--poisson-two-grid-strength must be non-negative");
    }

    load_run_config_from_params(cfg.params, cfg, steps_set, output_frequency_set);
    if (cfg.steps < 0) {
        cfg.steps = 100;
    }
    if (cfg.output_frequency <= 0) {
        cfg.output_frequency = 100;
    }
    return cfg;
}

template <typename StepFn>
double run_loop(Inamuro& solver, const RunConfig& cfg, StepFn step_fn)
{
    if (cfg.write_output) {
        std::filesystem::create_directories("out");
        if (cfg.output_steps.empty() || cfg.output_steps.count(0) > 0) {
            solver.writeResults(0);
        }
    }

    auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < cfg.steps; ++step) {
        const int completed_step = step + 1;
        const bool need_output = should_write_step(cfg, completed_step);
        step_fn(need_output);
        if (need_output) {
            solver.writeResults(completed_step);
        }
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const RunConfig cfg = parse_args(argc, argv);

        std::cout << "LBM runner mode=" << cfg.mode
                  << " poisson=" << cfg.poisson
                  << " pressure_boundary=" << cfg.pressure_boundary
                  << " poisson_check_interval=" << cfg.poisson_check_interval
                  << " poisson_tolerance=" << cfg.poisson_tolerance
                  << " poisson_graph=" << (cfg.poisson_graph ? "yes" : "no")
                  << " poisson_detail=" << (cfg.poisson_detail ? "yes" : "no")
                  << " poisson_fixed_point_relax=" << cfg.poisson_fixed_point_relax
                  << " poisson_anderson_m1=" << (cfg.poisson_anderson_m1 ? "yes" : "no")
                  << " poisson_anderson_beta_max=" << cfg.poisson_anderson_beta_max
                  << " poisson_two_grid_correction=" << (cfg.poisson_two_grid_correction ? "yes" : "no")
                  << " poisson_two_grid_strength=" << cfg.poisson_two_grid_strength
                  << " poisson_diagnostics=" << (cfg.poisson_diagnostics.empty() ? "none" : cfg.poisson_diagnostics)
                  << " poisson_spatial_diagnostics=" << (cfg.poisson_spatial_diagnostics ? "yes" : "no")
                  << " params=" << cfg.params
                  << " steps=" << cfg.steps
                  << " output_steps=" << cfg.output_steps.size()
                  << " write_output=" << (cfg.write_output ? "yes" : "no")
                  << std::endl;

        Inamuro solver(cfg.params);
        int lx = 0, ly = 0, lz = 0;
        solver.getGridSize(lx, ly, lz);
        const double cells = static_cast<double>(lx) * ly * lz;

        if (cfg.mode == "cpu") {
            const double ms = run_loop(solver, cfg, [&](bool) { solver.performTimeStep(); });
            std::cout << std::fixed << std::setprecision(3)
                      << "CPU total_ms=" << ms
                      << " avg_ms_per_step=" << (ms / cfg.steps)
                      << " MLUPS=" << (cells * cfg.steps / (ms * 1000.0))
                      << std::endl;
            return 0;
        }

        InamuroCUDA gpu(solver);
        gpu.setUseFusedPoisson(cfg.poisson == "fused");
        gpu.setUseOnePassPoisson(cfg.poisson == "onepass");
        gpu.setUseScalarPoisson(cfg.poisson == "scalar");
        gpu.setScalarPoissonSourceScale(cfg.scalar_source_scale);
        gpu.setUseSourceAwareHHInit(cfg.source_aware_hh_init);
        gpu.setSourceAwareHHScale(cfg.source_aware_hh_scale);
        gpu.setPressureRelaxScale(cfg.pressure_relax_scale);
        gpu.setPoissonFixedPointRelax(cfg.poisson_fixed_point_relax);
        gpu.setUsePoissonAndersonM1(cfg.poisson_anderson_m1);
        gpu.setPoissonAndersonBetaMax(cfg.poisson_anderson_beta_max);
        gpu.setUsePoissonTwoGridCorrection(cfg.poisson_two_grid_correction);
        gpu.setPoissonTwoGridStrength(cfg.poisson_two_grid_strength);
        gpu.setUseFusedBoundaryPressure(cfg.pressure_boundary == "fused");
        gpu.setUsePoissonGraph(cfg.poisson_graph);
        gpu.setEnablePoissonDetailTiming(cfg.poisson_detail);
        gpu.setPoissonConvergence(cfg.poisson_check_interval, cfg.poisson_tolerance);
        gpu.setPoissonDiagnosticsPath(cfg.poisson_diagnostics);
        gpu.setUsePoissonSpatialDiagnostics(cfg.poisson_spatial_diagnostics);
        const double ms = run_loop(solver, cfg, [&](bool need_output) {
            gpu.performTimeStepGPU();
            if (need_output) {
                gpu.downloadFieldsToCPU(solver);
            }
        });
        gpu.downloadFieldsToCPU(solver);

        std::cout << std::fixed << std::setprecision(3)
                  << "GPU wall_total_ms=" << ms
                  << " wall_avg_ms_per_step=" << (ms / cfg.steps)
                  << " wall_MLUPS=" << (cells * cfg.steps / (ms * 1000.0))
                  << std::endl;
        gpu.printPerformanceMetrics();
        if (cfg.print_roofline) {
            gpu.printRooflineSummary();
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
