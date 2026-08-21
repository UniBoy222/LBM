#include "Inamuro.hpp"
#include "InamuroCUDA.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct RunConfig {
    std::string params = "params.in";
    int steps = -1;
    int output_every = 0;
};

void printUsage(const char* argv0)
{
    std::cout << "Usage: " << argv0
              << " [--params FILE] [--steps N] [--output-every N]\n";
}

RunConfig parseArgs(int argc, char** argv)
{
    RunConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* name) -> std::string {
            if (++i >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
            return argv[i];
        };
        if (arg == "--params")
            cfg.params = value("--params");
        else if (arg == "--steps")
            cfg.steps = std::stoi(value("--steps"));
        else if (arg == "--output-every")
            cfg.output_every = std::stoi(value("--output-every"));
        else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else
            throw std::runtime_error("unknown argument: " + arg);
    }
    if (!std::filesystem::is_regular_file(cfg.params))
        throw std::runtime_error("parameter file does not exist: " + cfg.params);
    if (cfg.steps == 0 || cfg.steps < -1)
        throw std::runtime_error("--steps must be positive or omitted");
    if (cfg.output_every < 0)
        throw std::runtime_error("--output-every must be non-negative");
    return cfg;
}

int readConfiguredSteps(const std::string& path)
{
    std::ifstream in(path);
    int nx = 0, ny = 0, nz = 0, period = 0, steps = 0;
    if (!(in >> nx >> ny >> nz >> period >> steps) || steps <= 0)
        throw std::runtime_error("cannot read positive outer time-step count from: " + path);
    return steps;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const RunConfig cfg = parseArgs(argc, argv);
        const int steps = (cfg.steps > 0) ? cfg.steps : readConfiguredSteps(cfg.params);

        int device = 0;
        cudaDeviceProp prop{};
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaGetDeviceProperties(&prop, device) != cudaSuccess)
            throw std::runtime_error("cannot query CUDA device");

        std::cout << "Pure single-GPU Inamuro LBM\n"
                  << "device=" << prop.name << " params=" << cfg.params
                  << " outer_steps=" << steps
                  << " poisson_max=1000 check_interval=100 tolerance=1e-3\n";

        Inamuro host(cfg.params);
        InamuroCUDA gpu(host);
        for (int step = 1; step <= steps; ++step) {
            gpu.performTimeStepGPU();
            const auto& diag = gpu.getLastPoissonDiagnostics();
            if (!diag.finite)
                throw std::runtime_error("non-finite Poisson state at outer step " +
                                         std::to_string(step));
            std::cout << "step=" << step
                      << " poisson_iterations=" << diag.iterations
                      << " residual=" << diag.relative_residual
                      << " converged=" << (diag.converged ? "yes" : "no") << "\n";
            if (cfg.output_every > 0 && step % cfg.output_every == 0) {
                gpu.downloadFieldsToCPU(host);
                host.writeResults(step);
            }
        }
        gpu.printPerformanceMetrics();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        printUsage(argv[0]);
        return 1;
    }
}
