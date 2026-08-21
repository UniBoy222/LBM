#include "Inamuro.hpp"
#include "InamuroCUDA.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kRelTolerance = 1.0e-10;
constexpr double kAbsFloor = 1.0e-12;

struct FieldError {
    std::string name;
    double max_abs = 0.0;
    double max_ref = 0.0;
    double rel_l2 = 0.0;

    bool pass() const
    {
        return std::isfinite(max_abs) && std::isfinite(max_ref) &&
               std::isfinite(rel_l2) &&
               rel_l2 <= kRelTolerance &&
               max_abs <= kAbsFloor + kRelTolerance * max_ref;
    }
};

class DiagnosticInamuro final : public Inamuro {
public:
    using Inamuro::Inamuro;

    FieldError compareMacro(const DiagnosticInamuro& ref, const std::string& name) const
    {
        int nx = 0, ny = 0, nz = 0;
        getGridSize(nx, ny, nz);
        long double diff2 = 0.0L, ref2 = 0.0L;
        FieldError out{name};
        for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
        for (int z = 0; z < nz + 2; ++z) {
            const double a = macro(name, x, y, z);
            const double b = ref.macro(name, x, y, z);
            const double d = a - b;
            if (!std::isfinite(a) || !std::isfinite(b))
                return FieldError{name, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()};
            diff2 += static_cast<long double>(d) * d;
            ref2 += static_cast<long double>(b) * b;
            out.max_abs = std::max(out.max_abs, std::abs(d));
            out.max_ref = std::max(out.max_ref, std::abs(b));
        }
        out.rel_l2 = (ref2 > 0.0L)
            ? std::sqrt(static_cast<double>(diff2 / ref2))
            : std::sqrt(static_cast<double>(diff2));
        return out;
    }

    FieldError compareDistribution(const DiagnosticInamuro& ref,
                                   const std::string& family, int q) const
    {
        int nx = 0, ny = 0, nz = 0;
        getGridSize(nx, ny, nz);
        long double diff2 = 0.0L, ref2 = 0.0L;
        FieldError out{family + std::to_string(q)};
        for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
        for (int z = 0; z < nz; ++z) {
            const double a = distribution(family, q, x, y, z);
            const double b = ref.distribution(family, q, x, y, z);
            const double d = a - b;
            if (!std::isfinite(a) || !std::isfinite(b))
                return FieldError{out.name, std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity()};
            diff2 += static_cast<long double>(d) * d;
            ref2 += static_cast<long double>(b) * b;
            out.max_abs = std::max(out.max_abs, std::abs(d));
            out.max_ref = std::max(out.max_ref, std::abs(b));
        }
        out.rel_l2 = (ref2 > 0.0L)
            ? std::sqrt(static_cast<double>(diff2 / ref2))
            : std::sqrt(static_cast<double>(diff2));
        return out;
    }

    bool invariants(double& worst) const
    {
        int nx = 0, ny = 0, nz = 0;
        getGridSize(nx, ny, nz);
        worst = 0.0;
        const int pairs[5][2] = {{1,4}, {7,8}, {9,14}, {10,13}, {12,11}};

        for (int x = 0; x < nx; ++x)
        for (int y = 0; y < ny; ++y)
        for (int z = 0; z < nz; ++z) {
            double ff_sum = 0.0;
            double hh_sum = 0.0;
            const double pressure = p[x][y][z + 1];
            for (int q = 0; q < D3Q15::Q; ++q) {
                const double fq = ff[q][x][y][z];
                const double gq = gg[q][x][y][z];
                const double hq = hh[q][x][y][z];
                if (!std::isfinite(fq) || !std::isfinite(gq) || !std::isfinite(hq) ||
                    !std::isfinite(pressure) || !std::isfinite(fei[x][y][z + 1]) ||
                    !std::isfinite(rho[x][y][z + 1]) ||
                    !std::isfinite(u[x][y][z + 1]) ||
                    !std::isfinite(v[x][y][z + 1]) ||
                    !std::isfinite(w[x][y][z + 1]))
                    return false;
                ff_sum += fq;
                hh_sum += hq;
                worst = std::max(worst, std::abs(hq - D3Q15::Ei[q] * pressure));
            }
            worst = std::max(worst, std::abs(ff_sum - fei[x][y][z + 1]));
            worst = std::max(worst, std::abs(hh_sum - pressure));
        }

        for (int y = 0; y < ny; ++y)
        for (int z = 0; z < nz; ++z)
        for (const auto& pair : pairs) {
            worst = std::max(worst, std::abs(ff[pair[0]][0][y][z] - ff[pair[1]][0][y][z]));
            worst = std::max(worst, std::abs(gg[pair[0]][0][y][z] - gg[pair[1]][0][y][z]));
            worst = std::max(worst, std::abs(ff[pair[1]][nx-1][y][z] - ff[pair[0]][nx-1][y][z]));
            worst = std::max(worst, std::abs(gg[pair[1]][nx-1][y][z] - gg[pair[0]][nx-1][y][z]));
        }
        return std::isfinite(worst) && worst <= kAbsFloor;
    }

private:
    double macro(const std::string& name, int x, int y, int z) const
    {
        if (name == "rho") return rho[x][y][z];
        if (name == "fei") return fei[x][y][z];
        if (name == "u") return u[x][y][z];
        if (name == "v") return v[x][y][z];
        if (name == "w") return w[x][y][z];
        if (name == "p") return p[x][y][z];
        throw std::runtime_error("unknown macro field: " + name);
    }

    double distribution(const std::string& family, int q, int x, int y, int z) const
    {
        if (family == "ff") return ff[q][x][y][z];
        if (family == "gg") return gg[q][x][y][z];
        if (family == "hh") return hh[q][x][y][z];
        throw std::runtime_error("unknown distribution family: " + family);
    }
};

struct Config {
    std::string params = "params_small.in";
    int steps = 2;
};

Config parseArgs(int argc, char** argv)
{
    Config cfg;
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
        else
            throw std::runtime_error("unknown argument: " + arg);
    }
    if (!std::filesystem::is_regular_file(cfg.params))
        throw std::runtime_error("parameter file does not exist: " + cfg.params);
    if (cfg.steps <= 0)
        throw std::runtime_error("--steps must be positive");
    return cfg;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Config cfg = parseArgs(argc, argv);
        DiagnosticInamuro cpu(cfg.params);
        DiagnosticInamuro gpu_host(cfg.params);
#ifdef LBM_TESTING
        {
            InamuroCUDA finite_probe(gpu_host);
            if (!finite_probe.testFiniteGateRejectsNaN())
                throw std::runtime_error("finite-state negative gate failed");
            std::cout << "negative_finite_gate=PASS\n";
        }
#endif
        InamuroCUDA gpu(gpu_host);

        bool pass = true;
        double global_worst_rel = 0.0;
        double global_worst_abs = 0.0;
        std::string global_worst_name;
        const std::vector<std::string> macro_names = {"rho", "fei", "u", "v", "w", "p"};
        const std::vector<std::string> families = {"ff", "gg", "hh"};
        int converged_steps = 0;
        int cap_hits = 0;

        auto compare_state = [&](const std::string& label, bool require_exact) {
            bool state_pass = true;
            std::vector<FieldError> errors;
            for (const auto& name : macro_names)
                errors.push_back(gpu_host.compareMacro(cpu, name));
            for (const auto& family : families)
                for (int q = 0; q < D3Q15::Q; ++q)
                    errors.push_back(gpu_host.compareDistribution(cpu, family, q));

            for (const auto& err : errors) {
                if (err.rel_l2 > global_worst_rel) {
                    global_worst_rel = err.rel_l2;
                    global_worst_name = err.name;
                }
                global_worst_abs = std::max(global_worst_abs, err.max_abs);
                const bool field_pass = err.pass() && (!require_exact || err.max_abs == 0.0);
                if (!field_pass) {
                    state_pass = false;
                    std::cerr << "FIELD_FAIL phase=" << label << " field=" << err.name
                              << " max_abs=" << err.max_abs
                              << " max_ref=" << err.max_ref
                              << " rel_l2=" << err.rel_l2 << "\n";
                }
            }
            return state_pass;
        };

        std::cout << std::scientific << std::setprecision(12);
        gpu.downloadFieldsToCPU(gpu_host);
        const bool time0_fields = compare_state("time0", true);
        const bool time0_pass = time0_fields;
        pass = pass && time0_pass;
        std::cout << "time0_equivalence=" << (time0_pass ? "PASS" : "FAIL")
                  << " exact_roundtrip=" << (time0_fields ? "yes" : "no") << "\n";

        for (int step = 1; step <= cfg.steps; ++step) {
            cpu.performTimeStep();
            gpu.performTimeStepGPU();
            gpu.downloadFieldsToCPU(gpu_host);

            const bool field_pass = compare_state("step" + std::to_string(step), false);

            const auto& gpu_diag = gpu.getLastPoissonDiagnostics();
            const int cpu_iterations = cpu.getLastPoissonIterations();
            const double cpu_residual = cpu.getLastPoissonResidual();
            const auto& cpu_trace = cpu.getLastPoissonResidualTrace();
            const double residual_diff = std::abs(gpu_diag.relative_residual - cpu_residual);
            const double residual_limit = kAbsFloor + kRelTolerance * std::abs(cpu_residual);
            bool trace_pass = gpu_diag.residual_trace.size() == cpu_trace.size();
            double trace_max_diff = 0.0;
            if (trace_pass) {
                for (size_t i = 0; i < cpu_trace.size(); ++i) {
                    const double diff = std::abs(gpu_diag.residual_trace[i] - cpu_trace[i]);
                    trace_max_diff = std::max(trace_max_diff, diff);
                    const double limit = kAbsFloor + kRelTolerance * std::abs(cpu_trace[i]);
                    if (!std::isfinite(gpu_diag.residual_trace[i]) || diff > limit) {
                        trace_pass = false;
                        break;
                    }
                }
            }
            const bool diagnostic_pass =
                gpu_diag.finite &&
                gpu_diag.iterations == cpu_iterations &&
                residual_diff <= residual_limit &&
                trace_pass &&
                gpu_diag.converged == (cpu_residual < 1.0e-3);

            double cpu_invariant = 0.0, gpu_invariant = 0.0;
            const bool cpu_invariant_pass = cpu.invariants(cpu_invariant);
            const bool gpu_invariant_pass = gpu_host.invariants(gpu_invariant);
            const bool step_equivalence = field_pass && diagnostic_pass &&
                                          cpu_invariant_pass && gpu_invariant_pass;
            pass = pass && step_equivalence;
            const bool converged = gpu_diag.converged && cpu_residual < 1.0e-3;
            if (converged)
                ++converged_steps;
            else if (gpu_diag.iterations == 1000)
                ++cap_hits;

            std::cout << "step=" << step
                      << " cpu_iter=" << cpu_iterations
                      << " gpu_iter=" << gpu_diag.iterations
                      << " cpu_residual=" << cpu_residual
                      << " gpu_residual=" << gpu_diag.relative_residual
                      << " residual_diff=" << residual_diff
                      << " trace_points=" << cpu_trace.size()
                      << " trace_max_diff=" << trace_max_diff
                      << " cpu_invariant=" << cpu_invariant
                      << " gpu_invariant=" << gpu_invariant
                      << " equivalence=" << (step_equivalence ? "PASS" : "FAIL")
                      << " convergence=" << (converged ? "YES" : "NO")
                      << " cap_hit=" << ((!converged && gpu_diag.iterations == 1000) ? "yes" : "no")
                      << "\n";
        }

        std::cout << "summary steps=" << cfg.steps
                  << " time0_equivalence=" << (time0_pass ? "PASS" : "FAIL")
                  << " compared_fields_per_step=51"
                  << " worst_field=" << global_worst_name
                  << " worst_rel_l2=" << global_worst_rel
                  << " worst_max_abs=" << global_worst_abs
                  << " equivalence=" << (pass ? "PASS" : "FAIL")
                  << " converged_steps=" << converged_steps << "/" << cfg.steps
                  << " cap_hits=" << cap_hits << "\n";
        return pass ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
