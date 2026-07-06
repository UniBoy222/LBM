#include "Inamuro.hpp"
#include "InamuroCUDA.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CompareConfig {
    std::string params = "params.in";
    std::string poisson = "split";
    std::string pressure_boundary = "split";
    int steps = 1;
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
    bool poisson_graph = false;
    std::string poisson_diagnostics;
    bool poisson_spatial_diagnostics = false;
    double tolerance = 1.0e-8;
    double pressure_tolerance = 1.0e-8;
};

struct FieldError {
    std::string name;
    double max_abs = 0.0;
    double rel_l2 = 0.0;
};

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0 << " [--params FILE] [--steps N]\n"
        << "       [--poisson split|fused|onepass|scalar] [--pressure-boundary split|fused]\n"
        << "       [--poisson-check-interval N] [--poisson-tolerance X]\n"
        << "       [--scalar-source-scale X]\n"
        << "       [--source-aware-hh-init] [--source-aware-hh-scale X]\n"
        << "       [--pressure-relax-scale X] [--poisson-fixed-point-relax X]\n"
        << "       [--poisson-anderson-m1] [--poisson-anderson-beta-max X]\n"
        << "       [--poisson-two-grid-correction] [--poisson-two-grid-strength X]\n"
        << "       [--poisson-graph] [--poisson-diagnostics CSV]\n"
        << "       [--poisson-spatial-diagnostics]\n"
        << "       [--tolerance X] [--pressure-tolerance X]\n";
}

CompareConfig parse_args(int argc, char** argv)
{
    CompareConfig cfg;
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
        } else if (arg == "--poisson") {
            cfg.poisson = require_value(arg);
        } else if (arg == "--pressure-boundary") {
            cfg.pressure_boundary = require_value(arg);
        } else if (arg == "--params") {
            cfg.params = require_value(arg);
        } else if (arg == "--steps") {
            cfg.steps = std::stoi(require_value(arg));
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
        } else if (arg == "--poisson-graph") {
            cfg.poisson_graph = true;
        } else if (arg == "--poisson-diagnostics") {
            cfg.poisson_diagnostics = require_value(arg);
        } else if (arg == "--poisson-spatial-diagnostics") {
            cfg.poisson_spatial_diagnostics = true;
        } else if (arg == "--tolerance") {
            cfg.tolerance = std::stod(require_value(arg));
        } else if (arg == "--pressure-tolerance") {
            cfg.pressure_tolerance = std::stod(require_value(arg));
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (cfg.steps < 0) {
        throw std::runtime_error("--steps must be non-negative");
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
    return cfg;
}

class DiagnosticInamuro : public Inamuro {
public:
    using Inamuro::Inamuro;

    FieldError compare_field(const DiagnosticInamuro& ref, const std::string& name) const
    {
        int nx = 0, ny = 0, nz = 0;
        getGridSize(nx, ny, nz);

        double sum_diff2 = 0.0;
        double sum_ref2 = 0.0;
        double max_abs = 0.0;

        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < nz; ++z) {
                    const int zg = z + 1;
                    const double a = value(name, x, y, zg);
                    const double b = ref.value(name, x, y, zg);
                    const double diff = a - b;
                    sum_diff2 += diff * diff;
                    sum_ref2 += b * b;
                    max_abs = std::max(max_abs, std::abs(diff));
                }
            }
        }

        FieldError err;
        err.name = name;
        err.max_abs = max_abs;
        err.rel_l2 = (sum_ref2 > 0.0) ? std::sqrt(sum_diff2 / sum_ref2) : std::sqrt(sum_diff2);
        return err;
    }

    double phase_mass() const
    {
        int nx = 0, ny = 0, nz = 0;
        getGridSize(nx, ny, nz);
        double mass = 0.0;
        for (int x = 0; x < nx; ++x) {
            for (int y = 0; y < ny; ++y) {
                for (int z = 0; z < nz; ++z) {
                    mass += fei[x][y][z + 1];
                }
            }
        }
        return mass;
    }

private:
    double value(const std::string& name, int x, int y, int z) const
    {
        if (name == "fei") return fei[x][y][z];
        if (name == "rho") return rho[x][y][z];
        if (name == "u") return u[x][y][z];
        if (name == "v") return v[x][y][z];
        if (name == "w") return w[x][y][z];
        if (name == "p") return p[x][y][z];
        throw std::runtime_error("unknown field: " + name);
    }
};

} // namespace

int main(int argc, char** argv)
{
    try {
        const CompareConfig cfg = parse_args(argc, argv);

        DiagnosticInamuro cpu(cfg.params);
        DiagnosticInamuro gpu_host(cfg.params);
        InamuroCUDA gpu(gpu_host);
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
        gpu.setPoissonConvergence(cfg.poisson_check_interval, cfg.poisson_tolerance);
        gpu.setPoissonDiagnosticsPath(cfg.poisson_diagnostics);
        gpu.setUsePoissonSpatialDiagnostics(cfg.poisson_spatial_diagnostics);

        for (int step = 0; step < cfg.steps; ++step) {
            cpu.performTimeStep();
            gpu.performTimeStepGPU();
        }
        gpu.downloadFieldsToCPU(gpu_host);

        const std::vector<std::string> fields = {"fei", "rho", "u", "v", "w", "p"};
        bool pass = true;

        std::cout << "CPU/GPU comparison steps=" << cfg.steps
                  << " poisson=" << cfg.poisson
                  << " pressure_boundary=" << cfg.pressure_boundary
                  << " poisson_check_interval=" << cfg.poisson_check_interval
                  << " poisson_tolerance=" << cfg.poisson_tolerance
                  << " poisson_graph=" << (cfg.poisson_graph ? "yes" : "no")
                  << " poisson_fixed_point_relax=" << cfg.poisson_fixed_point_relax
                  << " poisson_anderson_m1=" << (cfg.poisson_anderson_m1 ? "yes" : "no")
                  << " poisson_anderson_beta_max=" << cfg.poisson_anderson_beta_max
                  << " poisson_two_grid_correction=" << (cfg.poisson_two_grid_correction ? "yes" : "no")
                  << " poisson_two_grid_strength=" << cfg.poisson_two_grid_strength
                  << " poisson_diagnostics=" << (cfg.poisson_diagnostics.empty() ? "none" : cfg.poisson_diagnostics)
                  << " poisson_spatial_diagnostics=" << (cfg.poisson_spatial_diagnostics ? "yes" : "no")
                  << " params=" << cfg.params << std::endl;
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "field,max_abs,relative_l2,status\n";
        for (const auto& field : fields) {
            const FieldError err = gpu_host.compare_field(cpu, field);
            const double tol = (field == "p") ? cfg.pressure_tolerance : cfg.tolerance;
            const bool field_pass = err.rel_l2 <= tol;
            pass = pass && field_pass;
            std::cout << err.name << ","
                      << err.max_abs << ","
                      << err.rel_l2 << ","
                      << (field_pass ? "PASS" : "FAIL") << "\n";
        }

        const double cpu_mass = cpu.phase_mass();
        const double gpu_mass = gpu_host.phase_mass();
        const double mass_rel = (std::abs(cpu_mass) > 0.0)
            ? std::abs(gpu_mass - cpu_mass) / std::abs(cpu_mass)
            : std::abs(gpu_mass - cpu_mass);
        const bool mass_pass = mass_rel <= cfg.tolerance;
        pass = pass && mass_pass;
        std::cout << "phase_mass_cpu=" << cpu_mass
                  << " phase_mass_gpu=" << gpu_mass
                  << " rel_diff=" << mass_rel
                  << " status=" << (mass_pass ? "PASS" : "FAIL") << "\n";

        gpu.printPerformanceMetrics();
        gpu.printRooflineSummary();
        return pass ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }
}
