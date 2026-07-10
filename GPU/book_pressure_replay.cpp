#include "BookPressureTest.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Inputs {
    int lx = 0, ly = 0, lz = 0;
    std::vector<double> p, hh, rho, u, v, w, divergence;
};

Inputs read_inputs(const std::string& feature_path, const std::string& state_path)
{
    Inputs result;
    std::ifstream feature(feature_path, std::ios::binary);
    char magic[8]{};
    std::int32_t header[4]{};
    feature.read(magic, 8); feature.read(reinterpret_cast<char*>(header), sizeof(header));
    static constexpr char feature_magic[8] = {'P','I','N','N','F','2','\0','\0'};
    if (!feature || std::memcmp(magic, feature_magic, 8) != 0 || header[3] != 7) {
        throw std::runtime_error("invalid feature snapshot");
    }
    result.lx = header[0]; result.ly = header[1]; result.lz = header[2];
    const int n = result.lx * result.ly * result.lz;
    std::vector<float> fields(static_cast<std::size_t>(n) * 7);
    feature.read(reinterpret_cast<char*>(fields.data()), static_cast<std::streamsize>(fields.size() * sizeof(float)));
    if (!feature) throw std::runtime_error("truncated feature snapshot");
    result.u.resize(n); result.v.resize(n); result.w.resize(n); result.rho.resize(n); result.divergence.resize(n);
    for (int i = 0; i < n; ++i) {
        result.u[i] = fields[i]; result.v[i] = fields[n + i]; result.w[i] = fields[2 * n + i];
        result.rho[i] = fields[3 * n + i];
        result.divergence[i] = fields[6 * n + i];
    }

    std::ifstream state(state_path, std::ios::binary);
    std::int32_t dims[3]{};
    state.read(magic, 8); state.read(reinterpret_cast<char*>(dims), sizeof(dims));
    static constexpr char state_magic[8] = {'P','I','N','N','S','1','\0','\0'};
    if (!state || std::memcmp(magic, state_magic, 8) != 0 ||
        dims[0] != result.lx || dims[1] != result.ly || dims[2] != result.lz) {
        throw std::runtime_error("invalid or mismatched state snapshot");
    }
    result.p.resize(n); result.hh.resize(static_cast<std::size_t>(n) * 15);
    state.read(reinterpret_cast<char*>(result.p.data()), static_cast<std::streamsize>(result.p.size() * sizeof(double)));
    state.read(reinterpret_cast<char*>(result.hh.data()), static_cast<std::streamsize>(result.hh.size() * sizeof(double)));
    if (!state) throw std::runtime_error("truncated state snapshot");
    return result;
}

void write_state(const std::string& path, int lx, int ly, int lz, const BookPressureSolution& solution)
{
    std::ofstream out(path, std::ios::binary);
    static constexpr char magic[8] = {'P','I','N','N','S','1','\0','\0'};
    const std::int32_t dims[3] = {lx, ly, lz};
    out.write(magic, 8); out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
    out.write(reinterpret_cast<const char*>(solution.pressure.data()),
              static_cast<std::streamsize>(solution.pressure.size() * sizeof(double)));
    out.write(reinterpret_cast<const char*>(solution.hh.data()),
              static_cast<std::streamsize>(solution.hh.size() * sizeof(double)));
    if (!out) throw std::runtime_error("failed to write target state");
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "usage: book_pressure_replay PRE_FEATURE PRE_STATE OUT_STATE [MAX_ITERATIONS]\n";
            return 1;
        }
        const int max_iterations = argc == 5 ? std::stoi(argv[4]) : 1000;
        Inputs input = read_inputs(argv[1], argv[2]);
        BookPressureSolution solution = solveBookPressureGPU(
            input.lx, input.ly, input.lz, input.hh, input.p,
            input.rho, input.u, input.v, input.w, input.divergence, max_iterations, 100, 1.0e-3);
        write_state(argv[3], input.lx, input.ly, input.lz, solution);
        std::cout << "iterations=" << solution.iterations
                  << " converged=" << (solution.converged ? 1 : 0)
                  << " relative_error=" << solution.relative_error << "\n";
        return solution.converged ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
