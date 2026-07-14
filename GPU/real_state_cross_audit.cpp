#include "common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct State
{
    int nx = 0, ny = 0, nz = 0;
    std::uint32_t step = 0;
    std::uint64_t iteration = 0;
    std::vector<double> p, rho, ux, vy, wz, h;
};

template <typename T, std::size_t N>
void read_exact(std::ifstream& input, std::array<T, N>& values)
{
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(sizeof(T) * N));
    if (!input) throw std::runtime_error("truncated state header");
}

State read_state(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open state file");
    std::array<char, 8> magic{};
    std::array<std::uint32_t, 10> h32{};
    std::array<std::uint64_t, 3> h64{};
    std::array<double, 2> hf64{};
    read_exact(input, magic); read_exact(input, h32);
    read_exact(input, h64); read_exact(input, hf64);
    const std::array<char, 8> expected = {'C','L','B','M','K','0','1','\0'};
    if (magic != expected || h32[0] != 0x01020304u || h32[1] != 8u ||
        h32[6] != 15u || h32[8] != 6u || h32[9] != 1u || hf64[1] != 1.0) {
        throw std::runtime_error("invalid state contract");
    }
    State state;
    state.nx = static_cast<int>(h32[2]);
    state.ny = static_cast<int>(h32[3]);
    state.nz = static_cast<int>(h32[4]);
    state.step = h32[7];
    state.iteration = h64[0];
    const std::uint64_t cells64 =
        static_cast<std::uint64_t>(state.nx) * state.ny * state.nz;
    if (h32[5] != static_cast<std::uint32_t>(state.nz + 2) ||
        h64[1] != cells64 || h64[2] != 20 * cells64 ||
        cells64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("state dimensions mismatch");
    }
    const std::size_t cells = static_cast<std::size_t>(cells64);
    state.p.resize(cells); state.rho.resize(cells); state.ux.resize(cells);
    state.vy.resize(cells); state.wz.resize(cells); state.h.resize(15 * cells);
    auto read_values = [&](std::vector<double>& values) {
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(double)));
        if (!input) throw std::runtime_error("truncated state payload");
    };
    read_values(state.p); read_values(state.rho); read_values(state.ux);
    read_values(state.vy); read_values(state.wz); read_values(state.h);
    char extra = 0;
    if (input.read(&extra, 1)) throw std::runtime_error("trailing state bytes");
    return state;
}

std::vector<double> read_stage(
    const std::filesystem::path& path, std::size_t values)
{
    const auto expected_bytes = static_cast<std::uintmax_t>(values * sizeof(double));
    if (!std::filesystem::exists(path) ||
        std::filesystem::file_size(path) != expected_bytes) {
        throw std::runtime_error("stage size mismatch: " + path.string());
    }
    std::vector<double> result(values);
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(expected_bytes));
    if (!input) throw std::runtime_error("cannot read stage: " + path.string());
    return result;
}

double max_abs(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) throw std::runtime_error("stage vector mismatch");
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::abs(a[i] - b[i]));
    }
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " STATE.bin TORCH_STAGE_DIR\n";
        return 64;
    }
    try {
        const State state = read_state(argv[1]);
        const int nx = state.nx, ny = state.ny, nz = state.nz;
        const std::size_t cells = state.p.size();
        const std::size_t h_values = state.h.size();
        auto cell = [=](int x, int y, int z) {
            return (z * ny + y) * nx + x;
        };
        std::vector<double> collision(h_values), stream(h_values), boundary(h_values);
        std::vector<double> pressure_image(cells), r_h(h_values), r_p(cells);
        for (std::size_t c = 0; c < cells; ++c) {
            const double tau = 1.0 / state.rho[c] + 0.5;
            const double div = state.ux[c] + state.vy[c] + state.wz[c];
            for (int q = 0; q < 15; ++q) {
                const std::size_t idx = c * 15 + q;
                collision[idx] = state.h[idx] -
                    (state.h[idx] - D3Q15::Ei[q] * state.p[c]) / tau -
                    (D3Q15::Ei[q] / 3.0) * div;
            }
        }
        for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x) for (int q = 0; q < 15; ++q) {
            int xs = x - D3Q15::ex[q], ys = y - D3Q15::ey[q],
                zs = z - D3Q15::ez[q];
            if (xs < 0) xs += nx; else if (xs >= nx) xs -= nx;
            if (ys < 0) ys += ny; else if (ys >= ny) ys -= ny;
            if (zs < 0) zs += nz; else if (zs >= nz) zs -= nz;
            stream[static_cast<std::size_t>(cell(x,y,z)) * 15 + q] =
                collision[static_cast<std::size_t>(cell(xs,ys,zs)) * 15 + q];
        }
        boundary = stream;
        constexpr int bounce[5][2] = {{1,4},{7,8},{9,14},{10,13},{12,11}};
        for (int z = 0; z < nz; ++z) for (int y = 0; y < ny; ++y) {
            const std::size_t left = static_cast<std::size_t>(cell(0,y,z)) * 15;
            const std::size_t right = static_cast<std::size_t>(cell(nx-1,y,z)) * 15;
            for (const auto& pair : bounce) {
                boundary[left + pair[0]] = boundary[left + pair[1]];
                boundary[right + pair[1]] = boundary[right + pair[0]];
            }
        }
        for (std::size_t c = 0; c < cells; ++c) {
            double image_sum = boundary[c * 15];
            double live_sum = state.h[c * 15];
            for (int q = 1; q < 15; ++q) {
                image_sum += boundary[c * 15 + q];
                live_sum += state.h[c * 15 + q];
            }
            pressure_image[c] = image_sum;
            r_p[c] = state.p[c] - live_sum;
            for (int q = 0; q < 15; ++q) {
                r_h[c * 15 + q] = state.h[c * 15 + q] - boundary[c * 15 + q];
            }
        }

        std::ostringstream prefix;
        prefix << "step" << std::setw(4) << std::setfill('0') << state.step
               << "_iter" << std::setw(8) << std::setfill('0') << state.iteration;
        const std::filesystem::path directory = argv[2];
        auto torch_stage = [&](const std::string& name, std::size_t values) {
            return read_stage(directory / (prefix.str() + "_" + name + ".bin"), values);
        };
        const std::array<std::pair<std::string, double>, 6> errors = {{
            {"collision_max_abs", max_abs(collision, torch_stage("collision", h_values))},
            {"stream_max_abs", max_abs(stream, torch_stage("stream", h_values))},
            {"boundary_max_abs", max_abs(boundary, torch_stage("boundary", h_values))},
            {"pressure_image_max_abs", max_abs(pressure_image, torch_stage("pressure_image", cells))},
            {"r_h_max_abs", max_abs(r_h, torch_stage("r_h", h_values))},
            {"r_p_max_abs", max_abs(r_p, torch_stage("r_p", cells))},
        }};
        double worst = 0.0;
        std::cout << "state=" << std::filesystem::absolute(argv[1]).string() << '\n'
                  << "step=" << state.step << '\n'
                  << "iteration=" << state.iteration << '\n'
                  << std::scientific << std::setprecision(17);
        for (const auto& item : errors) {
            std::cout << item.first << '=' << item.second << '\n';
            worst = std::max(worst, item.second);
        }
        const bool pass = std::isfinite(worst) && worst <= 1.0e-12;
        std::cout << "worst_max_abs=" << worst << '\n'
                  << "tolerance=1.00000000000000000e-12\n"
                  << "pass=" << (pass ? 1 : 0) << '\n';
        return pass ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "real-state cross-audit failed: " << error.what() << '\n';
        return 1;
    }
}
