#include "BookPressureTest.hpp"
#include "common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int cell_index(int x, int y, int z, int lx, int ly)
{
    return (z * ly + y) * lx + x;
}

BookPressureStages run_cpu(
    int lx, int ly, int lz,
    const std::vector<double>& hh,
    const std::vector<double>& pressure,
    const std::vector<double>& rho,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w)
{
    const int n = lx * ly * lz;
    BookPressureStages out;
    out.divergence.assign(n, 0.0);
    out.collision.resize(static_cast<std::size_t>(n) * 15);
    out.streamed.assign(static_cast<std::size_t>(n) * 15, 0.0);
    out.bounced.resize(static_cast<std::size_t>(n) * 15);
    out.pressure.assign(n, 0.0);

    for (int z = 0; z < lz; ++z) {
        for (int y = 0; y < ly; ++y) {
            for (int x = 0; x < lx; ++x) {
                double du_dx = 0.0, dv_dy = 0.0, dw_dz = 0.0;
                for (int q = 1; q < 15; ++q) {
                    int xp = x + D3Q15::ex[q];
                    int yp = y + D3Q15::ey[q];
                    int zp = z + D3Q15::ez[q];
                    if (xp >= lx) xp = lx - 2;
                    if (xp < 0) xp = 1;
                    if (yp >= ly) yp = 0;
                    if (yp < 0) yp = ly - 1;
                    if (zp >= lz) zp = 0;
                    if (zp < 0) zp = lz - 1;
                    const int src = cell_index(xp, yp, zp, lx, ly);
                    du_dx += u[src] * D3Q15::uc[q];
                    dv_dy += v[src] * D3Q15::vc[q];
                    dw_dz += w[src] * D3Q15::wc[q];
                }
                const int cell = cell_index(x, y, z, lx, ly);
                out.divergence[cell] = (du_dx + dv_dy + dw_dz) / 10.0;
                const double tauh = 1.0 / rho[cell] + 0.5;
                for (int q = 0; q < 15; ++q) {
                    const int id = cell * 15 + q;
                    const double hequ = D3Q15::Ei[q] * pressure[cell];
                    out.collision[id] = hh[id] - (hh[id] - hequ) / tauh
                        - (D3Q15::Ei[q] / 3.0) * out.divergence[cell];
                }
            }
        }
    }

    for (int z = 0; z < lz; ++z) {
        for (int y = 0; y < ly; ++y) {
            for (int x = 0; x < lx; ++x) {
                const int src = cell_index(x, y, z, lx, ly);
                for (int q = 0; q < 15; ++q) {
                    int xd = x + D3Q15::ex[q];
                    int yd = y + D3Q15::ey[q];
                    int zd = z + D3Q15::ez[q];
                    if (xd >= lx) xd = 0;
                    if (xd < 0) xd = lx - 1;
                    if (yd >= ly) yd = 0;
                    if (yd < 0) yd = ly - 1;
                    if (zd >= lz) zd = 0;
                    if (zd < 0) zd = lz - 1;
                    const int dst = cell_index(xd, yd, zd, lx, ly);
                    out.streamed[dst * 15 + q] = out.collision[src * 15 + q];
                }
            }
        }
    }
    out.bounced = out.streamed;
    for (int z = 0; z < lz; ++z) {
        for (int y = 0; y < ly; ++y) {
            const int left = cell_index(0, y, z, lx, ly) * 15;
            out.bounced[left + 1] = out.streamed[left + 4];
            out.bounced[left + 7] = out.streamed[left + 8];
            out.bounced[left + 9] = out.streamed[left + 14];
            out.bounced[left + 10] = out.streamed[left + 13];
            out.bounced[left + 12] = out.streamed[left + 11];
            const int right = cell_index(lx - 1, y, z, lx, ly) * 15;
            out.bounced[right + 4] = out.streamed[right + 1];
            out.bounced[right + 8] = out.streamed[right + 7];
            out.bounced[right + 14] = out.streamed[right + 9];
            out.bounced[right + 13] = out.streamed[right + 10];
            out.bounced[right + 11] = out.streamed[right + 12];
        }
    }
    for (int cell = 0; cell < n; ++cell) {
        for (int q = 0; q < 15; ++q) out.pressure[cell] += out.bounced[cell * 15 + q];
    }
    return out;
}

double max_error(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) throw std::runtime_error("stage size mismatch");
    double value = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) value = std::max(value, std::abs(a[i] - b[i]));
    return value;
}

void write_vector(std::ofstream& out, const std::vector<double>& values)
{
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
}

void write_stages(std::ofstream& out, const BookPressureStages& stages)
{
    write_vector(out, stages.divergence);
    write_vector(out, stages.collision);
    write_vector(out, stages.streamed);
    write_vector(out, stages.bounced);
    write_vector(out, stages.pressure);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const int lx = 6, ly = 5, lz = 4;
        const int n = lx * ly * lz;
        std::vector<double> hh(static_cast<std::size_t>(n) * 15);
        std::vector<double> pressure(n, 0.0), rho(n), u(n), v(n), w(n);
        for (int z = 0; z < lz; ++z) {
            for (int y = 0; y < ly; ++y) {
                for (int x = 0; x < lx; ++x) {
                    const int cell = cell_index(x, y, z, lx, ly);
                    rho[cell] = 1.0 + static_cast<double>((17 * x + 11 * y + 5 * z) % 37) / 10.0;
                    u[cell] = 0.01 * (x + 2.0 * y - 3.0 * z);
                    v[cell] = -0.015 * (2.0 * x - y + z);
                    w[cell] = 0.02 * (-x + 3.0 * y + 2.0 * z);
                    const double base_p = 0.2 + 0.003 * x - 0.002 * y + 0.004 * z;
                    for (int q = 0; q < 15; ++q) {
                        const double value = D3Q15::Ei[q] * base_p
                            + 1.0e-4 * (q + 1.0) * (1.0 + x + 2.0 * y + 3.0 * z);
                        hh[cell * 15 + q] = value;
                        pressure[cell] += value;
                    }
                }
            }
        }

        const BookPressureStages cpu = run_cpu(lx, ly, lz, hh, pressure, rho, u, v, w);
        const BookPressureStages gpu = runBookPressureStagesGPU(lx, ly, lz, hh, pressure, rho, u, v, w);
        const std::vector<std::pair<std::string, double>> errors = {
            {"firstord_div", max_error(cpu.divergence, gpu.divergence)},
            {"collision", max_error(cpu.collision, gpu.collision)},
            {"stream", max_error(cpu.streamed, gpu.streamed)},
            {"slip_bounceback", max_error(cpu.bounced, gpu.bounced)},
            {"getp", max_error(cpu.pressure, gpu.pressure)},
        };
        bool pass = true;
        std::cout << std::scientific << std::setprecision(12);
        for (const auto& [stage, error] : errors) {
            const bool stage_pass = error <= 1.0e-12;
            pass = pass && stage_pass;
            std::cout << stage << "," << error << "," << (stage_pass ? "PASS" : "FAIL") << "\n";
        }

        if (argc > 1) {
            std::ofstream out(argv[1], std::ios::binary);
            if (!out) throw std::runtime_error("failed to open stage dump");
            static constexpr char magic[8] = {'B', 'O', 'O', 'K', 'T', '1', '\0', '\0'};
            const std::int32_t dims[3] = {lx, ly, lz};
            out.write(magic, sizeof(magic));
            out.write(reinterpret_cast<const char*>(dims), sizeof(dims));
            write_stages(out, cpu);
            write_stages(out, gpu);
            if (!out) throw std::runtime_error("failed to write stage dump");
        }
        return pass ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
