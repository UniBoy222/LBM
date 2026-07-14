#include "Inamuro.hpp"
#include "InamuroCUDA.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct State
{
    int nx = 0, ny = 0, nz = 0;
    std::vector<double> p, u, v, w, rho, fei;
};

struct ShapeStats
{
    std::size_t liquid_voxels = 0;
    std::size_t component_count = 0;
    std::size_t largest_component = 0;
    std::array<double, 3> centroid = {0.0, 0.0, 0.0};
};

struct Comparison
{
    double p_raw_rel = 0.0;
    double p_demeaned_rel = 0.0;
    double p_mean = 0.0;
    double p_ref_mean = 0.0;
    double p_mean_abs_diff = 0.0;
    double pressure_gradient_rel = 0.0;
    double pressure_correction_rel = 0.0;
    double velocity_rel = 0.0;
    double u_rel = 0.0, v_rel = 0.0, w_rel = 0.0;
    double rho_rel = 0.0, fei_rel = 0.0;
    double mass_rel = 0.0;
    double shape_mismatch_fraction = 0.0;
    double shape_dice_error = 0.0;
    std::size_t candidate_components = 0, reference_components = 0;
    double largest_component_rel = 0.0;
    double centroid_distance_normalized = 0.0;
    double h_moment_rel = 0.0;
    double h_p_consistency_max_abs = 0.0;
    double h_equilibrium_max_abs = 0.0;
    double max_field_rel = 0.0;
    bool finite = true;
    bool bitwise_equal = false;
};

class OracleHost final : public Inamuro
{
public:
    using Inamuro::Inamuro;

    State capture() const
    {
        State state;
        state.nx = lx; state.ny = ly; state.nz = lz;
        const std::size_t n = static_cast<std::size_t>(lx) * ly * lz;
        state.p.reserve(n); state.u.reserve(n); state.v.reserve(n);
        state.w.reserve(n); state.rho.reserve(n); state.fei.reserve(n);
        for (int z = 0; z < lz; ++z)
            for (int y = 0; y < ly; ++y)
                for (int x = 0; x < lx; ++x) {
                    state.p.push_back(p[x][y][z + 1]);
                    state.u.push_back(u[x][y][z + 1]);
                    state.v.push_back(v[x][y][z + 1]);
                    state.w.push_back(w[x][y][z + 1]);
                    state.rho.push_back(rho[x][y][z + 1]);
                    state.fei.push_back(fei[x][y][z + 1]);
                }
        return state;
    }

    double phase_threshold() const { return 0.5 * (params.fei_L + params.fei_G); }
};

double milliseconds(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void write_header(std::ofstream& out, const char magic[8], const State& state,
                  std::uint32_t step, std::uint32_t fields)
{
    const std::uint32_t header[5] = {
        static_cast<std::uint32_t>(state.nx), static_cast<std::uint32_t>(state.ny),
        static_cast<std::uint32_t>(state.nz), step, fields};
    out.write(magic, 8);
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
}

void write_values(std::ofstream& out, const std::vector<double>& values)
{
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
}

void write_reference(const std::filesystem::path& path, const State& state, int step)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write reference state: " + path.string());
    const char magic[8] = {'C', 'L', 'B', 'M', 'R', 'F', '1', '\0'};
    write_header(out, magic, state, static_cast<std::uint32_t>(step), 6);
    write_values(out, state.p); write_values(out, state.u); write_values(out, state.v);
    write_values(out, state.w); write_values(out, state.rho); write_values(out, state.fei);
    if (!out) throw std::runtime_error("failed writing reference state: " + path.string());
}

std::array<std::uint32_t, 5> read_header(std::ifstream& in, const char expected[8])
{
    char magic[8]{};
    std::array<std::uint32_t, 5> header{};
    in.read(magic, 8);
    in.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (!in || std::memcmp(magic, expected, 8) != 0) throw std::runtime_error("bad/truncated Oracle file header");
    return header;
}

State read_reference(const std::filesystem::path& path, int nx, int ny, int nz, int step)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read reference state: " + path.string());
    const char magic[8] = {'C', 'L', 'B', 'M', 'R', 'F', '1', '\0'};
    const auto h = read_header(in, magic);
    if (h[0] != static_cast<std::uint32_t>(nx) || h[1] != static_cast<std::uint32_t>(ny) ||
        h[2] != static_cast<std::uint32_t>(nz) || h[3] != static_cast<std::uint32_t>(step) || h[4] != 6)
        throw std::runtime_error("reference state metadata mismatch");
    State state; state.nx = nx; state.ny = ny; state.nz = nz;
    const std::size_t n = static_cast<std::size_t>(nx) * ny * nz;
    auto read = [&](std::vector<double>& values) {
        values.resize(n);
        in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(n * sizeof(double)));
    };
    read(state.p); read(state.u); read(state.v); read(state.w); read(state.rho); read(state.fei);
    if (!in || in.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("bad reference payload");
    return state;
}

double mean(const std::vector<double>& values)
{
    long double sum = 0.0;
    for (double value : values) sum += value;
    return static_cast<double>(sum / values.size());
}

double rel_l2(const std::vector<double>& candidate, const std::vector<double>& reference,
              bool demean = false)
{
    if (candidate.size() != reference.size()) throw std::runtime_error("field size mismatch");
    const double cm = demean ? mean(candidate) : 0.0;
    const double rm = demean ? mean(reference) : 0.0;
    long double diff2 = 0.0, ref2 = 0.0;
    for (std::size_t i = 0; i < candidate.size(); ++i) {
        const long double a = candidate[i] - cm;
        const long double b = reference[i] - rm;
        const long double d = a - b;
        diff2 += d * d; ref2 += b * b;
    }
    return ref2 > 0.0 ? std::sqrt(static_cast<double>(diff2 / ref2)) : std::sqrt(static_cast<double>(diff2));
}

double vector_rel_l2(const std::array<const std::vector<double>*, 3>& candidate,
                     const std::array<const std::vector<double>*, 3>& reference)
{
    long double diff2 = 0.0, ref2 = 0.0;
    for (int component = 0; component < 3; ++component)
        for (std::size_t i = 0; i < candidate[component]->size(); ++i) {
            const long double a = (*candidate[component])[i];
            const long double b = (*reference[component])[i];
            const long double d = a - b;
            diff2 += d * d; ref2 += b * b;
        }
    return ref2 > 0.0 ? std::sqrt(static_cast<double>(diff2 / ref2)) : std::sqrt(static_cast<double>(diff2));
}

std::array<std::vector<double>, 3> pressure_correction(const State& state)
{
    const std::size_t n = state.p.size();
    std::array<std::vector<double>, 3> result = {std::vector<double>(n), std::vector<double>(n), std::vector<double>(n)};
    auto idx = [&](int x, int y, int z) { return static_cast<std::size_t>((z * state.ny + y) * state.nx + x); };
    for (int z = 0; z < state.nz; ++z)
        for (int y = 0; y < state.ny; ++y)
            for (int x = 0; x < state.nx; ++x) {
                const int xe = x == state.nx - 1 ? state.nx - 2 : x + 1;
                const int xw = x == 0 ? 1 : x - 1;
                const int yn = y == state.ny - 1 ? 0 : y + 1;
                const int ys = y == 0 ? state.ny - 1 : y - 1;
                const int zn = z == state.nz - 1 ? 0 : z + 1;
                const int zs = z == 0 ? state.nz - 1 : z - 1;
                const std::size_t c = idx(x, y, z);
                const double scale = 1.0 / (2.0 * state.rho[c]);
                result[0][c] = (state.p[idx(xe, y, z)] - state.p[idx(xw, y, z)]) * scale;
                result[1][c] = (state.p[idx(x, yn, z)] - state.p[idx(x, ys, z)]) * scale;
                result[2][c] = (state.p[idx(x, y, zn)] - state.p[idx(x, y, zs)]) * scale;
            }
    return result;
}

std::array<std::vector<double>, 3> pressure_gradient(const State& state)
{
    const std::size_t n = state.p.size();
    std::array<std::vector<double>, 3> result = {
        std::vector<double>(n), std::vector<double>(n), std::vector<double>(n)};
    auto idx = [&](int x, int y, int z) {
        return static_cast<std::size_t>((z * state.ny + y) * state.nx + x);
    };
    for (int z = 0; z < state.nz; ++z)
        for (int y = 0; y < state.ny; ++y)
            for (int x = 0; x < state.nx; ++x) {
                const int xe = x == state.nx - 1 ? state.nx - 2 : x + 1;
                const int xw = x == 0 ? 1 : x - 1;
                const int yn = y == state.ny - 1 ? 0 : y + 1;
                const int ys = y == 0 ? state.ny - 1 : y - 1;
                const int zn = z == state.nz - 1 ? 0 : z + 1;
                const int zs = z == 0 ? state.nz - 1 : z - 1;
                const std::size_t c = idx(x, y, z);
                result[0][c] = 0.5 * (state.p[idx(xe, y, z)] - state.p[idx(xw, y, z)]);
                result[1][c] = 0.5 * (state.p[idx(x, yn, z)] - state.p[idx(x, ys, z)]);
                result[2][c] = 0.5 * (state.p[idx(x, y, zn)] - state.p[idx(x, y, zs)]);
            }
    return result;
}

ShapeStats shape_stats(const State& state, double threshold)
{
    const std::size_t n = state.fei.size();
    std::vector<unsigned char> seen(n, 0);
    ShapeStats stats;
    auto idx = [&](int x, int y, int z) { return static_cast<std::size_t>((z * state.ny + y) * state.nx + x); };
    for (double value : state.fei) if (value >= threshold) ++stats.liquid_voxels;
    for (int z0 = 0; z0 < state.nz; ++z0)
        for (int y0 = 0; y0 < state.ny; ++y0)
            for (int x0 = 0; x0 < state.nx; ++x0) {
                const std::size_t start = idx(x0, y0, z0);
                if (seen[start] || state.fei[start] < threshold) continue;
                ++stats.component_count;
                std::queue<std::array<int, 3>> queue;
                queue.push({x0, y0, z0}); seen[start] = 1;
                std::size_t size = 0; long double sx = 0.0, sy = 0.0, sz = 0.0;
                while (!queue.empty()) {
                    const auto xyz = queue.front(); queue.pop();
                    const int x = xyz[0], y = xyz[1], z = xyz[2];
                    ++size; sx += x; sy += y; sz += z;
                    const std::array<std::array<int, 3>, 6> neighbors = {{{x - 1, y, z}, {x + 1, y, z},
                        {x, (y + state.ny - 1) % state.ny, z}, {x, (y + 1) % state.ny, z},
                        {x, y, (z + state.nz - 1) % state.nz}, {x, y, (z + 1) % state.nz}}};
                    for (const auto& next : neighbors) {
                        if (next[0] < 0 || next[0] >= state.nx) continue;
                        const std::size_t ni = idx(next[0], next[1], next[2]);
                        if (!seen[ni] && state.fei[ni] >= threshold) {
                            seen[ni] = 1; queue.push(next);
                        }
                    }
                }
                if (size > stats.largest_component) {
                    stats.largest_component = size;
                    stats.centroid = {static_cast<double>(sx / size), static_cast<double>(sy / size), static_cast<double>(sz / size)};
                }
            }
    return stats;
}

bool all_finite(const State& state, const std::vector<double>& hh)
{
    const std::array<const std::vector<double>*, 6> fields = {&state.p, &state.u, &state.v, &state.w, &state.rho, &state.fei};
    for (const auto* field : fields) for (double value : *field) if (!std::isfinite(value)) return false;
    for (double value : hh) if (!std::isfinite(value)) return false;
    return true;
}

bool state_bitwise_equal(const State& a, const State& b)
{
    const std::array<const std::vector<double>*, 6> aa = {&a.p, &a.u, &a.v, &a.w, &a.rho, &a.fei};
    const std::array<const std::vector<double>*, 6> bb = {&b.p, &b.u, &b.v, &b.w, &b.rho, &b.fei};
    for (int i = 0; i < 6; ++i) {
        if (aa[i]->size() != bb[i]->size() ||
            std::memcmp(aa[i]->data(), bb[i]->data(), aa[i]->size() * sizeof(double)) != 0) return false;
    }
    return true;
}

Comparison compare(const State& candidate, const State& reference,
                   const std::vector<double>& hh, double threshold)
{
    Comparison out;
    out.finite = all_finite(candidate, hh);
    if (!out.finite) throw std::runtime_error("candidate state contains NaN/Inf");
    out.p_raw_rel = rel_l2(candidate.p, reference.p);
    out.p_demeaned_rel = rel_l2(candidate.p, reference.p, true);
    out.p_mean = mean(candidate.p); out.p_ref_mean = mean(reference.p);
    out.p_mean_abs_diff = std::abs(out.p_mean - out.p_ref_mean);
    const auto candidate_pg = pressure_gradient(candidate);
    const auto reference_pg = pressure_gradient(reference);
    out.pressure_gradient_rel = vector_rel_l2(
        {&candidate_pg[0], &candidate_pg[1], &candidate_pg[2]},
        {&reference_pg[0], &reference_pg[1], &reference_pg[2]});
    const auto candidate_pc = pressure_correction(candidate);
    const auto reference_pc = pressure_correction(reference);
    out.pressure_correction_rel = vector_rel_l2(
        {&candidate_pc[0], &candidate_pc[1], &candidate_pc[2]},
        {&reference_pc[0], &reference_pc[1], &reference_pc[2]});
    out.u_rel = rel_l2(candidate.u, reference.u); out.v_rel = rel_l2(candidate.v, reference.v);
    out.w_rel = rel_l2(candidate.w, reference.w);
    out.velocity_rel = vector_rel_l2({&candidate.u, &candidate.v, &candidate.w},
                                     {&reference.u, &reference.v, &reference.w});
    out.rho_rel = rel_l2(candidate.rho, reference.rho);
    out.fei_rel = rel_l2(candidate.fei, reference.fei);
    long double candidate_mass = 0.0, reference_mass = 0.0;
    for (double value : candidate.fei) candidate_mass += value;
    for (double value : reference.fei) reference_mass += value;
    out.mass_rel = reference_mass != 0.0 ? std::abs(static_cast<double>((candidate_mass - reference_mass) / reference_mass))
                                         : std::abs(static_cast<double>(candidate_mass));

    std::size_t mismatch = 0, intersection = 0, candidate_liquid = 0, reference_liquid = 0;
    for (std::size_t i = 0; i < candidate.fei.size(); ++i) {
        const bool ca = candidate.fei[i] >= threshold, re = reference.fei[i] >= threshold;
        mismatch += ca != re; intersection += ca && re; candidate_liquid += ca; reference_liquid += re;
    }
    out.shape_mismatch_fraction = static_cast<double>(mismatch) / candidate.fei.size();
    const std::size_t dice_denominator = candidate_liquid + reference_liquid;
    out.shape_dice_error = dice_denominator ? 1.0 - 2.0 * intersection / static_cast<double>(dice_denominator) : 0.0;
    const ShapeStats cs = shape_stats(candidate, threshold), rs = shape_stats(reference, threshold);
    out.candidate_components = cs.component_count; out.reference_components = rs.component_count;
    out.largest_component_rel = rs.largest_component
        ? std::abs(static_cast<double>(cs.largest_component) - rs.largest_component) / rs.largest_component
        : static_cast<double>(cs.largest_component);
    const double dx = cs.centroid[0] - rs.centroid[0], dy = cs.centroid[1] - rs.centroid[1], dz = cs.centroid[2] - rs.centroid[2];
    const double diagonal = std::sqrt(static_cast<double>(candidate.nx * candidate.nx + candidate.ny * candidate.ny + candidate.nz * candidate.nz));
    out.centroid_distance_normalized = std::sqrt(dx * dx + dy * dy + dz * dz) / diagonal;

    if (hh.size() != candidate.p.size() * D3Q15::Q) throw std::runtime_error("hh size mismatch");
    long double moment_diff2 = 0.0, moment_ref2 = 0.0;
    for (std::size_t cell = 0; cell < candidate.p.size(); ++cell) {
        std::array<long double, 10> cm{}, rm{};
        for (int q = 0; q < D3Q15::Q; ++q) {
            const double hc = hh[cell * D3Q15::Q + q];
            const double hr = D3Q15::Ei[q] * reference.p[cell];
            const std::array<double, 10> basis = {1.0, static_cast<double>(D3Q15::ex[q]), static_cast<double>(D3Q15::ey[q]),
                static_cast<double>(D3Q15::ez[q]), static_cast<double>(D3Q15::ex[q] * D3Q15::ex[q]),
                static_cast<double>(D3Q15::ey[q] * D3Q15::ey[q]), static_cast<double>(D3Q15::ez[q] * D3Q15::ez[q]),
                static_cast<double>(D3Q15::ex[q] * D3Q15::ey[q]), static_cast<double>(D3Q15::ex[q] * D3Q15::ez[q]),
                static_cast<double>(D3Q15::ey[q] * D3Q15::ez[q])};
            for (int m = 0; m < 10; ++m) { cm[m] += hc * basis[m]; rm[m] += hr * basis[m]; }
            out.h_equilibrium_max_abs = std::max(out.h_equilibrium_max_abs,
                                                  std::abs(hc - D3Q15::Ei[q] * candidate.p[cell]));
        }
        out.h_p_consistency_max_abs = std::max(out.h_p_consistency_max_abs,
                                               std::abs(static_cast<double>(cm[0]) - candidate.p[cell]));
        for (int m = 0; m < 10; ++m) {
            const long double d = cm[m] - rm[m]; moment_diff2 += d * d; moment_ref2 += rm[m] * rm[m];
        }
    }
    out.h_moment_rel = moment_ref2 > 0.0 ? std::sqrt(static_cast<double>(moment_diff2 / moment_ref2))
                                         : std::sqrt(static_cast<double>(moment_diff2));
    out.max_field_rel = std::max({out.p_raw_rel, out.p_demeaned_rel, out.pressure_gradient_rel,
        out.pressure_correction_rel,
        out.velocity_rel, out.u_rel, out.v_rel, out.w_rel, out.rho_rel, out.fei_rel, out.mass_rel,
        out.shape_mismatch_fraction, out.shape_dice_error, out.largest_component_rel,
        out.centroid_distance_normalized, out.h_moment_rel});
    if (out.candidate_components != out.reference_components) out.max_field_rel = std::max(out.max_field_rel, 1.0);
    out.bitwise_equal = state_bitwise_equal(candidate, reference);
    return out;
}

std::string step_name(const char* prefix, int step)
{
    std::ostringstream out;
    out << prefix << std::setw(4) << std::setfill('0') << step << ".bin";
    return out.str();
}

struct Config
{
    std::string route;
    std::filesystem::path params, output, baseline;
    int steps = 20;
    bool benchmark = false;
};

Config parse_args(int argc, char** argv)
{
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() { if (++i >= argc) throw std::runtime_error("missing value for " + arg); return std::string(argv[i]); };
        if (arg == "--route") cfg.route = value();
        else if (arg == "--params") cfg.params = value();
        else if (arg == "--output") cfg.output = value();
        else if (arg == "--baseline-dir") cfg.baseline = value();
        else if (arg == "--steps") cfg.steps = std::stoi(value());
        else if (arg == "--benchmark") cfg.benchmark = true;
        else throw std::runtime_error("unknown argument: " + arg);
    }
    if (cfg.params.empty() || cfg.output.empty() || cfg.steps <= 0 || cfg.steps > 20)
        throw std::runtime_error("invalid --params/--output/--steps");
    if (cfg.route != "baseline" && cfg.route != "capture" && cfg.route != "replay")
        throw std::runtime_error("--route must be baseline, capture or replay");
    if (cfg.route == "replay" && cfg.baseline.empty())
        throw std::runtime_error("replay requires --baseline-dir");
    if (cfg.route != "replay" && !cfg.baseline.empty())
        throw std::runtime_error("only replay may receive --baseline-dir");
    if (cfg.benchmark && cfg.route == "capture")
        throw std::runtime_error("benchmark mode forbids capture");
    return cfg;
}

void write_csv_header(std::ofstream& out)
{
    out << "step,route,iterations,reference_iterations,tail_iteration,residual,pressure_l1_delta,pressure_l1_norm,"
           "fixed_point_h_relative,fixed_point_p_relative,fixed_point_relative,pressure_converged,"
           "fixed_point_converged,dual_residual_enabled,converged,fallback,warm_start_used,"
           "warm_start_first_check_converged,warm_start_io_ms,warm_start_restore_max_abs,"
           "compute_ms,e2e_ms,validation_io_ms,reference_write_io_ms,"
           "source_mean,source_abs_mean,projected_source_mean,pressure_gauge_target,pressure_gauge_max_shift,"
           "p_mean,p_ref_mean,p_mean_abs_diff,p_raw_rel_l2,p_demeaned_rel_l2,"
           "pressure_gradient_rel_l2,pressure_correction_rel_l2,velocity_rel_l2,u_rel_l2,v_rel_l2,w_rel_l2,"
           "rho_rel_l2,fei_rel_l2,mass_rel,"
           "shape_mismatch_fraction,shape_dice_error,candidate_components,reference_components,largest_component_rel,"
           "centroid_distance_normalized,h_moment_rel_l2,h_p_consistency_max_abs,h_equilibrium_max_abs,max_field_rel,"
           "finite,state_bitwise_equal\n";
}

bool diagnostics_finite(const InamuroCUDA::PoissonStepDiagnostics& d)
{
    const std::array<double, 16> values = {
        d.relative_error, d.pressure_l1_delta, d.pressure_l1_norm,
        d.fixed_point_h_relative, d.fixed_point_p_relative, d.fixed_point_relative,
        d.source_mean, d.source_abs_mean, d.projected_source_mean,
        d.pressure_gauge_target, d.pressure_gauge_max_shift,
        d.warm_start_io_ms, d.warm_start_restore_max_abs,
        d.fixed_point_h_l1, d.fixed_point_h_scale, d.fixed_point_p_l1};
    return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
}

void write_protocol(const Config& cfg)
{
    std::ofstream out(cfg.output / "FROZEN_PROTOCOL.txt", std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write FROZEN_PROTOCOL.txt");
    out << "route=" << cfg.route << '\n'
        << "benchmark=" << (cfg.benchmark ? 1 : 0) << '\n'
        << "steps=" << cfg.steps << '\n'
        << "poisson=split\npressure_boundary=split\n"
        << "dual_residual=1\ndeterministic_reductions=1\n"
        << "check_interval=100\ntolerance=1e-3\niteration_limit=0\n"
        << "fused=0\none_pass=0\nscalar=0\nsource_aware_hh_init=0\n"
        << "pressure_relax_scale=1\nfixed_point_relax=1\nanderson=0\ntwo_grid=0\ngraph=0\n"
        << "replay_field_relative_gate=1e-10\nreplay_required_steps=20\n";
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Config cfg = parse_args(argc, argv);
        std::filesystem::create_directories(cfg.output);
        write_protocol(cfg);
        if (cfg.route == "capture") {
            std::filesystem::create_directories(cfg.output / "tail_hneq_states");
            std::filesystem::create_directories(cfg.output / "reference_states");
        }
        OracleHost host(cfg.params.string());
        int nx = 0, ny = 0, nz = 0; host.getGridSize(nx, ny, nz);
        InamuroCUDA gpu(host);
        gpu.setUseFusedPoisson(false);
        gpu.setUseOnePassPoisson(false);
        gpu.setUseScalarPoisson(false);
        gpu.setUseSourceAwareHHInit(false);
        gpu.setPressureRelaxScale(1.0);
        gpu.setPoissonFixedPointRelax(1.0);
        gpu.setUsePoissonAndersonM1(false);
        gpu.setUsePoissonTwoGridCorrection(false);
        gpu.setUseFusedBoundaryPressure(false);
        gpu.setUsePoissonGraph(false);
        gpu.setUsePoissonDeterministicReductions(true);
        gpu.setUsePoissonDualResidual(true);
        gpu.setPoissonConvergence(100, 1.0e-3);
        gpu.setPoissonIterationLimit(0);
        if (!cfg.benchmark)
            gpu.setPoissonDiagnosticsPath((cfg.output / "poisson_diagnostics.csv").string());
        if (cfg.route == "capture")
            gpu.setTailHneqCaptureDirectory((cfg.output / "tail_hneq_states").string());
        else if (cfg.route == "replay")
            gpu.setTailHneqReplayDirectory((cfg.baseline / "tail_hneq_states").string());

        std::ofstream csv(cfg.output / "steps.csv", std::ios::trunc);
        if (!csv) throw std::runtime_error("cannot open steps.csv");
        csv << std::scientific << std::setprecision(17); write_csv_header(csv);
        int passed_steps = 0;
        std::uint64_t total_iterations = 0;
        double max_field_rel = 0.0;
        double max_restore_abs = 0.0;
        bool all_first_check = true;
        for (int step = 1; step <= cfg.steps; ++step) {
            const auto e2e_start = Clock::now();
            const auto compute_start = Clock::now();
            gpu.performTimeStepGPU();
            gpu.synchronize();
            const auto compute_end = Clock::now();
            const double compute_ms = milliseconds(compute_start, compute_end);
            const double e2e_ms = milliseconds(e2e_start, compute_end);
            const auto diagnostics = gpu.getLastPoissonDiagnostics();
            if (!diagnostics_finite(diagnostics) || !diagnostics.converged ||
                !diagnostics.pressure_converged || !diagnostics.fixed_point_converged ||
                !diagnostics.dual_residual_enabled || diagnostics.fallback_count != 0 ||
                !(diagnostics.relative_error < 1.0e-3) ||
                !(diagnostics.fixed_point_relative < 1.0e-3))
                throw std::runtime_error("strict convergence contract failed at step " + std::to_string(step));

            State state, reference;
            std::vector<double> hh;
            Comparison result;
            double validation_io_ms = 0.0, reference_write_io_ms = 0.0;
            if (!cfg.benchmark) {
                gpu.downloadFieldsToCPU(host);
                gpu.downloadHH(hh);
                state = host.capture();
                if (!all_finite(state, hh)) throw std::runtime_error("NaN/Inf state at step " + std::to_string(step));
                reference = state;
                if (cfg.route == "capture") {
                    const auto write_start = Clock::now();
                    write_reference(cfg.output / "reference_states" / step_name("state_", step), state, step);
                    reference_write_io_ms = milliseconds(write_start, Clock::now());
                } else if (cfg.route == "replay") {
                    const auto read_start = Clock::now();
                    reference = read_reference(cfg.baseline / "reference_states" / step_name("state_", step), nx, ny, nz, step);
                    validation_io_ms = milliseconds(read_start, Clock::now());
                }
                result = compare(state, reference, hh, host.phase_threshold());
            }
            bool step_pass = true;
            if (cfg.route == "replay") {
                step_pass = diagnostics.warm_start_used && diagnostics.warm_start_first_check_converged &&
                    diagnostics.iterations == 100 && diagnostics.reference_iterations >= 100 &&
                    diagnostics.tail_iteration + 100 == diagnostics.reference_iterations &&
                    diagnostics.warm_start_restore_max_abs <= 1.0e-12 &&
                    (cfg.benchmark || (result.finite && result.max_field_rel <= 1.0e-10));
                if (!step_pass)
                    throw std::runtime_error("warm-start replay gate failed at step " + std::to_string(step));
            }
            ++passed_steps;
            total_iterations += diagnostics.iterations;
            all_first_check = all_first_check && diagnostics.warm_start_first_check_converged;
            max_field_rel = std::max(max_field_rel, result.max_field_rel);
            max_restore_abs = std::max(max_restore_abs, diagnostics.warm_start_restore_max_abs);
            csv << step << ',' << cfg.route << ',' << diagnostics.iterations << ','
                << diagnostics.reference_iterations << ',' << diagnostics.tail_iteration << ','
                << diagnostics.relative_error << ',' << diagnostics.pressure_l1_delta << ','
                << diagnostics.pressure_l1_norm << ',' << diagnostics.fixed_point_h_relative << ','
                << diagnostics.fixed_point_p_relative << ',' << diagnostics.fixed_point_relative << ','
                << diagnostics.pressure_converged << ',' << diagnostics.fixed_point_converged << ','
                << diagnostics.dual_residual_enabled << ',' << diagnostics.converged << ','
                << diagnostics.fallback_count << ',' << diagnostics.warm_start_used << ','
                << diagnostics.warm_start_first_check_converged << ',' << diagnostics.warm_start_io_ms << ','
                << diagnostics.warm_start_restore_max_abs << ',' << compute_ms << ',' << e2e_ms << ','
                << validation_io_ms << ',' << reference_write_io_ms << ','
                << diagnostics.source_mean << ',' << diagnostics.source_abs_mean << ','
                << diagnostics.projected_source_mean << ',' << diagnostics.pressure_gauge_target << ','
                << diagnostics.pressure_gauge_max_shift << ','
                << result.p_mean << ',' << result.p_ref_mean << ',' << result.p_mean_abs_diff << ','
                << result.p_raw_rel << ',' << result.p_demeaned_rel << ',' << result.pressure_gradient_rel << ','
                << result.pressure_correction_rel << ','
                << result.velocity_rel << ',' << result.u_rel << ',' << result.v_rel << ',' << result.w_rel << ','
                << result.rho_rel << ',' << result.fei_rel << ',' << result.mass_rel << ','
                << result.shape_mismatch_fraction << ',' << result.shape_dice_error << ','
                << result.candidate_components << ',' << result.reference_components << ','
                << result.largest_component_rel << ',' << result.centroid_distance_normalized << ','
                << result.h_moment_rel << ',' << result.h_p_consistency_max_abs << ',' << result.h_equilibrium_max_abs << ','
                << result.max_field_rel << ',' << result.finite << ',' << result.bitwise_equal << '\n';
            if (!cfg.benchmark) csv.flush();
            if (!cfg.benchmark) std::cout << "route=" << cfg.route << " step=" << step
                      << " iterations=" << diagnostics.iterations
                      << " reference_iterations=" << diagnostics.reference_iterations
                      << " tail_iteration=" << diagnostics.tail_iteration
                      << " residual=" << diagnostics.relative_error
                      << " fixed_residual=" << diagnostics.fixed_point_relative
                      << " e2e_ms=" << e2e_ms
                      << " max_field_rel=" << result.max_field_rel
                      << " bitwise=" << result.bitwise_equal << std::endl;
        }
        const bool go = cfg.route == "replay" && passed_steps == cfg.steps && cfg.steps == 20 &&
            (cfg.benchmark || max_field_rel <= 1.0e-10);
        const bool qualification_pass = passed_steps == cfg.steps;
        std::ofstream summary(cfg.output / "summary.json", std::ios::trunc);
        if (!summary) throw std::runtime_error("cannot write summary.json");
        summary << std::scientific << std::setprecision(17)
                << "{\n  \"route\": \"" << cfg.route << "\",\n"
                << "  \"requested_steps\": " << cfg.steps << ",\n"
                << "  \"passed_steps\": " << passed_steps << ",\n"
                << "  \"total_iterations\": " << total_iterations << ",\n"
                << "  \"all_first_check\": " << (all_first_check ? "true" : "false") << ",\n"
                << "  \"max_field_rel\": " << max_field_rel << ",\n"
                << "  \"max_restore_abs\": " << max_restore_abs << ",\n"
                << "  \"qualification_pass\": " << (qualification_pass ? "true" : "false") << ",\n"
                << "  \"go\": " << (go ? "true" : "false") << "\n}\n";
        if (cfg.route == "replay" && cfg.steps != 20)
            std::cerr << "Replay qualification passed; Go remains false until the frozen 20-step gate.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Oracle run failed: " << e.what() << '\n';
        return 1;
    }
}
