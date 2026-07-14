#include "InamuroCUDA.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    try {
        const std::string dump_path = argc > 1 ? argv[1] : "tbook_stage_audit.bin";
        const std::filesystem::path parent = std::filesystem::path(dump_path).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        const auto result = InamuroCUDA::runTBookStageAudit(dump_path);
        std::cout << std::scientific << std::setprecision(17)
                  << "collision_max_abs=" << result.collision_max_abs << '\n'
                  << "stream_max_abs=" << result.stream_max_abs << '\n'
                  << "boundary_max_abs=" << result.boundary_max_abs << '\n'
                  << "pressure_max_abs=" << result.pressure_max_abs << '\n'
                  << "onepass_h_max_abs=" << result.onepass_h_max_abs << '\n'
                  << "onepass_p_max_abs=" << result.onepass_p_max_abs << '\n'
                  << "source_moment_max_abs=" << result.source_moment_max_abs << '\n'
                  << "projected_source_mean_abs=" << result.projected_source_mean_abs << '\n'
                  << "source_projection_gradient_max_abs=" << result.source_projection_gradient_max_abs << '\n'
                  << "gauge_gradient_max_abs=" << result.gauge_gradient_max_abs << '\n'
                  << "gauge_correct_uvw_max_abs=" << result.gauge_correct_uvw_max_abs << '\n'
                  << "gauge_p_sum_h_max_abs=" << result.gauge_p_sum_h_max_abs << '\n'
                  << "dump=" << dump_path << '\n';
        const double tolerance = 1.0e-12;
        const bool pass = result.collision_max_abs <= tolerance &&
                          result.stream_max_abs <= tolerance &&
                          result.boundary_max_abs <= tolerance &&
                          result.pressure_max_abs <= tolerance &&
                          result.onepass_h_max_abs <= tolerance &&
                          result.onepass_p_max_abs <= tolerance &&
                          result.source_moment_max_abs <= tolerance &&
                          result.projected_source_mean_abs <= tolerance &&
                          result.source_projection_gradient_max_abs <= tolerance &&
                          result.gauge_gradient_max_abs <= tolerance &&
                          result.gauge_correct_uvw_max_abs <= tolerance &&
                          result.gauge_p_sum_h_max_abs <= tolerance;
        return pass ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "T_book stage audit failed: " << e.what() << '\n';
        return 1;
    }
}
