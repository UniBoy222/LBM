#include "InamuroCUDA.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " FIXED_POINT_STATE.bin\n";
        return 64;
    }
    try {
        const auto result = InamuroCUDA::runFixedPointStateAudit(argv[1]);
        std::cout << std::scientific << std::setprecision(17)
                  << "step=" << result.step << '\n'
                  << "iteration=" << result.iteration << '\n'
                  << "cells=" << result.cells << '\n'
                  << "collision_max_abs=" << result.collision_max_abs << '\n'
                  << "stream_max_abs=" << result.stream_max_abs << '\n'
                  << "boundary_max_abs=" << result.boundary_max_abs << '\n'
                  << "live_pressure_max_abs=" << result.live_pressure_max_abs << '\n'
                  << "image_pressure_max_abs=" << result.image_pressure_max_abs << '\n'
                  << "fixed_point_terms_relative_max_abs="
                  << result.fixed_point_terms_relative_max_abs << '\n'
                  << "gauge_map_max_abs=" << result.gauge_map_max_abs << '\n'
                  << "nonfinite_values=" << result.nonfinite_values << '\n';
        const double tolerance = 1.0e-12;
        const bool pass = result.nonfinite_values == 0 &&
                          result.collision_max_abs <= tolerance &&
                          result.stream_max_abs <= tolerance &&
                          result.boundary_max_abs <= tolerance &&
                          result.live_pressure_max_abs <= tolerance &&
                          result.image_pressure_max_abs <= tolerance &&
                          result.fixed_point_terms_relative_max_abs <= tolerance &&
                          result.gauge_map_max_abs <= tolerance;
        std::cout << "pass=" << (pass ? 1 : 0) << '\n';
        return pass ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "fixed-point state audit failed: " << error.what() << '\n';
        return 1;
    }
}
