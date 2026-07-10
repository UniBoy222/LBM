#pragma once

#include <vector>

struct BookPressureStages {
    std::vector<double> divergence;
    std::vector<double> collision;
    std::vector<double> streamed;
    std::vector<double> bounced;
    std::vector<double> pressure;
};

struct BookPressureSolution {
    std::vector<double> pressure;
    std::vector<double> hh;
    int iterations = 0;
    bool converged = false;
    double relative_error = 0.0;
};

BookPressureStages runBookPressureStagesGPU(
    int lx, int ly, int lz,
    const std::vector<double>& hh,
    const std::vector<double>& pressure,
    const std::vector<double>& rho,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w);

BookPressureSolution solveBookPressureGPU(
    int lx, int ly, int lz,
    const std::vector<double>& hh,
    const std::vector<double>& pressure,
    const std::vector<double>& rho,
    const std::vector<double>& u,
    const std::vector<double>& v,
    const std::vector<double>& w,
    const std::vector<double>& divergence,
    int max_iterations,
    int check_interval,
    double tolerance);
