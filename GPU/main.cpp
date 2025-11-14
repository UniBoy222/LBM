#include "InamuroSolver.hpp"
#include <iostream>

int main()
{
    try
    {
        // 创建求解器（使用文件构造函数）
        InamuroSolver solver("params.in");

        // 运行仿真
        solver.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}