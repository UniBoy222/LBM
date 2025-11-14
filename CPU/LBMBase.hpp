#pragma once

#include <iostream>
#include <string>
#include <vector>

/**
 * 标准LBM抽象基类
 * 定义了所有LBM方法的标准接口和虚函数
 * 这是一个纯抽象"空壳"基类，为所有LBM算法提供统一的框架
 */
class LBMBase
{
protected:
    using Vector4D = std::vector<std::vector<std::vector<std::vector<double>>>>; // 4D向量，表示4个方向的分布函数
    using Vector3D = std::vector<std::vector<std::vector<double>>>;

public:
    LBMBase() = default;

    virtual ~LBMBase() = default;

    virtual void collision() = 0;

    virtual void stream(Vector4D& dist) = 0;

    virtual void applyBoundaryConditions(Vector4D& dist) = 0;

    virtual void getMacro() = 0;

    virtual void performTimeStep() = 0;

    virtual void writeResults(int timeStep) = 0;

    virtual void getGridSize(int& nx, int& ny, int& nz) const = 0;

    virtual std::string getAlgorithmName() const = 0;

    virtual void printInfo() const
    {
        std::cout << "\n=== LBM算法信息 ===" << std::endl;
        std::cout << "算法类型: " << getAlgorithmName() << std::endl;

        int nx, ny, nz;
        getGridSize(nx, ny, nz);
        std::cout << "网格尺寸: " << nx << " x " << ny << " x " << nz << std::endl;
        std::cout << "===================" << std::endl;
    }
};
