#include <iostream>
#include <Eigen/Dense>
#include <random>

int main() {
    Eigen::MatrixXd mat = Eigen::MatrixXd::Random(50, 50);

    Eigen::BDCSVD svd(mat, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);

    auto U = svd.matrixU();
    auto V = svd.matrixV();
    auto values = svd.singularValues();

    Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(50, 50);
    for (int i = 0; i < sigma.rows(); i++) {
        sigma(i, i) = values(i);
    }

    auto mat_prime = U*sigma*V.transpose().conjugate();

    if (std::abs((mat_prime - mat).maxCoeff()) > 1e-10) {
        return -1;
    }
    return 0;
}