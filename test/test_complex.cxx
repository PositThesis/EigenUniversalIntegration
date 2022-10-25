#include "Overrides.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <complex>
#include <universal/number/posit/posit.hpp>

using Eigen::MatrixX;
using sw::universal::posit;

template <typename Scalar>
MatrixX<Scalar> gen_matrix(Eigen::Index mat_size, Eigen::Index seed) {
  MatrixX<Scalar> M = MatrixX<Scalar>::Zero(mat_size, mat_size);

  std::default_random_engine e(seed);
  std::normal_distribution<double> dist(0, 1);

  for (Eigen::Index row = 0; row < mat_size; row++) {
    for (Eigen::Index col = 0; col < mat_size; col++) {
      M(row, col) = std::complex((typename Scalar::value_type)(dist(e)),
                                 (typename Scalar::value_type)(dist(e)));
    }
  }
  return M;
}
template <typename Scalar>
MatrixX<Scalar> gen_matrix_real(Eigen::Index mat_size, Eigen::Index seed) {
  MatrixX<Scalar> M = MatrixX<Scalar>::Zero(mat_size, mat_size);

  std::default_random_engine e(seed);
  std::normal_distribution<double> dist(0, 1);

  for (Eigen::Index row = 0; row < mat_size; row++) {
    for (Eigen::Index col = 0; col < mat_size; col++) {
      M(row, col) = (Scalar)(dist(e));
    }
  }
  return M;
}

template <typename Scalar_in, typename Scalar_out>
MatrixX<Scalar_out> convert_types(MatrixX<Scalar_in> in) {
    MatrixX<Scalar_out> m = MatrixX<Scalar_out>::Zero(in.rows(), in.cols());
    for (Eigen::Index i = 0; i < in.rows(); i++) {
        for (Eigen::Index j = 0; j < in.cols(); j++) {
            m(i, j) = std::complex<typename Scalar_out::value_type>((typename Scalar_out::value_type)in(i, j).real(), (typename Scalar_out::value_type)in(i, j).imag());
        }
    }
    return m;
}

int main() {
    MatrixX<std::complex<double>> ieee_1 = gen_matrix<std::complex<double>>(5, 5);
    MatrixX<std::complex<double>> ieee_2 = gen_matrix<std::complex<double>>(5, 6);
    MatrixX<double> ieee_1_r = gen_matrix_real<double>(5, 7);

    MatrixX<std::complex<posit<64,2>>> p_1 = gen_matrix<std::complex<posit<64,2>>>(5, 5);
    MatrixX<std::complex<posit<64,2>>> p_2 = gen_matrix<std::complex<posit<64,2>>>(5, 6);
    MatrixX<posit<64,2>> p_1_r = gen_matrix_real<posit<64,2>>(5, 7);

    {
        MatrixX<std::complex<double>> ieee_3 = ieee_1 * ieee_2;
        MatrixX<std::complex<posit<64,2>>> p_3 = p_1 * p_2;

        MatrixX<std::complex<double>> ieee_4 = ieee_1_r * ieee_2;
        MatrixX<std::complex<posit<64,2>>> p_4 = p_1_r * p_2;
        MatrixX<std::complex<double>> ieee_5 = ieee_2 * ieee_1_r;
        MatrixX<std::complex<posit<64,2>>> p_5 = p_2 * p_1_r;

        std::cout << "IEEE: \n" << ieee_3 << std::endl;
        std::cout << "Posit: \n" << p_3 << std::endl;
        std::cout << "IEEE: \n" << ieee_4 << std::endl;
        std::cout << "Posit: \n" << p_4 << std::endl;
        std::cout << "IEEE: \n" << ieee_5 << std::endl;
        std::cout << "Posit: \n" << p_5 << std::endl;

        MatrixX<std::complex<double>> d1 = ieee_3 - convert_types<std::complex<posit<64,2>>, std::complex<double>>(p_3);
        MatrixX<std::complex<double>> d2 = ieee_4 - convert_types<std::complex<posit<64,2>>, std::complex<double>>(p_4);
        MatrixX<std::complex<double>> d3 = ieee_5 - convert_types<std::complex<posit<64,2>>, std::complex<double>>(p_5);

        std::cout << "Delta 1: \n" << d1.cwiseAbs().maxCoeff() << std::endl;
        std::cout << "Delta 2: \n" << d2.cwiseAbs().maxCoeff() << std::endl;
        std::cout << "Delta 3: \n" << d3.cwiseAbs().maxCoeff() << std::endl;

        if (d1.cwiseAbs().maxCoeff() > 1e-7 || d2.cwiseAbs().maxCoeff() > 1e-7 || d3.cwiseAbs().maxCoeff() > 1e-7) {
            return -1;
        }
    }

    // test non-square matrices
    {
        MatrixX<std::complex<double>> ieee_3 = ieee_1.block(0, 0, 4, 3) * ieee_2.block(0, 0, 3, 2);
        MatrixX<std::complex<posit<64,2>>> p_3 = p_1.block(0, 0, 4, 3) * p_2.block(0, 0, 3, 2);

        MatrixX<std::complex<double>> ieee_4 = ieee_1_r.block(0, 0, 4, 3) * ieee_2.block(0, 0, 3, 2);
        MatrixX<std::complex<posit<64,2>>> p_4 = p_1_r.block(0, 0, 4, 3) * p_2.block(0, 0, 3, 2);
        MatrixX<std::complex<double>> ieee_5 = ieee_2.block(0, 0, 4, 3) * ieee_1_r.block(0, 0, 3, 2);
        MatrixX<std::complex<posit<64,2>>> p_5 = p_2.block(0, 0, 4, 3) * p_1_r.block(0, 0, 3, 2);

        std::cout << "IEEE: \n" << ieee_3 << std::endl;
        std::cout << "Posit: \n" << p_3 << std::endl;
        std::cout << "IEEE: \n" << ieee_4 << std::endl;
        std::cout << "Posit: \n" << p_4 << std::endl;
        std::cout << "IEEE: \n" << ieee_5 << std::endl;
        std::cout << "Posit: \n" << p_5 << std::endl;

        MatrixX<std::complex<double>> d1 = ieee_3 - convert_types<std::complex<posit<64,2>>, std::complex<double>>(p_3);
        MatrixX<std::complex<double>> d2 = ieee_4 - convert_types<std::complex<posit<64,2>>, std::complex<double>>(p_4);
        MatrixX<std::complex<double>> d3 = ieee_5 - convert_types<std::complex<posit<64,2>>, std::complex<double>>(p_5);

        std::cout << "Delta 1: \n" << d1.cwiseAbs().maxCoeff() << std::endl;
        std::cout << "Delta 2: \n" << d2.cwiseAbs().maxCoeff() << std::endl;
        std::cout << "Delta 3: \n" << d3.cwiseAbs().maxCoeff() << std::endl;

        if (d1.cwiseAbs().maxCoeff() > 1e-7 || d2.cwiseAbs().maxCoeff() > 1e-7 || d3.cwiseAbs().maxCoeff() > 1e-7) {
            return -1;
        }
    }

    // test adjoint noalias product
    {
        MatrixX<std::complex<double>> ieee_3 = MatrixX<std::complex<double>>::Identity(4, 2);
        MatrixX<std::complex<posit<64,2>>> p_3 = MatrixX<std::complex<posit<64, 2>>>::Identity(4, 2);

        ieee_3 = ieee_1.block(0, 0, 3, 4).adjoint() * ieee_2.block(0, 0, 3, 2);
        std::cout << "real lhs: \n" << p_1.block(0, 0, 3, 4).adjoint() << std::endl << "type is: " << typeid(p_1.block(0, 0, 3, 4).adjoint()).name() << std::endl;
        p_3 = p_1.block(0, 0, 3, 4).adjoint().eval() * p_2.block(0, 0, 3, 2); // eval is necessary to force eigen to use the correct types

        MatrixX<std::complex<double>> d1 = ieee_3 - convert_types<std::complex<posit<64,2>>, std::complex<double>>(p_3);
        std::cout << "Delta 1: \n" << d1.cwiseAbs().maxCoeff() << std::endl;
        if (d1.cwiseAbs().maxCoeff() > 1e-7) {
            return -1;
        }
    }

    {
        Eigen::ColPivHouseholderQR<MatrixX<std::complex<double>>>      qr_i(ieee_1);
        Eigen::ColPivHouseholderQR<MatrixX<std::complex<posit<64,2>>>> qr_p(p_1);

        MatrixX<std::complex<double>> s_i = qr_i.householderQ();
        MatrixX<std::complex<posit<64,2>>> s_p = qr_p.householderQ();

        std::cout << "ieee hh: \n" << s_i << std::endl;
        std::cout << "posit hh: \n" << s_p << std::endl;

        MatrixX<std::complex<double>> d1 = s_i - convert_types<std::complex<posit<64,2>>, std::complex<double>>(s_p);

        std::cerr << "Delta hh: " << d1.cwiseAbs().maxCoeff() << std::endl;
        if (d1.cwiseAbs().maxCoeff() > 1e-7) {
            return -1;
        }
    }

    {
        Eigen::JacobiSVD<MatrixX<std::complex<double>>>      svd_i(ieee_1.block(0, 0, 5, 4), Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::JacobiSVD<MatrixX<std::complex<posit<64,2>>>> svd_p(p_1.block(0, 0, 5, 4), Eigen::ComputeThinU | Eigen::ComputeThinV);

        MatrixX<std::complex<double>> s_i = svd_i.singularValues();
        MatrixX<std::complex<posit<64,2>>> s_p = svd_p.singularValues();

        std::cout << "ieee SV: \n" << s_i << std::endl;
        std::cout << "posit SV: \n" << s_p << std::endl;

        MatrixX<std::complex<double>> d1 = s_i - convert_types<std::complex<posit<64,2>>, std::complex<double>>(s_p);

        std::cerr << "Delta SV: " << d1.cwiseAbs().maxCoeff() << std::endl;
        if (d1.cwiseAbs().maxCoeff() > 1e-7) {
            return -1;
        }
    }
}