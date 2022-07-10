#include "Overrides.hpp"
#include "internal/test_bitmask.hpp"
#include <Eigen/Dense>
#include <complex>
#include <string>
#include <universal/number/posit/posit.hpp>

using posit = sw::universal::posit<64, 4>;
using cposit = std::complex<posit>;
using cdouble = std::complex<double>;

using MatrixXp = Eigen::Matrix<posit, -1, -1>;
using MatrixXcp = Eigen::Matrix<cposit, -1, -1>;
using Eigen::MatrixXcd;

using VectorXp = Eigen::Matrix<posit, -1, 1>;
using VectorXcp = Eigen::Matrix<cposit, -1, 1>;
using Eigen::VectorXcd;

posit abs_posit(posit p) { return p < 0 ? -p : p; }

bool test_fdp() {
  // run a few fdp computations
  // first, real values
  {
    MatrixXp vec1 = MatrixXp::Zero(5, 1);
    MatrixXp vec2 = MatrixXp::Zero(5, 1);

    vec1(0) = 1;
    vec1(1) = 2;
    vec1(2) = -3;
    vec1(3) = 0;
    vec1(4) = 10;

    vec2(0) = 2;
    vec2(1) = 1;
    vec2(2) = -4;
    vec2(3) = 5;
    vec2(4) = -0.1;

    posit result1 = eigen_fdp(vec1, vec2);
    if (abs_posit(result1 - 15) > 1e-15) {
      return false;
    }
    posit result2 = eigen_fdp(vec1.transpose(), vec2);
    if (abs_posit(result2 - 15) > 1e-15) {
      return false;
    }
    posit result3 = eigen_fdp(vec1, vec2.transpose());
    if (abs_posit(result3 - 15) > 1e-15) {
      return false;
    }
    posit result4 = eigen_fdp(vec1.transpose(), vec2.transpose());
    if (abs_posit(result4 - 15) > 1e-15) {
      return false;
    }
  }

  // then complex values
  {
    MatrixXcp vec1 = MatrixXcp::Zero(5, 1);
    MatrixXcp vec2 = MatrixXcp::Zero(1, 5); // flipped intentionally

    vec1(0) = cposit(1, 1);
    vec1(1) = cposit(2, -2);
    vec1(2) = cposit(-3, 3.1);
    vec1(3) = cposit(0, -1);
    vec1(4) = cposit(10, 0);

    vec2(0) = cposit(2, -1);
    vec2(1) = cposit(1, 1);
    vec2(2) = cposit(-4, 2);
    vec2(3) = cposit(5, 0);
    vec2(4) = cposit(-0.1, 10);

    cposit expected = cposit(11.8, 77.6);
    cposit result1 = eigen_fdp(vec1, vec2);
    // tolerance to 1e-14 because that is the error introduced by abs
    // calculation...
    if (std::abs(result1 - expected) > 1e-14) {
      return false;
    }
    cposit result2 = eigen_fdp(vec1.transpose(), vec2);
    if (std::abs(result2 - expected) > 1e-14) {
      return false;
    }
    cposit result3 = eigen_fdp(vec1, vec2.transpose());
    if (std::abs(result3 - expected) > 1e-14) {
      return false;
    }
    cposit result4 = eigen_fdp(vec1.transpose(), vec2.transpose());
    if (std::abs(result4 - expected) > 1e-14) {
      return false;
    }
  }

  // finally, random complex vaules
  {
    MatrixXcd dvec1 = MatrixXcd::Random(5, 1);
    MatrixXcd dvec2 = MatrixXcd::Random(1, 5);

    MatrixXcp vec1 = MatrixXcp::Zero(5, 1);
    MatrixXcp vec2 = MatrixXcp::Zero(1, 5); // flipped intentionally

    for (int idx = 0; idx < vec1.rows(); idx++) {
      vec1(idx) = cposit(posit(dvec1(idx).real()), posit(dvec1(idx).imag()));
      vec2(idx) = cposit(posit(dvec2(idx).real()), posit(dvec2(idx).imag()));
    }

    cposit manual = vec1(0) * vec2(0) + vec1(1) * vec2(1) + vec1(2) * vec2(2) +
                    vec1(3) * vec2(3) + vec1(4) * vec2(4);

    cposit result1 = eigen_fdp(vec1, vec2);
    if (std::abs(result1 - manual) > 1e-10) {
      return false;
    }
    cposit result2 = eigen_fdp(vec1.transpose(), vec2);
    if (std::abs(result2 - manual) > 1e-10) {
      return false;
    }
    cposit result3 = eigen_fdp(vec1, vec2.transpose());
    if (std::abs(result3 - manual) > 1e-10) {
      return false;
    }
    cposit result4 = eigen_fdp(vec1.transpose(), vec2.transpose());
    if (std::abs(result4 - manual) > 1e-10) {
      return false;
    }
  }
  return true;
}

bool test_gemm() {
  // first real
  {
    MatrixXp mat1 = MatrixXp::Zero(2, 2);
    MatrixXp mat2 = MatrixXp::Zero(2, 2);

    mat1(0, 0) = 1;
    mat1(0, 1) = 0.1;
    mat1(1, 0) = -2;
    mat1(1, 1) = 0;

    mat2(0, 0) = 1;
    mat2(0, 1) = -1;
    mat2(1, 0) = 1;
    mat2(1, 1) = 1;

    MatrixXp result = MatrixXp::Zero(2, 2);

    MatrixXp expected = MatrixXp::Zero(2, 2);
    expected(0, 0) = 1.1;
    expected(0, 1) = -0.9;
    expected(1, 0) = -2;
    expected(1, 1) = 2;

    Eigen::internal::generic_product_impl<MatrixXp, MatrixXp, Eigen::DenseShape,
                                          Eigen::DenseShape,
                                          Eigen::GemmProduct>::evalTo(result,
                                                                      mat1,
                                                                      mat2);

    MatrixXp residual = MatrixXp::Zero(2, 2);
    for (int idx_r = 0; idx_r < residual.rows(); idx_r++) {
      for (int idx_c = 0; idx_c < residual.cols(); idx_c++) {
        residual(idx_r, idx_c) =
            abs_posit(expected(idx_r, idx_c) - result(idx_r, idx_c));
      }
    }

    if (residual.maxCoeff() > 1e-13) {
      std::cerr << "real gemm failed with residual: " << residual.maxCoeff()
                << std::endl;
      return false;
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }
  // then complex
  {
    MatrixXcp mat1 = MatrixXcp::Zero(2, 2);
    MatrixXcp mat2 = MatrixXcp::Zero(2, 2);

    mat1(0, 0) = cposit(1, 1);
    mat1(0, 1) = cposit(0.1, -1);
    mat1(1, 0) = cposit(-2, 0.5);
    mat1(1, 1) = cposit(0, 0);

    mat2(0, 0) = cposit(1, 3);
    mat2(0, 1) = cposit(-1, -2);
    mat2(1, 0) = cposit(1, 0);
    mat2(1, 1) = cposit(1, 1);

    MatrixXcp result = MatrixXcp::Zero(2, 2);

    MatrixXcp expected = MatrixXcp::Zero(2, 2);
    expected(0, 0) = cposit(-1.9, 3);
    expected(0, 1) = cposit(2.1, -3.9);
    expected(1, 0) = cposit(-3.5, -5.5);
    expected(1, 1) = cposit(3, 3.5);

    Eigen::internal::generic_product_impl<MatrixXcp, MatrixXcp,
                                          Eigen::DenseShape, Eigen::DenseShape,
                                          Eigen::GemmProduct>::evalTo(result,
                                                                      mat1,
                                                                      mat2);

    MatrixXp residual = MatrixXp::Zero(2, 2);
    for (int idx_r = 0; idx_r < residual.rows(); idx_r++) {
      for (int idx_c = 0; idx_c < residual.cols(); idx_c++) {
        residual(idx_r, idx_c) =
            std::abs(expected(idx_r, idx_c) - result(idx_r, idx_c));
      }
    }

    if (residual.maxCoeff() > 1e-13) {
      std::cerr << "complex gemm failed with residual: " << residual.maxCoeff()
                << std::endl;
      return false;
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }
  // finally, random complex numbers
  for (int rows = 1; rows < 10; rows += 2) {
    for (int cols = 1; cols < 10; cols += 2) {
      for (int depth = 1; depth < 10; depth += 2) {
        MatrixXcd dmat1 = MatrixXcd::Random(rows, depth);
        MatrixXcd dmat2 = MatrixXcd::Random(depth, cols);

        MatrixXcp mat1 = MatrixXcp::Zero(rows, depth);
        MatrixXcp mat2 = MatrixXcp::Zero(depth, cols);

        for (int idx_r = 0; idx_r < mat1.rows(); idx_r++) {
          for (int idx_c = 0; idx_c < mat1.cols(); idx_c++) {
            mat1(idx_r, idx_c) = cposit(posit(dmat1(idx_r, idx_c).real()),
                                        posit(dmat1(idx_r, idx_c).imag()));
          }
        }
        for (int idx_r = 0; idx_r < mat2.rows(); idx_r++) {
          for (int idx_c = 0; idx_c < mat2.cols(); idx_c++) {
            mat2(idx_r, idx_c) = cposit(posit(dmat2(idx_r, idx_c).real()),
                                        posit(dmat2(idx_r, idx_c).imag()));
          }
        }

        MatrixXcp result = MatrixXcp::Zero(rows, cols);

        MatrixXcp expected = MatrixXcp::Zero(rows, cols);
        // well, check one implementation vs another... still a valid test
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
            for (int k = 0; k < depth; k++) {
              expected(i, j) += mat1(i, k) * mat2(k, j);
            }
          }
        }

        Eigen::internal::generic_product_impl<
            MatrixXcp, MatrixXcp, Eigen::DenseShape, Eigen::DenseShape,
            Eigen::GemmProduct>::evalTo(result, mat1, mat2);

        MatrixXp residual = MatrixXp::Zero(rows, cols);
        for (int idx_r = 0; idx_r < rows; idx_r++) {
          for (int idx_c = 0; idx_c < cols; idx_c++) {
            residual(idx_r, idx_c) =
                std::abs(expected(idx_r, idx_c) - result(idx_r, idx_c));
          }
        }

        if (residual.maxCoeff() > 1e-13) {
          std::cerr << "random gemm (" << rows << "x" << cols << "x" << depth
                    << ") failed with residual: " << residual.maxCoeff()
                    << std::endl;
          return false;
        }
      }
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }

  return true;
}

bool test_gemv() {
  // first real
  {
    MatrixXp mat1 = MatrixXp::Zero(2, 2);
    MatrixXp mat2 = MatrixXp::Zero(2, 1);

    mat1(0, 0) = 1;
    mat1(0, 1) = 0.1;
    mat1(1, 0) = -2;
    mat1(1, 1) = 0;

    mat2(0, 0) = 1;
    mat2(1, 0) = 1;

    MatrixXp result = MatrixXp::Zero(2, 1);

    MatrixXp expected = MatrixXp::Zero(2, 1);
    expected(0, 0) = 1.1;
    expected(1, 0) = -2;

    Eigen::internal::generic_product_impl<MatrixXp, MatrixXp, Eigen::DenseShape,
                                          Eigen::DenseShape,
                                          Eigen::GemvProduct>::evalTo(result,
                                                                      mat1,
                                                                      mat2);

    MatrixXp residual = MatrixXp::Zero(2, 1);
    for (int idx_r = 0; idx_r < residual.rows(); idx_r++) {
      residual(idx_r, 0) = abs_posit(expected(idx_r, 0) - result(idx_r, 0));
    }

    if (residual.maxCoeff() > 1e-13) {
      std::cerr << "real gemv failed with residual: " << residual.maxCoeff()
                << std::endl;
      return false;
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }
  // then complex
  {
    MatrixXcp mat1 = MatrixXcp::Zero(2, 2);
    MatrixXcp mat2 = MatrixXcp::Zero(2, 1);

    mat1(0, 0) = cposit(1, 1);
    mat1(0, 1) = cposit(0.1, -1);
    mat1(1, 0) = cposit(-2, 0.5);
    mat1(1, 1) = cposit(0, 0);

    mat2(0, 0) = cposit(1, 3);
    mat2(1, 0) = cposit(1, 0);

    MatrixXcp result = MatrixXcp::Zero(2, 1);

    MatrixXcp expected = MatrixXcp::Zero(2, 1);
    expected(0, 0) = cposit(-1.9, 3);
    expected(1, 0) = cposit(-3.5, -5.5);

    Eigen::internal::generic_product_impl<MatrixXcp, MatrixXcp,
                                          Eigen::DenseShape, Eigen::DenseShape,
                                          Eigen::GemvProduct>::evalTo(result,
                                                                      mat1,
                                                                      mat2);

    MatrixXp residual = MatrixXp::Zero(2, 1);
    for (int idx_r = 0; idx_r < residual.rows(); idx_r++) {
      residual(idx_r, 0) = std::abs(expected(idx_r, 0) - result(idx_r, 0));
    }

    if (residual.maxCoeff() > 1e-13) {
      std::cerr << "complex gemv failed with residual: " << residual.maxCoeff()
                << std::endl;
      return false;
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }
  // finally, random complex numbers
  for (int rows = 1; rows < 10; rows += 2) {
    for (int cols = 1; cols < 10; cols += 2) {
      MatrixXcd dmat1 = MatrixXcd::Random(rows, cols);
      MatrixXcd dmat2 = MatrixXcd::Random(cols, 1);

      MatrixXcp mat1 = MatrixXcp::Zero(rows, cols);
      MatrixXcp mat2 = MatrixXcp::Zero(cols, 1);

      for (int idx_r = 0; idx_r < mat1.rows(); idx_r++) {
        for (int idx_c = 0; idx_c < mat1.cols(); idx_c++) {
          mat1(idx_r, idx_c) = cposit(posit(dmat1(idx_r, idx_c).real()),
                                      posit(dmat1(idx_r, idx_c).imag()));
        }
      }
      for (int idx_r = 0; idx_r < mat2.rows(); idx_r++) {
        for (int idx_c = 0; idx_c < mat2.cols(); idx_c++) {
          mat2(idx_r, idx_c) = cposit(posit(dmat2(idx_r, idx_c).real()),
                                      posit(dmat2(idx_r, idx_c).imag()));
        }
      }

      MatrixXcp result = MatrixXcp::Zero(rows, 1);

      MatrixXcp expected = MatrixXcp::Zero(rows, 1);
      // well, check one implementation vs another... still a valid test
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          expected(i, 0) += mat1(i, j) * mat2(j, 0);
        }
      }

      Eigen::internal::generic_product_impl<
          MatrixXcp, MatrixXcp, Eigen::DenseShape, Eigen::DenseShape,
          Eigen::GemvProduct>::evalTo(result, mat1, mat2);

      MatrixXp residual = MatrixXp::Zero(rows, 1);
      for (int idx_r = 0; idx_r < rows; idx_r++) {
        residual(idx_r, 0) = std::abs(expected(idx_r, 0) - result(idx_r, 0));
      }

      if (residual.maxCoeff() > 1e-13) {
        std::cerr << "random gemv (" << rows << "x" << cols
                  << ") failed with residual: " << residual.maxCoeff()
                  << std::endl;
        return false;
      }
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }

  return true;
}

bool test_inner() {
  // first real
  {
    MatrixXp mat1 = MatrixXp::Zero(2, 1);
    MatrixXp mat2 = MatrixXp::Zero(2, 1);

    mat1(0, 0) = 1;
    mat1(1, 0) = -2;

    mat2(0, 0) = 1;
    mat2(1, 0) = 1;

    MatrixXp result = MatrixXp::Zero(1, 1);

    MatrixXp expected = MatrixXp::Zero(1, 1);
    expected(0, 0) = -1;

    Eigen::internal::generic_product_impl<
        MatrixXp, MatrixXp, Eigen::DenseShape, Eigen::DenseShape,
        Eigen::InnerProduct>::evalTo(result, mat1.transpose(), mat2);

    MatrixXp residual = MatrixXp::Zero(1, 1);
    residual(0, 0) = abs_posit(expected(0, 0) - result(0, 0));

    if (residual.maxCoeff() > 1e-13) {
      std::cerr << "real inner failed with residual: " << residual.maxCoeff()
                << std::endl;
      return false;
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }
  // then complex
  {
    MatrixXcp mat1 = MatrixXcp::Zero(2, 1);
    MatrixXcp mat2 = MatrixXcp::Zero(2, 1);

    mat1(0, 0) = cposit(1, 1);
    mat1(1, 0) = cposit(-2, 0.5);

    mat2(0, 0) = cposit(1, 3);
    mat2(1, 0) = cposit(1, 0);

    MatrixXcp result = MatrixXcp::Zero(1, 1);

    MatrixXcp expected = MatrixXcp::Zero(1, 1);
    expected(0, 0) = cposit(-4, 4.5);

    Eigen::internal::generic_product_impl<
        MatrixXcp, MatrixXcp, Eigen::DenseShape, Eigen::DenseShape,
        Eigen::InnerProduct>::evalTo(result, mat1.transpose(), mat2);

    MatrixXp residual = MatrixXp::Zero(1, 1);
    residual(0, 0) = std::abs(expected(0, 0) - result(0, 0));

    if (residual.maxCoeff() > 1e-13) {
      std::cerr << "complex inner failed with residual: " << residual.maxCoeff()
                << std::endl;
      return false;
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }
  // finally, random complex numbers
  for (int rows = 1; rows < 10; rows += 2) {
    MatrixXcd dmat1 = MatrixXcd::Random(rows, 1);
    MatrixXcd dmat2 = MatrixXcd::Random(rows, 1);

    MatrixXcp mat1 = MatrixXcp::Zero(rows, 1);
    MatrixXcp mat2 = MatrixXcp::Zero(rows, 1);

    for (int idx = 0; idx < rows; idx++) {
      mat1(idx, 0) =
          cposit(posit(dmat1(idx, 0).real()), posit(dmat1(idx, 0).imag()));
      mat2(idx, 0) =
          cposit(posit(dmat2(idx, 0).real()), posit(dmat2(idx, 0).imag()));
    }

    MatrixXcp result = MatrixXcp::Zero(1, 1);

    MatrixXcp expected = MatrixXcp::Zero(1, 1);
    // well, check one implementation vs another... still a valid test
    for (int i = 0; i < rows; i++) {
      expected(0, 0) += mat1(i, 0) * mat2(i, 0);
    }

    Eigen::internal::generic_product_impl<
        MatrixXcp, MatrixXcp, Eigen::DenseShape, Eigen::DenseShape,
        Eigen::InnerProduct>::evalTo(result, mat1.transpose(), mat2);

    MatrixXp residual = MatrixXp::Zero(1, 1);
    residual(0, 0) = std::abs(expected(0, 0) - result(0, 0));

    if (residual.maxCoeff() > 1e-13) {
      std::cerr << "random inner (" << rows
                << ") failed with residual: " << residual.maxCoeff()
                << std::endl;
      return false;
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }

  return true;
}

bool test_dot() {
  // first real
  {
    VectorXp mat1 = VectorXp::Zero(2);
    VectorXp mat2 = VectorXp::Zero(2);

    mat1(0) = 1;
    mat1(1) = -2;

    mat2(0) = 1;
    mat2(1) = 1;

    posit result =
        Eigen::internal::dot_nocheck<VectorXp, VectorXp>::run(mat1, mat2);

    posit expected = -1;

    posit residual = abs_posit(expected - result);

    if (residual > 1e-13) {
      std::cerr << "real dot failed with residual: " << residual << std::endl;
      return false;
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }
  // then complex
  {
    VectorXcp mat1 = VectorXcp::Zero(2);
    VectorXcp mat2 = VectorXcp::Zero(2);

    mat1(0) = cposit(1, 1);
    mat1(1) = cposit(-2, 0.5);

    mat2(0) = cposit(1, 3);
    mat2(1) = cposit(1, 0);

    cposit result =
        Eigen::internal::dot_nocheck<VectorXcp, VectorXcp>::run(mat1, mat2);

    cposit expected = cposit(2, 1.5);

    posit residual = std::abs(expected - result);

    if (residual > 1e-13) {
      std::cerr << "complex dot failed with residual: " << residual
                << std::endl;
      return false;
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }
  // finally, random complex numbers
  for (int rows = 1; rows < 10; rows += 2) {
    VectorXcd dmat1 = VectorXcd::Random(rows);
    VectorXcd dmat2 = VectorXcd::Random(rows);

    VectorXcp mat1 = VectorXcp::Zero(rows);
    VectorXcp mat2 = VectorXcp::Zero(rows);

    for (int idx = 0; idx < rows; idx++) {
      mat1(idx) = cposit(posit(dmat1(idx).real()), posit(dmat1(idx).imag()));
      mat2(idx) = cposit(posit(dmat2(idx).real()), posit(dmat2(idx).imag()));
    }

    cposit result =
        Eigen::internal::dot_nocheck<VectorXcp, VectorXcp>::run(mat1, mat2);

    // well, check one implementation vs another... still a valid test
    cposit expected = cposit(0, 0);
    for (int i = 0; i < rows; i++) {
      expected += mat1.conjugate()(i) * mat2(i);
    }

    posit residual = std::abs(expected - result);

    if (residual > 1e-13) {
      std::cerr << "random inner (" << rows
                << ") failed with residual: " << residual << std::endl;
      return false;
    }
    assert(eigen_usage_vector != 0);
    eigen_usage_vector = 0;
  }

  return true;
}

namespace Colors {
const std::string red("\033[1;31m");
const std::string green("\033[1;32m");
const std::string bold("\033[1m");
const std::string reset("\033[0m");
} // namespace Colors

template <typename Test> bool run_test(Test t, std::string name) {
  bool result = t();
  if (result) {
    std::cout << name << ": " << Colors::green << "OK" << Colors::reset
              << std::endl;
  } else {
    std::cout << name << ": " << Colors::red << "FAIL" << Colors::reset
              << std::endl;
  }
  return result;
}

int main() {
  bool good = true;
  good &= run_test(test_fdp, std::string("FDP  "));
  good &= run_test(test_gemm, std::string("GEMM "));
  good &= run_test(test_gemv, std::string("GEMV "));
  good &= run_test(test_inner, std::string("INNER"));
  good &= run_test(test_dot, std::string("DOT  "));
  if (!good) {
    return -1;
  }
  return 0;
}
