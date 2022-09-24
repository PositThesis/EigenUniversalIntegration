#include "Overrides.hpp"
#include "internal/test_bitmask.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <universal/number/posit/posit.hpp>

#ifdef REAL
using Posit = sw::universal::posit<64, 4>;
using IEEE = double;
#endif

#ifdef COMPLEX
using Posit = std::complex<sw::universal::posit<64, 4>>;
using IEEE = std::complex<double>;
#endif

template <typename Scalar>
using Matrix = Eigen::SparseMatrix<Scalar>;
template <typename Scalar> using Vector = Eigen::SparseVector<Scalar>;

template <typename Scalar>
Matrix<Scalar> gen_matrix(Eigen::Index mat_size, Eigen::Index seed) {
  Matrix<Scalar> M(mat_size, mat_size);

  std::default_random_engine e(seed);
  std::normal_distribution<double> dist(0, 1);

  for (Eigen::Index row = 0; row < mat_size; row++) {
    for (Eigen::Index col = 0; col < mat_size; col++) {
        // only fill some
        if (row * col % 3 == 0) {
#ifdef REAL
      M.coeffRef(row, col) = Scalar(dist(e));
#endif
#ifdef COMPLEX
      M.coeffRef(row, col) = std::complex((typename Scalar::value_type)(dist(e)),
                                 (typename Scalar::value_type)(dist(e)));
#endif
        }
    }
  }
  return M;
}

template <typename Scalar>
Matrix<Scalar> gen_matrix(Eigen::Index mat_rows, Eigen::Index mat_cols,
                                 Eigen::Index seed) {
  Matrix<Scalar> M(mat_rows, mat_cols);

  std::default_random_engine e(seed);
  std::normal_distribution<double> dist(0, 1);

  for (Eigen::Index row = 0; row < mat_rows; row++) {
    for (Eigen::Index col = 0; col < mat_cols; col++) {
        // only fill some
        if (row * col % 3 == 0) {
#ifdef REAL
      M.coeffRef(row, col) = Scalar(dist(e));
#endif
#ifdef COMPLEX
      M.coeffRef(row, col) = std::complex((typename Scalar::value_type)(dist(e)),
                                 (typename Scalar::value_type)(dist(e)));
#endif
        }
    }
  }
  return M;
}

template <typename Scalar>
Vector<Scalar> gen_vector(Eigen::Index mat_size, Eigen::Index seed) {
  Vector<Scalar> V(mat_size);

  std::default_random_engine e(seed);
  std::normal_distribution<double> dist(0, 1);

  for (Eigen::Index row = 0; row < mat_size; row++) {
        // only fill some
        if (row % 3 == 0) {
#ifdef REAL
    V.coeffRef(row, 0) = Scalar(dist(e));
#endif
#ifdef COMPLEX
    V.coeffRef(row, 0) = std::complex((typename Scalar::value_type)(dist(e)),
                             (typename Scalar::value_type)(dist(e)));
#endif
        }
  }
  return V;
}

double matrix_difference(Matrix<IEEE> A, Matrix<Posit> B) {
  assert(A.rows() == B.rows() && A.cols() == B.cols());
  Matrix<IEEE> B_as_IEEE(A.rows(), A.cols());

  for (Eigen::Index row = 0; row < A.rows(); row++) {
    for (Eigen::Index col = 0; col < A.cols(); col++) {
#ifdef REAL
      B_as_IEEE.coeffRef(row, col) = (IEEE)B.coeff(row, col);
#endif
#ifdef COMPLEX
      B_as_IEEE.coeffRef(row, col) =
          std::complex((typename IEEE::value_type)B.coeff(row, col).real(),
                       (typename IEEE::value_type)B.coeff(row, col).imag());
#endif
    }
  }

  Matrix<IEEE> errors = A - B_as_IEEE;
#ifdef REAL
  IEEE max = 0;
#endif
#ifdef COMPLEX
  typename IEEE::value_type max = 0;
#endif
  for (Eigen::Index row = 0; row < A.rows(); row++) {
    for (Eigen::Index col = 0; col < A.cols(); col++) {
#ifdef REAL
      IEEE error = std::abs(errors.coeff(row, col));
#endif
#ifdef COMPLEX
      typename IEEE::value_type error = std::abs(errors.coeff(row, col));
#endif
      if (error > max)
        max = error;
    }
  }
  return max;
}
double vector_difference(Vector<IEEE> A, Vector<Posit> B) {
  assert(A.rows() == B.rows() && A.cols() == B.cols());
  Vector<IEEE> B_as_IEEE(A.rows());

  for (Eigen::Index row = 0; row < A.rows(); row++) {
#ifdef REAL
    B_as_IEEE.coeffRef(row, 0) = (IEEE)B.coeff(row, 0);
#endif
#ifdef COMPLEX
    B_as_IEEE.coeffRef(row, 0) =
        std::complex((typename IEEE::value_type)B.coeff(row, 0).real(),
                     (typename IEEE::value_type)B.coeff(row, 0).imag());
#endif
  }

  Vector<IEEE> errors = A - B_as_IEEE;
#ifdef REAL
  IEEE max = 0;
#endif
#ifdef COMPLEX
  typename IEEE::value_type max = 0;
#endif
  for (Eigen::Index row = 0; row < A.rows(); row++) {
#ifdef REAL
    IEEE error = std::abs(errors.coeff(row, 0));
#endif
#ifdef COMPLEX
    typename IEEE::value_type error = std::abs(errors.coeff(row, 0));
#endif
    if (error > max)
      max = error;
  }
  return max;
}

int main() {
  Matrix<Posit> Ap = gen_matrix<Posit>(30, 0);
  Matrix<IEEE> Ad = gen_matrix<IEEE>(30, 0);

  Matrix<Posit> Rp = Ap * Ap;
  Matrix<IEEE> Rd = Ad * Ad;

  // also test for GEMM: the decision of GEMM vs lazy is done in the GEMM
  // product_impl, which we override to always do GEMM
  if ((eigen_usage_vector & ((uint64_t)EigenOverrideMask::LAZYPRODUCT |
                             (uint64_t)EigenOverrideMask::GENERIC_LAZY |
                             (uint64_t)EigenOverrideMask::GENERIC_GEMM)) ==
      0x0) {
    std::cerr << "Lazyproduct not used: usage vector is " << eigen_usage_vector
              << std::endl;
    return -1;
  }
  if ((eigen_usage_vector & ((uint64_t)EigenOverrideMask::SPARSE_DST)) == 0x0) {
    std::cerr << "Sparse assign not used " << eigen_usage_vector << std::endl;
    return -1;
  }
  if ((eigen_usage_vector & ((uint64_t)EigenOverrideMask::SPARSE_LHS)) == 0x0) {
    std::cerr << "Sparse lhs not used " << eigen_usage_vector << std::endl;
    return -1;
  }
  if ((eigen_usage_vector & ((uint64_t)EigenOverrideMask::SPARSE_RHS)) == 0x0) {
    std::cerr << "Sparse rhs not used " << eigen_usage_vector << std::endl;
    return -1;
  }

  double residual = matrix_difference(Rd, Rp);

  if (residual < 1e-10) {
    return 0;
  } else {
    std::cerr << "Result matrices IEEE: \n"
              << Rd << "\nPosit: \n"
              << Rp << std::endl;
    std::cerr << "Lazyproduct residual: " << residual << std::endl;
    return -1;
  }
}