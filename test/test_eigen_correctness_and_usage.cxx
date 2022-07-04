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

template <typename Scalar, int order>
using Matrix = Eigen::Matrix<Scalar, -1, -1, order>;
template <typename Scalar> using Vector = Eigen::Matrix<Scalar, -1, 1>;

template <typename Scalar, int order>
Matrix<Scalar, order> gen_matrix(Eigen::Index mat_size, Eigen::Index seed) {
  Matrix<Scalar, order> M = Matrix<Scalar, order>::Zero(mat_size, mat_size);

  std::default_random_engine e(seed);
  std::normal_distribution<double> dist(0, 1);

  for (Eigen::Index row = 0; row < mat_size; row++) {
    for (Eigen::Index col = 0; col < mat_size; col++) {
#ifdef REAL
      M(row, col) = Scalar(dist(e));
#endif
#ifdef COMPLEX
      M(row, col) = std::complex((typename Scalar::value_type)(dist(e)),
                                 (typename Scalar::value_type)(dist(e)));
#endif
    }
  }
  return M;
}

template <typename Scalar, int order>
Matrix<Scalar, order> gen_matrix(Eigen::Index mat_rows, Eigen::Index mat_cols,
                                 Eigen::Index seed) {
  Matrix<Scalar, order> M = Matrix<Scalar, order>::Zero(mat_rows, mat_cols);

  std::default_random_engine e(seed);
  std::normal_distribution<double> dist(0, 1);

  for (Eigen::Index row = 0; row < mat_rows; row++) {
    for (Eigen::Index col = 0; col < mat_cols; col++) {
#ifdef REAL
      M(row, col) = Scalar(dist(e));
#endif
#ifdef COMPLEX
      M(row, col) = std::complex((typename Scalar::value_type)(dist(e)),
                                 (typename Scalar::value_type)(dist(e)));
#endif
    }
  }
  return M;
}

template <typename Scalar>
Vector<Scalar> gen_vector(Eigen::Index mat_size, Eigen::Index seed) {
  Vector<Scalar> V = Vector<Scalar>::Zero(mat_size, 1);

  std::default_random_engine e(seed);
  std::normal_distribution<double> dist(0, 1);

  for (Eigen::Index row = 0; row < mat_size; row++) {
#ifdef REAL
    V(row, 0) = Scalar(dist(e));
#endif
#ifdef COMPLEX
    V(row, 0) = std::complex((typename Scalar::value_type)(dist(e)),
                             (typename Scalar::value_type)(dist(e)));
#endif
  }
  return V;
}

template <int order>
double matrix_difference(Matrix<IEEE, order> A, Matrix<Posit, order> B) {
  assert(A.rows() == B.rows() && A.cols() == B.cols());
  Matrix<IEEE, order> B_as_IEEE = Matrix<IEEE, order>::Zero(A.rows(), A.cols());

  for (Eigen::Index row = 0; row < A.rows(); row++) {
    for (Eigen::Index col = 0; col < A.cols(); col++) {
#ifdef REAL
      B_as_IEEE(row, col) = (IEEE)B(row, col);
#endif
#ifdef COMPLEX
      B_as_IEEE(row, col) =
          std::complex((typename IEEE::value_type)B(row, col).real(),
                       (typename IEEE::value_type)B(row, col).imag());
#endif
    }
  }

  Matrix<IEEE, order> errors = A - B_as_IEEE;
#ifdef REAL
  IEEE max = 0;
#endif
#ifdef COMPLEX
  typename IEEE::value_type max = 0;
#endif
  for (Eigen::Index row = 0; row < A.rows(); row++) {
    for (Eigen::Index col = 0; col < A.cols(); col++) {
#ifdef REAL
      IEEE error = std::abs(errors(row, col));
#endif
#ifdef COMPLEX
      typename IEEE::value_type error = std::abs(errors(row, col));
#endif
      if (error > max)
        max = error;
    }
  }
  return max;
}
double vector_difference(Vector<IEEE> A, Vector<Posit> B) {
  assert(A.rows() == B.rows() && A.cols() == B.cols());
  Vector<IEEE> B_as_IEEE = Vector<IEEE>::Zero(A.rows(), 1);

  for (Eigen::Index row = 0; row < A.rows(); row++) {
#ifdef REAL
    B_as_IEEE(row, 0) = (IEEE)B(row, 0);
#endif
#ifdef COMPLEX
    B_as_IEEE(row, 0) =
        std::complex((typename IEEE::value_type)B(row, 0).real(),
                     (typename IEEE::value_type)B(row, 0).imag());
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
    IEEE error = std::abs(errors(row, 0));
#endif
#ifdef COMPLEX
    typename IEEE::value_type error = std::abs(errors(row, 0));
#endif
    if (error > max)
      max = error;
  }
  return max;
}

int main() {
#ifdef LAZY
  Matrix<Posit, Eigen::ColMajor> Ap = gen_matrix<Posit, Eigen::ColMajor>(3, 0);
  Matrix<IEEE, Eigen::ColMajor> Ad = gen_matrix<IEEE, Eigen::ColMajor>(3, 0);

  Matrix<Posit, Eigen::ColMajor> Rp = Ap * Ap;
  Matrix<IEEE, Eigen::ColMajor> Rd = Ad * Ad;

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
#endif

#ifdef GEMM_COL
  Matrix<Posit, Eigen::ColMajor> Ap =
      gen_matrix<Posit, Eigen::ColMajor>(20, 25, 0);
  Matrix<IEEE, Eigen::ColMajor> Ad =
      gen_matrix<IEEE, Eigen::ColMajor>(20, 25, 0);
  Matrix<Posit, Eigen::ColMajor> Ap2 =
      gen_matrix<Posit, Eigen::ColMajor>(25, 20, 1);
  Matrix<IEEE, Eigen::ColMajor> Ad2 =
      gen_matrix<IEEE, Eigen::ColMajor>(25, 20, 1);
  Matrix<Posit, Eigen::ColMajor> Ap3 =
      gen_matrix<Posit, Eigen::ColMajor>(20, 20, 2);
  Matrix<IEEE, Eigen::ColMajor> Ad3 =
      gen_matrix<IEEE, Eigen::ColMajor>(20, 20, 2);

  // simple case
  Matrix<Posit, Eigen::ColMajor> Rp = Ap * Ap2;
  Matrix<IEEE, Eigen::ColMajor> Rd = Ad * Ad2;

  // check usage in simple case
  if ((eigen_usage_vector & ((uint64_t)EigenOverrideMask::GEMM_COLMAJOR |
                             (uint64_t)EigenOverrideMask::GENERIC_GEMM)) ==
      0x0) {
    std::cerr << "GEMM not used in simple case" << std::endl;
    return -1;
  }
  eigen_usage_vector = 0x0;

  // more complex case
  Matrix<Posit, Eigen::ColMajor> Rp2 = Ap * Ap2 + Ap3;
  Matrix<IEEE, Eigen::ColMajor> Rd2 = Ad * Ad2 + Ad3;

  if ((eigen_usage_vector & ((uint64_t)EigenOverrideMask::GEMM_COLMAJOR |
                             (uint64_t)EigenOverrideMask::GENERIC_GEMM)) ==
      0x0) {
    std::cerr << "GEMM not used in complex case" << std::endl;
    return -1;
  }

  double residual = matrix_difference(Rd, Rp);
  double residual2 = matrix_difference(Rd2, Rp2);

  if (residual < 1e-10 && residual2 < 1e-10) {
    return 0;
  } else {
    std::cerr << "Input matrices IEEE: \n"
              << Ad << "\n\n"
              << Ad2 << "\n\n"
              << Ad3 << std::endl;
    std::cerr << "Input matrices Posit: \n"
              << Ap << "\n\n"
              << Ap2 << "\n\n"
              << Ap3 << std::endl;

    std::cerr << "Result matrices IEEE: \n"
              << Rd2 << "\nPosit: \n"
              << Rp2 << std::endl;
    std::cerr << "GEMM A*B residual: " << residual << std::endl;
    std::cerr << "GEMM A*B+C residual: " << residual2 << std::endl;
    return -1;
  }
#endif

#ifdef GEMM_ROW
  Matrix<Posit, Eigen::RowMajor> Ap =
      gen_matrix<Posit, Eigen::RowMajor>(20, 25, 0);
  Matrix<IEEE, Eigen::RowMajor> Ad =
      gen_matrix<IEEE, Eigen::RowMajor>(20, 25, 0);
  Matrix<Posit, Eigen::RowMajor> Ap2 =
      gen_matrix<Posit, Eigen::RowMajor>(25, 20, 1);
  Matrix<IEEE, Eigen::RowMajor> Ad2 =
      gen_matrix<IEEE, Eigen::RowMajor>(25, 20, 1);
  Matrix<Posit, Eigen::RowMajor> Ap3 =
      gen_matrix<Posit, Eigen::RowMajor>(20, 20, 1);
  Matrix<IEEE, Eigen::RowMajor> Ad3 =
      gen_matrix<IEEE, Eigen::RowMajor>(20, 20, 1);

  // simple case
  Matrix<Posit, Eigen::RowMajor> Rp = Ap * Ap2;
  Matrix<IEEE, Eigen::RowMajor> Rd = Ad * Ad2;

  // check usage in simple case
  if ((eigen_usage_vector & ((uint64_t)EigenOverrideMask::GEMM_ROWMAJOR |
                             (uint64_t)EigenOverrideMask::GENERIC_GEMM)) ==
      0x0) {
    std::cerr << "GEMM not used in simple case" << std::endl;
    return -1;
  }
  eigen_usage_vector = 0x0;

  // more complex case
  Matrix<Posit, Eigen::RowMajor> Rp2 = Ap * Ap2 + Ap3;
  Matrix<IEEE, Eigen::RowMajor> Rd2 = Ad * Ad2 + Ad3;

  if ((eigen_usage_vector & ((uint64_t)EigenOverrideMask::GEMM_ROWMAJOR |
                             (uint64_t)EigenOverrideMask::GENERIC_GEMM)) ==
      0x0) {
    std::cerr << "GEMM not used in complex case" << std::endl;
    return -1;
  }

  double residual = matrix_difference(Rd, Rp);
  double residual2 = matrix_difference(Rd2, Rp2);

  if (residual < 1e-10 && residual2 < 1e-10) {
    return 0;
  } else {
    std::cerr << "Input matrices IEEE: \n"
              << Ad << "\n\n"
              << Ad2 << "\n\n"
              << Ad3 << std::endl;
    std::cerr << "Input matrices Posit: \n"
              << Ap << "\n\n"
              << Ap2 << "\n\n"
              << Ap3 << std::endl;

    std::cerr << "Result matrices IEEE: \n"
              << Rd2 << "\nPosit: \n"
              << Rp2 << std::endl;
    std::cerr << "GEMM A*B residual: " << residual << std::endl;
    std::cerr << "GEMM A*B+C residual: " << residual2 << std::endl;
    return -1;
  }
#endif

#ifdef GEMV_COL
  Matrix<Posit, Eigen::ColMajor> Ap =
      gen_matrix<Posit, Eigen::ColMajor>(20, 25, 0);
  Matrix<IEEE, Eigen::ColMajor> Ad =
      gen_matrix<IEEE, Eigen::ColMajor>(20, 25, 0);

  Vector<Posit> xp = gen_vector<Posit>(25, 1);
  Vector<IEEE> xd = gen_vector<IEEE>(25, 1);

  Vector<Posit> xp2 = gen_vector<Posit>(20, 1);
  Vector<IEEE> xd2 = gen_vector<IEEE>(20, 1);

  Vector<Posit> Rp = Ap * xp;
  Vector<IEEE> Rd = Ad * xd;
  Vector<Posit> Rp2 = Ap * xp + xp2;
  Vector<IEEE> Rd2 = Ad * xd + xd2;

  if ((eigen_usage_vector & ((uint64_t)EigenOverrideMask::GEMV_COLMAJOR |
                             (uint64_t)EigenOverrideMask::GENERIC_GEMV)) ==
      0x0) {
    std::cerr << "GEMV not used" << std::endl;
    return -1;
  }

  double residual = vector_difference(Rd, Rp);
  double residual2 = vector_difference(Rd2, Rp2);

  if (residual < 1e-10 && residual < 1e-10) {
    return 0;
  } else {
    std::cerr << "Result vectors IEEE: \n"
              << Rd << "\nPosit: \n"
              << Rp << std::endl;
    std::cerr << "Result vectors IEEE: \n"
              << Rd2 << "\nPosit: \n"
              << Rp2 << std::endl;
    std::cerr << "GEMV residual: " << residual << std::endl;
    std::cerr << "GEMV2 residual: " << residual2 << std::endl;
    return -1;
  }
#endif

#ifdef GEMV_ROW
  Matrix<Posit, Eigen::RowMajor> Ap =
      gen_matrix<Posit, Eigen::RowMajor>(20, 25, 0);
  Matrix<IEEE, Eigen::RowMajor> Ad =
      gen_matrix<IEEE, Eigen::RowMajor>(20, 25, 0);

  Vector<Posit> xp = gen_vector<Posit>(25, 1);
  Vector<IEEE> xd = gen_vector<IEEE>(25, 1);

  Vector<Posit> xp2 = gen_vector<Posit>(20, 1);
  Vector<IEEE> xd2 = gen_vector<IEEE>(20, 1);

  Vector<Posit> Rp = Ap * xp;
  Vector<IEEE> Rd = Ad * xd;
  Vector<Posit> Rp2 = Ap * xp + xp2;
  Vector<IEEE> Rd2 = Ad * xd + xd2;

  if ((eigen_usage_vector & ((uint64_t)EigenOverrideMask::GEMV_ROWMAJOR |
                             (uint64_t)EigenOverrideMask::GENERIC_GEMV)) ==
      0x0) {
    std::cerr << "GEMV not used" << std::endl;
    return -1;
  }

  double residual = vector_difference(Rd, Rp);
  double residual2 = vector_difference(Rd2, Rp2);

  if (residual < 1e-10 && residual < 1e-10) {
    return 0;
  } else {
    std::cerr << "Result vectors IEEE: \n"
              << Rd << "\nPosit: \n"
              << Rp << std::endl;
    std::cerr << "Result vectors IEEE: \n"
              << Rd2 << "\nPosit: \n"
              << Rp2 << std::endl;
    std::cerr << "GEMV residual: " << residual << std::endl;
    std::cerr << "GEMV2 residual: " << residual2 << std::endl;
    return -1;
  }
#endif

#ifdef _DOT
  Vector<Posit> xp = gen_vector<Posit>(3, 0);
  Vector<IEEE> xd = gen_vector<IEEE>(3, 0);
  Vector<Posit> xp2 = gen_vector<Posit>(3, 1);
  Vector<IEEE> xd2 = gen_vector<IEEE>(3, 1);

  Posit Rp = xp.dot(xp2);
  IEEE Rd = xd.dot(xd2);

  if ((eigen_usage_vector ^ (uint64_t)EigenOverrideMask::DOT) != 0x0) {
    std::cerr << "DOT not used" << std::endl;
    return -1;
  }

#ifdef REAL
  IEEE Rp_as_ieee = (IEEE)Rp;
#endif
#ifdef COMPLEX
  IEEE Rp_as_ieee = std::complex((typename IEEE::value_type)(Rp.real()),
                                 (typename IEEE::value_type)(Rp.imag()));
#endif
  double residual = std::abs(Rd) - std::abs(Rp_as_ieee);

  if (residual < 1e-10) {
    return 0;
  } else {
    std::cerr << "Results IEEE: \n" << Rd << "\nPosit: \n" << Rp << std::endl;
    std::cerr << "DOT residual: " << residual << std::endl;
    return -1;
  }

#endif

#ifdef _DOT_T
  Vector<Posit> xp = gen_vector<Posit>(3, 0);
  Vector<IEEE> xd = gen_vector<IEEE>(3, 0);
  Vector<Posit> xp2 = gen_vector<Posit>(3, 1);
  Vector<IEEE> xd2 = gen_vector<IEEE>(3, 1);

  Posit Rp = xp.transpose().dot(xp2);
  IEEE Rd = xd.transpose().dot(xd2);

  if ((eigen_usage_vector ^ (uint64_t)EigenOverrideMask::DOT_T) != 0x0) {
    std::cerr << "DOT_T not used" << std::endl;
    return -1;
  }

#ifdef REAL
  IEEE Rp_as_ieee = (IEEE)Rp;
#endif
#ifdef COMPLEX
  IEEE Rp_as_ieee = std::complex((typename IEEE::value_type)(Rp.real()),
                                 (typename IEEE::value_type)(Rp.imag()));
#endif
  double residual = std::abs(Rd) - std::abs(Rp_as_ieee);

  if (residual < 1e-10) {
    return 0;
  } else {
    std::cerr << "Results IEEE: \n" << Rd << "\nPosit: \n" << Rp << std::endl;
    std::cerr << "DOT_T residual: " << residual << std::endl;
    return -1;
  }

#endif
}