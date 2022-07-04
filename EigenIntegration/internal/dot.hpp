#ifndef DOT_OVERRIDE_HPP
#define DOT_OVERRIDE_HPP

#include "posit_concepts.hpp"
#include <Eigen/Dense>
#include <universal/number/posit/fdp.hpp>
#include <universal/number/posit/posit.hpp>
#include <universal/traits/posit_traits.hpp>

#ifdef TEST_EIGEN_USAGE
#include "test_bitmask.hpp"
#endif

namespace Eigen {
namespace internal {

using Eigen::Matrix;

// Overrides for real dot products

template <typename T, typename U>
requires HasPositScalar<T>
struct dot_nocheck<T, U, false> {
  typedef scalar_conj_product_op<typename traits<T>::Scalar,
                                 typename traits<U>::Scalar>
      conj_prod;
  typedef typename conj_prod::result_type ResScalar;
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE
  static ResScalar run(const MatrixBase<T> &a, const MatrixBase<U> &b) {
    std::vector<typename T::Scalar> lhs_vec;
    std::vector<typename T::Scalar> rhs_vec;

#ifdef TEST_EIGEN_USAGE
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::DOT;
#endif

    if (a.rows() == 1) {
      for (int idx = 0; idx < a.cols(); idx++) {
        lhs_vec.push_back(a(0, idx));
        rhs_vec.push_back(b(0, idx));
      }
    } else {
      for (int idx = 0; idx < a.rows(); idx++) {
        lhs_vec.push_back(a(idx, 0));
        rhs_vec.push_back(b(idx, 0));
      }
    }

    return sw::universal::fdp(lhs_vec, rhs_vec);
  }
};

template <typename T, typename U>
requires HasPositScalar<T>
struct dot_nocheck<T, U, true> {
  typedef scalar_conj_product_op<typename traits<T>::Scalar,
                                 typename traits<U>::Scalar>
      conj_prod;
  typedef typename conj_prod::result_type ResScalar;
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE
  static ResScalar run(const MatrixBase<T> &a, const MatrixBase<U> &b) {
#ifdef TEST_EIGEN_USAGE
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::DOT_T;
#endif

    std::vector<typename T::Scalar> lhs_vec;
    std::vector<typename T::Scalar> rhs_vec;

    if (a.rows() == 1) {
      for (int idx = 0; idx < a.cols(); idx++) {
        lhs_vec.push_back(a(0, idx));
        rhs_vec.push_back(b(idx, 0));
      }
    } else {
      for (int idx = 0; idx < a.rows(); idx++) {
        lhs_vec.push_back(a(idx, 0));
        rhs_vec.push_back(b(0, idx));
      }
    }

    return sw::universal::fdp(lhs_vec, rhs_vec);
  }
};

// Overrides for complex dot products

template <typename T, typename U>
requires HasComplexPositScalar<T>
struct dot_nocheck<T, U, false> {
  typedef scalar_conj_product_op<typename traits<T>::Scalar,
                                 typename traits<U>::Scalar>
      conj_prod;
  typedef typename conj_prod::result_type ResScalar;
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE
  static ResScalar run(const MatrixBase<T> &a, const MatrixBase<U> &b) {

#ifdef TEST_EIGEN_USAGE
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::DOT;
#endif
    std::vector<typename T::Scalar::value_type> lhs_vec;
    std::vector<typename T::Scalar::value_type> rhs_vec;
    std::vector<typename T::Scalar::value_type> rhs_vec_swapped;

    if (a.rows() == 1) {
      for (int idx = 0; idx < a.cols(); idx++) {
        lhs_vec.push_back(a(0, idx).real());
        lhs_vec.push_back(a(0, idx).imag());
        rhs_vec.push_back(b(0, idx).real());
        rhs_vec.push_back(b(0, idx).imag());
        rhs_vec_swapped.push_back(b(0, idx).imag());
        rhs_vec_swapped.push_back(-b(0, idx).real());
      }
    } else {
      for (int idx = 0; idx < a.rows(); idx++) {
        lhs_vec.push_back(a(idx, 0).real());
        lhs_vec.push_back(a(idx, 0).imag());
        rhs_vec.push_back(b(idx, 0).real());
        rhs_vec.push_back(b(idx, 0).imag());
        rhs_vec_swapped.push_back(b(idx, 0).imag());
        rhs_vec_swapped.push_back(-b(idx, 0).real());
      }
    }

    return std::complex(sw::universal::fdp(lhs_vec, rhs_vec),
                        sw::universal::fdp(lhs_vec, rhs_vec_swapped));
  }
};

// Note to self: Read the eigen documentation the next time.
// For complex numbers, this returns the hermitian dot product
// this means that the lhs get conjugated. I also originally thought the right
// side gets conjugated, so the signs of the rhs terms is manipulated.

template <typename T, typename U>
requires HasComplexPositScalar<T>
struct dot_nocheck<T, U, true> {
  typedef scalar_conj_product_op<typename traits<T>::Scalar,
                                 typename traits<U>::Scalar>
      conj_prod;
  typedef typename conj_prod::result_type ResScalar;
  EIGEN_DEVICE_FUNC
  EIGEN_STRONG_INLINE
  static ResScalar run(const MatrixBase<T> &a, const MatrixBase<U> &b) {
#ifdef TEST_EIGEN_USAGE
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::DOT_T;
#endif

    std::vector<typename T::Scalar::value_type> lhs_vec;
    std::vector<typename T::Scalar::value_type> rhs_vec;
    std::vector<typename T::Scalar::value_type> rhs_vec_swapped;

    if (a.rows() == 1) {
      for (int idx = 0; idx < a.cols(); idx++) {
        lhs_vec.push_back(a(0, idx).real());
        lhs_vec.push_back(a(0, idx).imag());
        rhs_vec.push_back(b(idx, 0).real());
        rhs_vec.push_back(b(idx, 0).imag());
        rhs_vec_swapped.push_back(b(idx, 0).imag());
        rhs_vec_swapped.push_back(-b(idx, 0).real());
      }
    } else {
      for (int idx = 0; idx < a.rows(); idx++) {
        lhs_vec.push_back(a(idx, 0).real());
        lhs_vec.push_back(a(idx, 0).imag());
        rhs_vec.push_back(b(0, idx).real());
        rhs_vec.push_back(b(0, idx).imag());
        rhs_vec_swapped.push_back(b(0, idx).imag());
        rhs_vec_swapped.push_back(-b(0, idx).real());
      }
    }

    return std::complex(sw::universal::fdp(lhs_vec, rhs_vec),
                        sw::universal::fdp(lhs_vec, rhs_vec_swapped));
  }
};

} // namespace internal
} // namespace Eigen
#endif