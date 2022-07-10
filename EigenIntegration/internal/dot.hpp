#ifndef DOT_OVERRIDE_HPP
#define DOT_OVERRIDE_HPP

#include "eigen_fdp.hpp"
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

template <typename T, typename U>
requires HasPositOrComplexPositScalar<T>
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
    return eigen_fdp(a.conjugate(), b);
  }
};

template <typename T, typename U>
requires HasPositOrComplexPositScalar<T>
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
    return eigen_fdp(a.conjugate(), b);
  }
};

} // namespace internal
} // namespace Eigen
#endif