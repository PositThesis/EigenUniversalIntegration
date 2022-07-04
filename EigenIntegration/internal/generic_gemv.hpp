#ifndef GENERIC_GEMV_OVERRIDE_HPP
#define GENERIC_GEMV_OVERRIDE_HPP

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

template <typename Lhs, typename Rhs>
requires HasPositOrComplexPositScalar<Lhs>
struct generic_product_impl<Lhs, Rhs, DenseShape, DenseShape, GemvProduct>
    : generic_product_impl_base<
          Lhs, Rhs,
          generic_product_impl<Lhs, Rhs, DenseShape, DenseShape, GemvProduct>> {
  typedef typename nested_eval<Lhs, 1>::type LhsNested;
  typedef typename nested_eval<Rhs, 1>::type RhsNested;
  typedef typename Product<Lhs, Rhs>::Scalar Scalar;
  enum { Side = Lhs::IsVectorAtCompileTime ? OnTheLeft : OnTheRight };
  typedef typename internal::remove_all<typename internal::conditional<
      int(Side) == OnTheRight, LhsNested, RhsNested>::type>::type MatrixType;

  template <typename Dest>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void
  scaleAndAddTo(Dest &dst, const Lhs &lhs, const Rhs &rhs,
                const Scalar &alpha) {

#ifdef TEST_EIGEN_USAGE
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::GENERIC_GEMV;
#endif

    // Fallback to inner product if both the lhs and rhs is a runtime vector.
    if (lhs.rows() == 1 && rhs.cols() == 1) {
      dst.coeffRef(0, 0) += alpha * lhs.row(0).conjugate().dot(rhs.col(0));
      return;
    }
    LhsNested actual_lhs(lhs);
    RhsNested actual_rhs(rhs);
    // calculate fdp
#pragma omp parallel for collapse(2)
    for (int row = 0; row < dst.rows(); row++) {
      for (int col = 0; col < dst.cols();
           col++) { // technically, dst.cols "should" be one, but maybe this is
                    // used for vector*matrix aswell?
        dst.coeffRef(row, col) +=
            alpha * eigen_fdp(actual_lhs.row(row), actual_rhs.col(col));
      }
    }
  }
};
} // namespace internal
} // namespace Eigen
#endif