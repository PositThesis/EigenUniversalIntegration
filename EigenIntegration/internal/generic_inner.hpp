#ifndef GENERIC_INNER_OVERRIDE_HPP
#define GENERIC_INNER_OVERRIDE_HPP

#include "eigen_fdp.hpp"
#include "posit_concepts.hpp"
#include <Eigen/Dense>
#include <omp.h>
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
struct generic_product_impl<Lhs, Rhs, DenseShape, DenseShape, InnerProduct>
    : generic_product_impl_base<
          Lhs, Rhs,
          generic_product_impl<Lhs, Rhs, DenseShape, DenseShape,
                               InnerProduct>> {
  typedef typename Product<Lhs, Rhs>::Scalar Scalar;
  typedef typename Lhs::Scalar LhsScalar;
  typedef typename Rhs::Scalar RhsScalar;

  typedef internal::blas_traits<Lhs> LhsBlasTraits;
  typedef typename LhsBlasTraits::DirectLinearAccessType ActualLhsType;
  typedef
      typename internal::remove_all<ActualLhsType>::type ActualLhsTypeCleaned;

  typedef internal::blas_traits<Rhs> RhsBlasTraits;
  typedef typename RhsBlasTraits::DirectLinearAccessType ActualRhsType;
  typedef
      typename internal::remove_all<ActualRhsType>::type ActualRhsTypeCleaned;

  enum {
    MaxDepthAtCompileTime = EIGEN_SIZE_MIN_PREFER_FIXED(
        Lhs::MaxColsAtCompileTime, Rhs::MaxRowsAtCompileTime)
  };

  typedef generic_product_impl<Lhs, Rhs, DenseShape, DenseShape,
                               CoeffBasedProductMode>
      lazyproduct;

  template <typename Dst>
  static void evalTo(Dst &dst, const Lhs &lhs, const Rhs &rhs) {
    // See http://eigen.tuxfamily.org/bz/show_bug.cgi?id=404 for a discussion
    // and helper program to determine the following heuristic.
    // EIGEN_GEMM_TO_COEFFBASED_THRESHOLD is typically defined to 20 in
    // GeneralProduct.h, unless it has been specialized by the user or for a
    // given architecture. Note that the condition rhs.rows()>0 was required
    // because lazy product is (was?) not happy with empty inputs. I'm not sure
    // it is still required.

    dst.setZero();
    scaleAndAddTo(dst, lhs, rhs, Scalar(1));
  }

  template <typename Dst>
  static void addTo(Dst &dst, const Lhs &lhs, const Rhs &rhs) {
    scaleAndAddTo(dst, lhs, rhs, Scalar(1));
  }

  template <typename Dst>
  static void subTo(Dst &dst, const Lhs &lhs, const Rhs &rhs) {
    scaleAndAddTo(dst, lhs, rhs, Scalar(-1));
  }

  template <typename Dest>
  static void scaleAndAddTo(Dest &dst, const Lhs &a_lhs, const Rhs &a_rhs,
                            const Scalar &alpha) {
#ifdef TEST_EIGEN_USAGE
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::GENERIC_INNER;
#endif
    eigen_assert(dst.rows() == a_lhs.rows() && dst.cols() == a_rhs.cols());
    if (a_lhs.cols() == 0 || a_lhs.rows() == 0 || a_rhs.cols() == 0)
      return;

    typename internal::add_const_on_value_type<ActualLhsType>::type lhs =
        LhsBlasTraits::extract(a_lhs);
    typename internal::add_const_on_value_type<ActualRhsType>::type rhs =
        RhsBlasTraits::extract(a_rhs);

    Scalar actualAlpha = combine_scalar_factors(alpha, a_lhs, a_rhs);

    // calculate fdp
    dst.coeffRef(0, 0) += actualAlpha * eigen_fdp(lhs, rhs);
  }
};

} // namespace internal
} // namespace Eigen
#endif