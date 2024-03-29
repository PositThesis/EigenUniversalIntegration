
#include "eigen_fdp.hpp"
#include "posit_concepts.hpp"
#include <Eigen/Dense>
#include <omp.h>
#include <sys/types.h>
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
struct generic_product_impl<Lhs, Rhs, LeftSideShape__, RightSideShape__, DefinedProductType__>
    : generic_product_impl_base<
          Lhs, Rhs,
          generic_product_impl<Lhs, Rhs, LeftSideShape__, RightSideShape__, DefinedProductType__>> {
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

  template <typename Dst>
  static void evalTo(Dst &dst, const Lhs &lhs, const Rhs &rhs) {
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
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::GENERIC_GEMM;
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::GENERIC_GEMV;
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::GENERIC_LAZY;
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
    bool blocking = lhs.rows() + lhs.cols() + rhs.cols() > 20; // same heuristic as in eigen
    if (!blocking) {
      // only when the problem is small -> no omp needed
      // #pragma omp parallel for collapse(2)
      for (int row = 0; row < dst.rows(); row++) {
        for (int col = 0; col < dst.cols(); col++) {
          if constexpr (is_sparse<Dest>) {
            // only insert into the matrix if it is non zero
            Scalar result = actualAlpha * eigen_fdp(lhs.row(row), rhs.col(col));
            if (result != Scalar(0)) {
              // #pragma omp critical
              dst.coeffRef(row, col) += result;
            }
            std::cout << "is sparse" << std::endl;
            #ifdef TEST_EIGEN_USAGE
            eigen_usage_vector |= (uint64_t)EigenOverrideMask::SPARSE_DST;
            #endif
          } else {
            dst.coeffRef(row, col) +=
                actualAlpha * eigen_fdp(lhs.row(row), rhs.col(col));
          }
        }
      }
    } else {
      int block_size = 10; // some other heuristic?
      for (int blocking_row_start = 0; blocking_row_start < dst.rows();
           blocking_row_start += block_size) {
        int max_row = blocking_row_start + block_size > dst.rows()
                          ? dst.rows()
                          : blocking_row_start + block_size;
        for (int blocking_col_start = 0; blocking_col_start < dst.cols();
             blocking_col_start += block_size) {
          int max_col = blocking_col_start + block_size > dst.cols()
                            ? dst.cols()
                            : blocking_col_start + block_size;
          // #pragma omp parallel for collapse(2)
          for (int row = blocking_row_start; row < max_row; row++) {
            for (int col = blocking_col_start; col < max_col; col++) {
              if constexpr (is_sparse<Dest>) {
                // only insert into the matrix if it is non zero
                Scalar result = actualAlpha * eigen_fdp(lhs.row(row), rhs.col(col));
                if (result != Scalar(0)) {
                  // #pragma omp critical
                  dst.coeffRef(row, col) += result;
                }
                #ifdef TEST_EIGEN_USAGE
                eigen_usage_vector |= (uint64_t)EigenOverrideMask::SPARSE_DST;
                #endif
              } else {
                dst.coeffRef(row, col) +=
                    actualAlpha * eigen_fdp(lhs.row(row), rhs.col(col));
              }
            }
          }
        }
      }
    }
  }
};

} // namespace internal
} // namespace Eigen