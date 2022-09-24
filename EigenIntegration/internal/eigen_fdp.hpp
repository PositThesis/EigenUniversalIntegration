#ifndef EIGEN_FDP_HPP
#define EIGEN_FDP_HPP

#include "posit_concepts.hpp"
#include <Eigen/Dense>
#include <universal/number/posit/quire.hpp>
#include "test_bitmask.hpp"

template <typename Lhs, typename Rhs> struct MultiplyReturnType {
  typedef Lhs::Scalar type;
};

template <typename Lhs, typename Rhs>
requires (HasComplexPositScalar<Lhs> && HasPositScalar<Rhs>)
    struct MultiplyReturnType<Lhs, Rhs> {
  typedef Lhs::Scalar type;
};

template <typename Lhs, typename Rhs>
requires HasComplexPositScalar<Rhs>
    struct MultiplyReturnType<Lhs, Rhs> {
  typedef Rhs::Scalar type;
};


template <typename Lhs, typename Rhs>
requires HasPositOrComplexPositScalar<Lhs>
typename MultiplyReturnType<Lhs, Rhs>::type eigen_fdp(const Lhs &lhs, const Rhs &rhs) {
  // based on the implementation in the stillwater universal library
  Eigen::Index depth = lhs.rows() != 1 ? lhs.rows() : lhs.cols();

  assert(lhs.rows() == 1 || lhs.cols() == 1);
  assert(rhs.rows() == 1 || rhs.cols() == 1);
  assert(lhs.rows() * lhs.cols() == rhs.rows() * rhs.cols());

  // first, turn sparse vectors into dense vectors by extracting the non-zero elements
  if constexpr (is_sparse<Lhs> && is_sparse<Rhs>) {
    Eigen::VectorX<typename Lhs::Scalar> lhs_dense = Eigen::VectorX<typename Lhs::Scalar>::Zero(depth);
    Eigen::VectorX<typename Rhs::Scalar> rhs_dense = Eigen::VectorX<typename Rhs::Scalar>::Zero(depth);

    // about that 0: according to some simple tests, it should always work out, given our premise that one of the axes has to have unit size
    Eigen::Index current_index = 0;
    typename Eigen::InnerIterator<Lhs> lit(lhs, 0);
    typename Eigen::InnerIterator<Rhs> rit(rhs, 0);
    while(lit && rit) {
      if (lit.index() < rit.index()) {
        ++lit;
        continue;
      }
      if (lit.index() > rit.index()) {
        ++rit;
        continue;
      }
      if (lit.index() == rit.index()) {
        lhs_dense(current_index) = lit.value();
        rhs_dense(current_index) = rit.value();
        current_index++;
        ++lit;
        ++rit;
      }
    }

    lhs_dense.conservativeResize(current_index);
    rhs_dense.conservativeResize(current_index);

    #ifdef TEST_EIGEN_USAGE
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::SPARSE_LHS;
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::SPARSE_RHS;
    #endif
    return eigen_fdp(lhs_dense, rhs_dense);

  } else if constexpr (is_sparse<Lhs>) {
    Eigen::VectorX<typename Lhs::Scalar> lhs_dense = Eigen::VectorX<typename Lhs::Scalar>::Zero(depth);
    Eigen::VectorX<typename Rhs::Scalar> rhs_dense = Eigen::VectorX<typename Rhs::Scalar>::Zero(depth);

    Eigen::Index current_index = 0;
    for(typename Eigen::InnerIterator<Lhs> it(lhs, 0); it; ++it) {
      lhs_dense(current_index) = it.value();
      rhs_dense(current_index) = rhs(it.index());
      current_index++;
    }
    lhs_dense.conservativeResize(current_index);
    rhs_dense.conservativeResize(current_index);
    #ifdef TEST_EIGEN_USAGE
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::SPARSE_LHS;
    #endif
    return eigen_fdp(lhs_dense, rhs_dense);
  } else if constexpr (is_sparse<Rhs>) {
    Eigen::VectorX<typename Lhs::Scalar> lhs_dense = Eigen::VectorX<typename Lhs::Scalar>::Zero(depth);
    Eigen::VectorX<typename Rhs::Scalar> rhs_dense = Eigen::VectorX<typename Rhs::Scalar>::Zero(depth);

    Eigen::Index current_index = 0;
    for(typename Eigen::InnerIterator<Rhs> it(rhs, 0); it; ++it) {
      lhs_dense(current_index) = lhs(it.index());
      rhs_dense(current_index) = it.value();
      current_index++;
    }
    lhs_dense.conservativeResize(current_index);
    rhs_dense.conservativeResize(current_index);

    #ifdef TEST_EIGEN_USAGE
    eigen_usage_vector |= (uint64_t)EigenOverrideMask::SPARSE_RHS;
    #endif
    return eigen_fdp(lhs_dense, rhs_dense);
  } else {
    // then perform the actual computation

    if constexpr (!is_complex<typename Lhs::Scalar>) {
      constexpr size_t nbits = Lhs::Scalar::nbits;
      constexpr size_t es = Lhs::Scalar::es;
      constexpr size_t capacity = 20; // support vectors up to 1M elements

      if constexpr (!is_complex<typename Rhs::Scalar>) { // Lhs and Rhs real
        sw::universal::quire<nbits, es, capacity> q(0);
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q += sw::universal::quire_mul(lhs.coeff(idx), rhs.coeff(idx));
        }
        typename Lhs::Scalar sum;
        convert(q.to_value(), sum);
        return sum;
      } else { // Lhs real, rhs complex
        constexpr size_t nbits = Rhs::Scalar::value_type::nbits;
        constexpr size_t es = Rhs::Scalar::value_type::es;
        constexpr size_t capacity = 20; // support vectors up to 1M elements
        sw::universal::quire<nbits, es, capacity> q_real(0);
        sw::universal::quire<nbits, es, capacity> q_imag(0);
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q_real +=
              sw::universal::quire_mul(lhs.coeff(idx), rhs.coeff(idx).real());
          // left here as comment for completeness
          // q_real -=
          //     sw::universal::quire_mul(Scalar(0), rhs.coeff(idx).imag());

          q_imag +=
              sw::universal::quire_mul(lhs.coeff(idx), rhs.coeff(idx).imag());
          // q_imag +=
          //     sw::universal::quire_mul(Scalar(0), rhs.coeff(idx).real());
        }

        typename Rhs::Scalar::value_type sum_real;
        typename Rhs::Scalar::value_type sum_imag;
        convert(q_real.to_value(), sum_real);
        convert(q_imag.to_value(), sum_imag);
        return std::complex(sum_real, sum_imag);
      }
    } else {
      constexpr size_t nbits = Lhs::Scalar::value_type::nbits;
      constexpr size_t es = Lhs::Scalar::value_type::es;
      constexpr size_t capacity = 20; // support vectors up to 1M elements
      sw::universal::quire<nbits, es, capacity> q_real(0);
      sw::universal::quire<nbits, es, capacity> q_imag(0);

      if constexpr (!is_complex<typename Rhs::Scalar>) { // Lhs complex, Rhs real
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q_real +=
              sw::universal::quire_mul(lhs.coeff(idx).real(), rhs.coeff(idx));
          // left here as comments for completeness
          // q_real -=
          //      sw::universal::quire_mul(lhs.coeff(idx).imag(), Scalar(0));

          // q_imag +=
          //     sw::universal::quire_mul(lhs.coeff(idx).real(), Scalar(0));
          q_imag +=
              sw::universal::quire_mul(lhs.coeff(idx).imag(), rhs.coeff(idx));
        }

        typename Lhs::Scalar::value_type sum_real;
        typename Lhs::Scalar::value_type sum_imag;
        convert(q_real.to_value(), sum_real);
        convert(q_imag.to_value(), sum_imag);
        return std::complex(sum_real, sum_imag);
      } else { // Lhs and Rhs complex
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q_real +=
              sw::universal::quire_mul(lhs.coeff(idx).real(), rhs.coeff(idx).real());
          q_real -=
              sw::universal::quire_mul(lhs.coeff(idx).imag(), rhs.coeff(idx).imag());

          q_imag +=
              sw::universal::quire_mul(lhs.coeff(idx).real(), rhs.coeff(idx).imag());
          q_imag +=
              sw::universal::quire_mul(lhs.coeff(idx).imag(), rhs.coeff(idx).real());
        }
        typename Lhs::Scalar::value_type sum_real;
        typename Lhs::Scalar::value_type sum_imag;
        convert(q_real.to_value(), sum_real);
        convert(q_imag.to_value(), sum_imag);
        return std::complex(sum_real, sum_imag);
      }
    }
  }
}

#endif