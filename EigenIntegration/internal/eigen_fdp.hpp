#ifndef EIGEN_FDP_HPP
#define EIGEN_FDP_HPP

#include "posit_concepts.hpp"
#include <Eigen/Dense>
#include <universal/number/posit/quire.hpp>

template <typename Lhs, typename Rhs>
requires HasPositOrComplexPositScalar<Lhs>
typename Lhs::Scalar eigen_fdp(const Lhs &lhs, const Rhs &rhs) {
  // based on the implementation in the stillwater universal library
  Eigen::Index depth = lhs.rows() != 1 ? lhs.rows() : lhs.cols();

  assert(lhs.rows() == 1 || lhs.cols() == 1);
  assert(rhs.rows() == 1 || rhs.cols() == 1);

  if constexpr (!is_complex<typename Lhs::Scalar>) {
    constexpr size_t nbits = Lhs::Scalar::nbits;
    constexpr size_t es = Lhs::Scalar::es;
    constexpr size_t capacity = 20; // support vectors up to 1M elements
    sw::universal::quire<nbits, es, capacity> q(0);

    // decide which axes have to be iterated over. Each condition is evaluated
    // only once per fdp.
    if (lhs.rows() == 1) {
      if (rhs.rows() == 1) {
        assert(lhs.cols() == rhs.cols());
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q += sw::universal::quire_mul(lhs(0, idx), rhs(0, idx));
        }
      } else {
        assert(lhs.cols() == rhs.rows());
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q += sw::universal::quire_mul(lhs(0, idx), rhs(idx, 0));
        }
      }
    } else {
      if (rhs.rows() == 1) {
        assert(lhs.rows() == rhs.cols());
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q += sw::universal::quire_mul(lhs(idx, 0), rhs(0, idx));
        }
      } else {
        assert(lhs.rows() == rhs.rows());
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q += sw::universal::quire_mul(lhs(idx, 0), rhs(idx, 0));
        }
      }
    }
    typename Lhs::Scalar sum;
    convert(q.to_value(), sum);
    return sum;
  } else {
    constexpr size_t nbits = Lhs::Scalar::value_type::nbits;
    constexpr size_t es = Lhs::Scalar::value_type::es;
    constexpr size_t capacity = 20; // support vectors up to 1M elements
    sw::universal::quire<nbits, es, capacity> q_real(0);
    sw::universal::quire<nbits, es, capacity> q_imag(0);

    // decide which axes have to be iterated over. Each condition is evaluated
    // only once per fdp.
    if (lhs.rows() == 1) {
      if (rhs.rows() == 1) {
        assert(lhs.cols() == rhs.cols());
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q_real +=
              sw::universal::quire_mul(lhs(0, idx).real(), rhs(0, idx).real());
          q_real -=
              sw::universal::quire_mul(lhs(0, idx).imag(), rhs(0, idx).imag());

          q_imag +=
              sw::universal::quire_mul(lhs(0, idx).real(), rhs(0, idx).imag());
          q_imag +=
              sw::universal::quire_mul(lhs(0, idx).imag(), rhs(0, idx).real());
        }
      } else {
        assert(lhs.cols() == rhs.rows());
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q_real +=
              sw::universal::quire_mul(lhs(0, idx).real(), rhs(idx, 0).real());
          q_real -=
              sw::universal::quire_mul(lhs(0, idx).imag(), rhs(idx, 0).imag());

          q_imag +=
              sw::universal::quire_mul(lhs(0, idx).real(), rhs(idx, 0).imag());
          q_imag +=
              sw::universal::quire_mul(lhs(0, idx).imag(), rhs(idx, 0).real());
        }
      }
    } else {
      if (rhs.rows() == 1) {
        assert(lhs.rows() == rhs.cols());
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q_real +=
              sw::universal::quire_mul(lhs(idx, 0).real(), rhs(0, idx).real());
          q_real -=
              sw::universal::quire_mul(lhs(idx, 0).imag(), rhs(0, idx).imag());

          q_imag +=
              sw::universal::quire_mul(lhs(idx, 0).real(), rhs(0, idx).imag());
          q_imag +=
              sw::universal::quire_mul(lhs(idx, 0).imag(), rhs(0, idx).real());
        }
      } else {
        assert(lhs.rows() == rhs.rows());
        for (Eigen::Index idx = 0; idx < depth; idx++) {
          q_real +=
              sw::universal::quire_mul(lhs(idx, 0).real(), rhs(idx, 0).real());
          q_real -=
              sw::universal::quire_mul(lhs(idx, 0).imag(), rhs(idx, 0).imag());

          q_imag +=
              sw::universal::quire_mul(lhs(idx, 0).real(), rhs(idx, 0).imag());
          q_imag +=
              sw::universal::quire_mul(lhs(idx, 0).imag(), rhs(idx, 0).real());
        }
      }
    }

    typename Lhs::Scalar::value_type sum_real;
    typename Lhs::Scalar::value_type sum_imag;
    convert(q_real.to_value(), sum_real);
    convert(q_imag.to_value(), sum_imag);
    return std::complex(sum_real, sum_imag);
  }
}

#endif