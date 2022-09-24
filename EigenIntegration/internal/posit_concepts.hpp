#ifndef POSIT_CONCEPTS_HPP
#define POSIT_CONCEPTS_HPP

#include <complex>
#include <concepts>
#include <type_traits>
#include <universal/number/posit/posit.hpp>
#include <universal/traits/posit_traits.hpp>
#include <Eigen/Sparse>

// complex number traits, adapted from the stillwater universal library's is_posit trait
template <typename _Ty> struct is_complex_trait : std::false_type {};

template <typename InnerScalar>
struct is_complex_trait<std::complex<InnerScalar>> : std::true_type {};

template <typename Num>
constexpr bool is_complex = is_complex_trait<Num>::value;

template <typename _Ty> struct is_complex_posit_trait : std::false_type {};
template <size_t nbits, size_t es>
struct is_complex_posit_trait<std::complex<sw::universal::posit<nbits, es>>>
    : std::true_type {};

template <typename Num>
constexpr bool is_complex_posit = is_complex_posit_trait<Num>::value;

template <typename Num>
constexpr bool is_posit_or_complex_posit =
    is_complex_posit<Num> || sw::universal::is_posit<Num>;

template <typename Num>
constexpr bool is_real_posit = sw::universal::is_posit<Num> && !is_complex<Num>;

template <typename Num, typename Type = void>
using enable_if_complex = std::enable_if_t<is_complex<Num>, Type>;

// Real Valued posits
template <typename A>
concept HasPositScalar = sw::universal::is_posit<typename A::Scalar>;

template <typename A>
concept HasRealPositScalar = sw::universal::is_posit<typename A::Scalar>;

template <typename Scalar>
concept ScalarIsPosit = sw::universal::is_posit<Scalar>;

// Complex Values in general
template <typename A>
concept HasComplexScalar = is_complex<typename A::Scalar>;

template <typename Scalar>
concept ScalarIsComplex = is_complex<Scalar>;

// Complex Valued posits
template <typename A>
concept HasComplexPositScalar = is_complex_posit<typename A::Scalar>;

template <typename Scalar>
concept ScalarIsComplexPosit = is_complex_posit<Scalar>;

template <typename A>
concept HasPositOrComplexPositScalar =
    is_posit_or_complex_posit<typename A::Scalar>;


// sparse matrices/vectors
template <typename _Ty> struct is_sparse_vector_trait : std::false_type {};
template <typename InnerScalar>
struct is_sparse_vector_trait<Eigen::SparseVector<InnerScalar>> : std::true_type {};
template<typename _Ty>
constexpr bool is_sparse_vector = is_sparse_vector_trait<typename _Ty::Base>::value;

template <typename StorageKind> struct is_sparse_trait : std::false_type {};
template <>
struct is_sparse_trait<Eigen::Sparse> : std::true_type {};
template<typename _Ty>
constexpr bool is_sparse = is_sparse_trait<typename _Ty::StorageKind>::value;

#endif