#ifndef STD_INTEGRATION_HPP
#define STD_INTEGRATION_HPP

#include <cmath>
#include <universal/number/posit/posit.hpp>
#include <universal/number/posit/mathlib.hpp>
#include "internal/posit_concepts.hpp"

#define make_std_posit_fun(name) template <typename T> requires ScalarIsPosit<T> T name(T t) { return sw::universal::name(t); }

namespace std {
//     template <typename T> requires ScalarIsPosit<T>
//     T sin(T t) {
//         return sw::universal::sin(t);
//     }
//     template <typename T> requires ScalarIsPosit<T>
//     T cos(T t) {
//         return sw::universal::cos(t);
//     }
//     template <typename T> requires ScalarIsPosit<T>
//     T sqrt(T t) {
//         return sw::universal::sqrt(t);
//     }
//     template <typename T> requires ScalarIsPosit<T>
//     T abs(T t) {
//         return sw::universal::abs(t);
//     }
//     template <typename T> requires ScalarIsPosit<T>
//     T floor(T t) {
//         return sw::universal::floor(t);
//     }
//     template <typename T> requires ScalarIsPosit<T>
//     T ceil(T t) {
//         return sw::universal::ceil(t);
//     }
//     template <typename T> requires ScalarIsPosit<T>
//     T exp(T t) {
//         return sw::universal::exp(t);
//     }
//     template <typename T> requires ScalarIsPosit<T>
//     T log(T t) {
//         return sw::universal::log(t);
//     }
    make_std_posit_fun(sin)
    make_std_posit_fun(cos)
    make_std_posit_fun(sqrt)
    make_std_posit_fun(abs)
    make_std_posit_fun(floor)
    make_std_posit_fun(ceil)
    make_std_posit_fun(exp)
    make_std_posit_fun(log)
    make_std_posit_fun(atan)

    template <typename T> requires ScalarIsPosit<T>
    T pow(T base, T exp) {
        return sw::universal::pow(base, exp);
    }
}

#endif