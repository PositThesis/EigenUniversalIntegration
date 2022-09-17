#ifndef STD_INTEGRATION_HPP
#define STD_INTEGRATION_HPP

#include <string>
#include <cmath>
#include <universal/number/posit/posit.hpp>
#include <universal/number/posit/mathlib.hpp>
#include "internal/posit_concepts.hpp"

#define make_std_posit_fun(name) template <typename T> requires ScalarIsPosit<T> T name(T t) { return sw::universal::name(t); }

namespace std {
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
    std::string to_string(T t) {
        return sw::universal::to_string(t);
    }

    template <typename T> requires ScalarIsPosit<T>
    T pow(T base, T exp) {
        return sw::universal::pow(base, exp);
    }

    template <typename T> requires ScalarIsPosit<T>
    T atan2(T y, T x) {
        return sw::universal::atan2(y, x);
    }
}

#endif