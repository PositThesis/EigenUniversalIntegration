#ifndef STD_INTEGRATION_HPP
#define STD_INTEGRATION_HPP

#include <cmath>
#include <universal/number/posit/posit.hpp>
#include <universal/number/posit/mathlib.hpp>
#include "internal/posit_concepts.hpp"

namespace std {
    template <typename T> requires ScalarIsPosit<T>
    T sin(T t) {
        return sw::universal::sin(t);
    }
    template <typename T> requires ScalarIsPosit<T>
    T cos(T t) {
        return sw::universal::cos(t);
    }
    template <typename T> requires ScalarIsPosit<T>
    T sqrt(T t) {
        return sw::universal::sqrt(t);
    }
    template <typename T> requires ScalarIsPosit<T>
    T abs(T t) {
        return sw::universal::abs(t);
    }
    template <typename T> requires ScalarIsPosit<T>
    T floor(T t) {
        return sw::universal::floor(t);
    }
    template <typename T> requires ScalarIsPosit<T>
    T ceil(T t) {
        return sw::universal::ceil(t);
    }
}

#endif