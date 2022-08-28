#include <cmath>
#include "std_integration.hpp"
#include <universal/number/posit/posit.hpp>
#include <iostream>

int main() {
    sw::universal::posit<32, 2> a = 3.14159;

    int code = 0;

    if (std::abs(std::sin(a)) > 0.001) {
        std::cerr << "residual of sin call is: " << std::abs(std::sin(a)) << std::endl;
        code = -1;
    }
    if (std::abs(std::cos(a) + 1.0) > 0.001) {
        std::cerr << "result of cos call is: " << std::cos(a) << std::endl;
        code = -1;
    }
    if (std::abs(a) != std::abs(-a)) {
        std::cerr << "absolute values: " << std::abs(a) << "; " << std::abs(-a) << std::endl;
        code = -1;
    }

    if (std::floor(a) != 3.0) {
        std::cerr << "floor value: " << std::floor(a) << std::endl;
        code = -1;
    }
    if (std::ceil(a) != 4.0) {
        std::cerr << "ceil value: " << std::ceil(a) << std::endl;
        code = -1;
    }
    if (std::abs(std::exp(a) - 23.1406) > 0.001) {
        std::cerr << "exp value: " << std::exp(a) << std::endl;
        code = -1;
    }
    if (std::abs(std::pow(a, a) - 36.46215) > 0.001) {
        std::cerr << "ceil value: " << std::ceil(a) << std::endl;
        code = -1;
    }

    return code;
}
