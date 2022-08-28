#include <cmath>
#include "std_integration.hpp"
#include <string>
#include <universal/number/posit/posit.hpp>
#include <iostream>

using Scalar = sw::universal::posit<32, 2>;

int main() {
    Scalar a = 3.14159;

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
        std::cerr << "pow value: " << std::pow(a, a) << std::endl;
        code = -1;
    }
    if (std::abs(std::log(a) - 1.14472) > 0.001) {
        std::cerr << "log value: " << std::log(a) << std::endl;
        code = -1;
    }
    if (std::abs(std::atan(a) - 1.262627) > 0.001) {
        std::cerr << "atan value: " << std::atan(a) << std::endl;
        code = -1;
    }
    if (std::to_string(a).compare("3.1415899991989136") != 0) { // 3.14159 cannot be represented directly. this is the closest posit
        std::cerr << "string value: " << std::to_string(a) << std::endl;
        code = -1;
    }

    return code;
}
