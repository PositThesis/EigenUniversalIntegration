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

    return code;
}
