#include <iostream>
#include "MtxIO.hpp"
#include <string>
#include <Eigen/Dense>
#include <cmath>
#include <regex>
#include <complex>

// fills the given matrix with predefined values
void fill_matrix(Eigen::MatrixXd& mat) {
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            if ((i + j) % 2 == 0) {
                mat(i, j) = std::sqrt(i * i * j);
            }
            if ((i + j) % 2 == 0) {
                mat(i, j) = - std::sqrt(i * j * j);
            }
        }
    }
}
void fill_cmatrix(Eigen::MatrixXcd& mat) {
    for (int i = 0; i < mat.rows(); i++) {
        for (int j = 0; j < mat.cols(); j++) {
            if ((i + j) % 2 == 0) {
                mat(i, j) = std::complex<double>(std::sqrt(i * i * j), double(i * i) / double(j+1));
            }
            if ((i + j) % 2 == 0) {
                mat(i, j) = -std::complex<double>(std::sqrt(i * j * j), double(j * j) / double(i+1));
            }
        }
    }
}

int main() {
    std::regex normal_expr("(\\d+)\\s+(\\d+)\\s+([\\-\\+\\d\\.e]+)");
    std::regex complex_expr(
      "(\\d+)\\s+(\\d+)\\s+([\\-\\+\\d\\.e]+)\\s+([\\-\\+\\d\\.e]+)");
    std::smatch match;

    std::string name("test.mtx");
    std::string line;
    // test write: pos/neg integers and floats, small and big values
    {
    Eigen::MatrixXd mat_w = Eigen::MatrixXd::Zero(3, 3);
    mat_w(0, 0) = 0;
    mat_w(0, 1) = 1.1;
    mat_w(0, 2) = -0.12;
    mat_w(1, 0) = 12;
    mat_w(1, 1) = 1.5e6;
    mat_w(1, 2) = 2.0;
    mat_w(2, 0) = 0.0;
    mat_w(2, 1) = 4e-15;
    mat_w(2, 2) = -2;
    write_matrix(mat_w, name);

    std::ifstream file(name);
    getline(file, line);
    assert(line.compare("# auto-generated") == 0);
    getline(file, line);
    assert(line.compare("3 3 9") == 0);
    getline(file, line);
    std::regex_match(line, match, normal_expr);
    assert(std::stoi(match[1]) == 1 && std::stoi(match[2]) == 1 && std::stod(match[3]) == 0);
    getline(file, line);
    std::regex_match(line, match, normal_expr);
    assert(std::stoi(match[1]) == 1 && std::stoi(match[2]) == 2 && std::stod(match[3]) == 1.1);
    getline(file, line);
    std::regex_match(line, match, normal_expr);
    assert(std::stoi(match[1]) == 1 && std::stoi(match[2]) == 3 && std::stod(match[3]) == -0.12);
    getline(file, line);
    std::regex_match(line, match, normal_expr);
    assert(std::stoi(match[1]) == 2 && std::stoi(match[2]) == 1 && std::stod(match[3]) == 12);
    getline(file, line);
    std::regex_match(line, match, normal_expr);
    assert(std::stoi(match[1]) == 2 && std::stoi(match[2]) == 2 && std::stod(match[3]) == 15e5);
    getline(file, line);
    std::regex_match(line, match, normal_expr);
    assert(std::stoi(match[1]) == 2 && std::stoi(match[2]) == 3 && std::stod(match[3]) == 2);
    getline(file, line);
    std::regex_match(line, match, normal_expr);
    assert(std::stoi(match[1]) == 3 && std::stoi(match[2]) == 1 && std::stod(match[3]) == 0.0);
    getline(file, line);
    std::regex_match(line, match, normal_expr);
    assert(std::stoi(match[1]) == 3 && std::stoi(match[2]) == 2 && std::stod(match[3]) == 4e-15);
    getline(file, line);
    std::regex_match(line, match, normal_expr);
    assert(std::stoi(match[1]) == 3 && std::stoi(match[2]) == 3 && std::stod(match[3]) == -2);
    if (getline(file, line)) {
        std::cerr << "More lines than expected" << std::endl;
        return -1;
    }
    file.close();
    }
    // test complex write
    Eigen::MatrixXcd mat_cw = Eigen::MatrixXcd::Zero(3, 2);
    mat_cw(0, 0) = std::complex<double>(0, 12);
    mat_cw(0, 1) = std::complex<double>(1.1, -1);
    mat_cw(1, 0) = std::complex<double>(12, -4e-2);
    mat_cw(1, 1) = std::complex<double>(1.5e6, 0);
    mat_cw(2, 0) = std::complex<double>(0.0, 7);
    mat_cw(2, 1) = std::complex<double>(4e-15, 0);
    write_matrix(mat_cw, name);

    std::ifstream file(name);
    getline(file, line);
    assert(line.compare("# auto-generated") == 0);
    getline(file, line);
    assert(line.compare("3 2 6") == 0);
    getline(file, line);
    std::regex_match(line, match, complex_expr);
    assert(std::stoi(match[1]) == 1 && std::stoi(match[2]) == 1 && std::stod(match[3]) == 0 && std::stod(match[4]) == 12);
    getline(file, line);
    std::regex_match(line, match, complex_expr);
    assert(std::stoi(match[1]) == 1 && std::stoi(match[2]) == 2 && std::stod(match[3]) == 1.1 && std::stod(match[4]) == -1);
    getline(file, line);
    std::regex_match(line, match, complex_expr);
    assert(std::stoi(match[1]) == 2 && std::stoi(match[2]) == 1 && std::stod(match[3]) == 12 && std::stod(match[4]) == -4e-2);
    getline(file, line);
    std::regex_match(line, match, complex_expr);
    assert(std::stoi(match[1]) == 2 && std::stoi(match[2]) == 2 && std::stod(match[3]) == 15e5 && std::stod(match[4]) == 0);
    getline(file, line);
    std::regex_match(line, match, complex_expr);
    assert(std::stoi(match[1]) == 3 && std::stoi(match[2]) == 1 && std::stod(match[3]) == 0 && std::stod(match[4]) == 7);
    getline(file, line);
    std::regex_match(line, match, complex_expr);
    assert(std::stoi(match[1]) == 3 && std::stoi(match[2]) == 2 && std::stod(match[3]) == 4e-15 && std::stod(match[4]) == 0);
    if (getline(file, line)) {
        std::cerr << "More lines than expected" << std::endl;
        return -1;
    }
    file.close();


    // test read + write
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(5, 5);
    fill_matrix(mat);
    write_matrix(mat, name);
    Eigen::MatrixXd read_mat = get_matrix_from_mtx_file<double>(name);

    assert(read_mat.rows() == mat.rows() && read_mat.cols() == mat.cols());

    Eigen::MatrixXd err = mat - read_mat;


    for (int i = 0; i < err.rows(); i++) {
        for (int j = 0; j < err.cols(); j++) {
            if (mat(i, j) == 0) {
                auto absolute_error = err(i, j);
                assert(std::abs(absolute_error) < 1e-10);
            } else {
                auto relative_error = err(i, j) / mat(i, j);
                assert(std::abs(relative_error) < 1e-5);
            }
        }
    }

    // test complex read + write
    Eigen::MatrixXcd mat2 = Eigen::MatrixXcd::Zero(5, 5);
    fill_cmatrix(mat2);
    write_matrix(mat2, name);
    Eigen::MatrixXcd read_mat2 = get_matrix_from_mtx_file<std::complex<double>>(name);

    assert(read_mat.rows() == mat.rows() && read_mat.cols() == mat.cols());

    Eigen::MatrixXcd err2 = mat2 - read_mat2;


    for (int i = 0; i < err2.rows(); i++) {
        for (int j = 0; j < err2.cols(); j++) {
            if (mat2(i, j) == std::complex<double>(0, 0)) {
                auto absolute_error = err2(i, j);
                assert(std::abs(absolute_error) < 1e-10);
            } else {
                auto relative_error = err2(i, j) / mat2(i, j);
                assert(std::abs(relative_error) < 1e-5);
            }
        }
    }


    return 0;
}