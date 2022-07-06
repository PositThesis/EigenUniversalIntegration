#ifndef SHERMAN_MATRIX_H
#define SHERMAN_MATRIX_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <unsupported/Eigen/SparseExtra>

#include "internal/posit_concepts.hpp" // for complex check

using Eigen::Dynamic;
using Eigen::Matrix;

#if 0
template <typename FP, size_t part = 3312>
Matrix<FP, Dynamic, Dynamic> get_sherman_matrix(std::string &path) {

  Matrix<FP, Dynamic, Dynamic> m(part, part);

  std::regex expr("(\\d+)\\s+(\\d+)\\s+([\\-\\+\\d\\.e]+)");
  std::smatch match;

  std::string line;
  std::ifstream matrix_file;
  matrix_file.open(path);
  if (matrix_file.is_open()) {
    // skip the first two lines

    getline(matrix_file, line);
    getline(matrix_file, line);
    while (getline(matrix_file, line)) {
      std::regex_match(line, match, expr);
      if (match.size() > 0) {
        int x = std::stoi(match[1]) - 1;
        int y = std::stoi(match[2]) - 1;

        if (x < part && y < part) {
          double temp_val = std::stod(match[3]);
          FP val = FP(temp_val);
          m(x, y) = val;
        }
      }
    }
  } else {
    std::cout << "opening failed..." << std::endl;
    exit(-1);
  }

  return m;
}
#endif

template <typename FP>
Eigen::SparseMatrix<FP> get_sherman_sparse(std::string &path) {
  Eigen::SparseMatrix<FP> m;
  Eigen::loadMarket(m, path);
  return m;
}

template <typename FP, int max_size = -1>
Matrix<FP, Dynamic, Dynamic> get_matrix_from_mtx_file(std::string &path) {
  constexpr bool complex = is_complex<FP>;
  std::regex normal_expr("(\\d+)\\s+(\\d+)\\s+([\\-\\+\\d\\.e]+)");
  std::regex complex_expr(
      "(\\d+)\\s+(\\d+)\\s+([\\-\\+\\d\\.e]+)\\s+([\\-\\+\\d\\.e]+)");
  std::smatch match;

  std::string line;
  std::ifstream matrix_file;
  matrix_file.open(path);
  if (matrix_file.is_open()) {
    // skip the first line

    getline(matrix_file, line);

    // determine matrix size
    getline(matrix_file, line);
    std::regex_match(line, match, normal_expr);
    int height = std::stoi(match[1]);
    if constexpr (max_size > 0) {
      height = std::min(height, max_size);
    }
    int width = std::stoi(match[2]);
    if constexpr (max_size > 0) {
      width = std::min(width, max_size);
    }
    Matrix<FP, Dynamic, Dynamic> m =
        Matrix<FP, Dynamic, Dynamic>::Zero(height, width);

    while (getline(matrix_file, line)) {
      if (!complex) {
        std::regex_match(line, match, normal_expr);
      } else {
        std::regex_match(line, match, complex_expr);
      }
      if (match.size() > 0) {
        int x = std::stoi(match[1]) - 1;
        int y = std::stoi(match[2]) - 1;

        if (x < height && y < width) {
          if constexpr (!complex) {
            double temp_val = std::stod(match[3]);
            FP val = FP(temp_val);
            m(x, y) = val;
          } else {
            using RealFP = typename FP::value_type;
            double temp_val_r = std::stod(match[3]);
            double temp_val_i = std::stod(match[4]);
            RealFP val_r = RealFP(temp_val_r);
            RealFP val_i = RealFP(temp_val_i);
            m(x, y) = std::complex(val_r, val_i);
          }
        }
      }
    }

    return m;
  } else {
    std::cout << "opening failed..." << std::endl;
    exit(-1);
  }
}

template <typename MatrixType>
void write_matrix(MatrixType mat, std::string path) {
  constexpr bool complex = is_complex<typename MatrixType::Scalar>;
  std::ofstream matrix_file;
  matrix_file.open(path);

  if (matrix_file.is_open()) {
    matrix_file << "# auto-generated\n"
                << mat.rows() << " " << mat.cols() << " "
                << mat.rows() * mat.cols() << std::endl;

    for (int row = 0; row < mat.rows(); row++) {
      for (int col = 0; col < mat.cols(); col++) {
        matrix_file << row + 1 << " " << col + 1 << " ";
        if constexpr (!complex) {
          matrix_file << mat(row, col) << std::endl;
        } else {
          matrix_file << mat(row, col).real() << " " << mat(row, col).imag()
                      << std::endl;
        }
      }
    }
  } else {
    std::cerr << "failed to write result to disk, here it is:\n"
              << mat << std::endl;
  }
}

#endif