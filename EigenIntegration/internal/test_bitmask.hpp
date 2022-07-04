#ifndef TEST_BITMASK_HPP
#define TEST_BITMASK_HPP

#include <thread>

enum class EigenOverrideMask {
  LAZYPRODUCT = 0x1ul << 0,
  GEMM_ROWMAJOR = 0x1ul << 1,
  GEMM_COLMAJOR = 0x1ul << 2,
  GEMV_ROWMAJOR = 0x1ul << 3,
  GEMV_COLMAJOR = 0x1ul << 4,
  DOT = 0x1ul << 5,
  DOT_T = 0x1ul << 6,
  CMPLX_LAZYPRODUCT = 0x1ul << 7,
  CMPLX_GEMM_ROWMAJOR = 0x1ul << 8,
  CMPLX_GEMM_COLMAJOR = 0x1ul << 9,
  CMPLX_GEMV_ROWMAJOR = 0x1ul << 10,
  CMPLX_GEMV_COLMAJOR = 0x1ul << 11,
  CMPLX_DOT = 0x1ul << 12,
  CMPLX_DOT_T = 0x1ul << 13,
  DISPATCHER = 0x1ul << 14,
  GENERIC_GEMM = 0x1ul << 15,
  GENERIC_GEMV = 0x1ul << 16,
  GENERIC_LAZY = 0x1ul << 17,
  GENERIC_INNER = 0x1ul << 18,
};

// A global variable (that can also be shared across threads) which tells us
// which overrides have been used
std::atomic<uint64_t> eigen_usage_vector;

#endif