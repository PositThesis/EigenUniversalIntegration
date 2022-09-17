#define RightSideShape__ DenseShape
#include "make_generic_left.hpp"
#undef RightSideShape__

#define RightSideShape__ SparseShape
#include "make_generic_left.hpp"
#undef RightSideShape__
