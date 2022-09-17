#define LeftSideShape__ DenseShape
#include "make_generic_product_type.hpp"
#undef LeftSideShape__

#define LeftSideShape__ SparseShape
#include "make_generic_product_type.hpp"
#undef LeftSideShape__
