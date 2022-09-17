
#define DefinedProductType__ GemmProduct
#include "generic.hpp"
#undef DefinedProductType__

#define DefinedProductType__ GemvProduct
#include "generic.hpp"
#undef DefinedProductType__

#define DefinedProductType__ LazyProduct
#include "generic.hpp"
#undef DefinedProductType__

#define DefinedProductType__ InnerProduct
#include "generic.hpp"
#undef DefinedProductType__