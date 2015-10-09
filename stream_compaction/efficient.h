#pragma once
#include "../src/rasterizeTools.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

namespace StreamCompaction {
namespace Efficient {
	int Compact(int n, Triangle *odata, Triangle *idata);
}
}
