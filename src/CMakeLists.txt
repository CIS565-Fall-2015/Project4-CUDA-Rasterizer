set(SOURCE_FILES
    "rasterize.cu"
    "rasterize.h"
    "rasterize_min.cu"
    "rasterize_min.h"
    "rasterizeTools.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
