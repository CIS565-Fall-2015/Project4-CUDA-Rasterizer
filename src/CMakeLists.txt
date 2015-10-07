set(SOURCE_FILES
    "rasterize.cu"
    "rasterize.h"
    "rasterizeTools.h"
    "EasyBmp.cpp"
    "EasyBmp.h"
    "EasyBMP_BMP.h"
    "EasyBMP_DataStructures.h"
    "EasyBMP_VariousBMPutilities.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
