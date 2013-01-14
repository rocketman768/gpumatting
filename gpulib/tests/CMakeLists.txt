
CUDA_ADD_EXECUTABLE( csmtest "CompressedMatrixTest.cu" )
TARGET_LINK_LIBRARIES( csmtest gpulib )

CUDA_ADD_EXECUTABLE( vectortest "VectorTest.cu" )
TARGET_LINK_LIBRARIES( vectortest gpulib )
