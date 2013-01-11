INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )

#Include the FindCUDA script
INCLUDE(FindCUDA)
#Linking with libcutil (expected to be in ${CUDA_SDK_ROOT_DIR}/C/lib)
#LINK_LIBRARIES("-L/${CUDA_SDK_ROOT_DIR}/C/lib -lcutil")

# Make a list of the cpp files called SRC_FILES
SET( SRC_FILES
   "CoordMatrix.cu"
   "Vector.cu"
)

SET( H_FILES
   "CoordMatrix.h"
   "Vector.h"
)

# Adds a library that depends on the source files.
CUDA_ADD_LIBRARY( gpulib SHARED ${SRC_FILES} )

# Links the new library with any dependencies we have.
#TARGET_LINK_LIBRARIES(
#   gpulib
#   someotherlib
#)

ADD_SUBDIRECTORY( "tests/" )
