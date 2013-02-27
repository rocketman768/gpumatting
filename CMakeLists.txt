PROJECT( gpumatting )
CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )

#==============Set Directories=================
SET( SOURCE_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
SET( doc_DIR "doc/" )
SET( test_DIR "tests/" )

SET( EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin" )
SET( LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/lib" )

#=================Options=======================
OPTION( DEBUG_MODE
        "If on, will build everything with debug flags."
                ON )
OPTION( DO_TESTING
        "If on, will make unit tests."
                ON )
OPTION( DO_PROFILING
        "If on, and using g++, will perform profiling."
                OFF )

#==============Custom Cmake Modules=============
SET( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules" )

#===================Find Shit================
FIND_PACKAGE(CUDA REQUIRED)

SET(CUDA_SEPARABLE_COMPILATION OFF)
IF( NOT CUDA_NVCC_FLAGS )
   SET( CUDA_NVCC_FLAGS "-arch sm_30" )
ENDIF()

#Include the FindCUDA script
INCLUDE(FindCUDA)
CUDA_BUILD_CLEAN_TARGET()

#===========Set release/debug settings==========

# OpenMP
IF( CMAKE_COMPILER_IS_GNUCXX )
   SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp" )
   SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp" )
ELSEIF( MSVC )
   SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp" )
   SET( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp" )
ENDIF()

IF( CMAKE_COMPILER_IS_GNUCXX )
   SET( CMAKE_CXX_FLAGS_DEBUG "-g3 -Wall -Wno-sign-compare" )
   SET( LINK_FLAGS_DEBUG "-rdynamic" )
   SET( CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wno-deprecated -Wno-sign-compare" )
   SET( LINK_FLAGS_RELEASE "-rdynamic" )
   IF( DO_PROFILING )
      SET( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg" )
      SET( CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pg" )
    ENDIF()
ENDIF()

IF( DEBUG_MODE )
   SET( CMAKE_BUILD_TYPE Debug )
ELSE()
   SET( CMAKE_BUILD_TYPE Release )
   # This definition disables the assert() macro.
   ADD_DEFINITIONS( -DNDEBUG )
ENDIF()

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR} )
CUDA_ADD_EXECUTABLE( matting main.cu SLIC.cpp )
IF( DO_TESTING )
   ADD_SUBDIRECTORY( "tests/" )
ENDIF()
