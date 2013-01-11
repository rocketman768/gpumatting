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

#===========Set release/debug settings==========
IF( NOT CUDA_NVCC_FLAGS )
   SET( CUDA_NVCC_FLAGS "-arch sm_20" )
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

ADD_SUBDIRECTORY( "gpulib/" )