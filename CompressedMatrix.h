/*
 * CompressedMatrix.h is part of gpumatting and Copyright Philip G. Lee <rocketman768@gmail.com> 2013
 * all rights reserved.
 */

#ifndef COMPRESSEDMATRIX_H
#define COMPRESSEDMATRIX_H

#include <stddef.h>
#include <cuda.h>

/*!
 * \brief Coordinate-based sparse matrix.
 */
typedef struct CompressedMatrix_s{
   int* p;
   int* j;
   float* k;
   int rows;
   int cols;
   //! \brief Number of nonzero entries.
   size_t nnz;
   //! \brief Length of arrays i, j, k
   size_t length;
} CompressedMatrix;

#endif /*COMPRESSEDMATRIX_H*/
