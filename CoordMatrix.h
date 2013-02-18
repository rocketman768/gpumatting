/*
 * CoordMatrix.h is part of gpumatting and Copyright Philip G. Lee <rocketman768@gmail.com> 2013
 * all rights reserved.
 */

#ifndef COORDMATRIX_H
#define COORDMATRIX_H

#include <stddef.h>

/*!
 * \brief Coordinate-based sparse matrix.
 */
typedef struct{
   int* i;
   int* j;
   float* k;
   int rows;
   int cols;
   //! \brief Number of nonzero entries.
   size_t nnz;
   //! \brief Length of arrays i, j, k
   size_t length;
} CoordMatrix;

#endif /*COORDMATRIX_H*/
