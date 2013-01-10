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

//! \brief Initialize a CoordMatrix with space for \c length nonzeros.
void cmInit( CoordMatrix* m, int rows, int cols, size_t length );

//! \brief Release all memory for \c m.
void cmRelease( CoordMatrix* m );

#endif /*COORDMATRIX_H*/
