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

//! \brief Append elements to the matrix. Duplicate (\c i,\c j) will add.
void cmAppend( CoordMatrix* m, int* i, int* j, int* k, size_t length );

//! \brief Compute \c b += \c a \c x + \c y.
__global__ void cmAxpy( float* b, CoordMatrix const* a, float const* x, float const* y );
#endif /*COORDMATRIX_H*/
