/*
 * CoordMatrix.cu is part of gpumatting and Copyright Philip G. Lee <rocketman768@gmail.com> 2013
 * all rights reserved.
 */

#include "CoordMatrix.h"

void cmInit( CoordMatrix* m, int rows, int cols, size_t length )
{
   m->rows = rows;
   m->cols = cols;
   m->nnz = 0;
   m->length = length;

   cudaMalloc( &(m->i), length );
   cudaMalloc( &(m->j), length );
   cudaMalloc( &(m->k), length );
}

void cmRelease( CoordMatrix* m )
{
   cudaFree( m->i );
   cudaFree( m->j );
   cudaFree( m->k );
   m->rows = 0;
   m->cols = 0;
   m->nnz = 0;
   m->length = 0;
}
