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

void cmAppend( CoordMatrix* m, int* i, int* j, int* k, size_t length )
{
   cudaMemcpy(m->i, i, length*sizeof(int), cudaMemcpyHostToHost );
   cudaMemcpy(m->j, j, length*sizeof(int), cudaMemcpyHostToHost );
   cudaMemcpy(m->k, k, length*sizeof(float), cudaMemcpyHostToHost );
   m->nnz += length;
}

__global__ void cmAxpy( float* b, CoordMatrix const* a, float const* x, float const* y )
{
   // Find out total number of threads N.
   // Find out thread index 0 <= ti < N.
   
   // Foreach i,j,k in ijk[ nnz/N*ti .. nnz/N*(ti+1)-1 ]
   //    atomicAdd( b+i, k*x[j] )
   
   // if( N > a->rows )
   //    if( ti < N )
   //       b[ti] += y[ti]
}
