/*
 * CompressedMatrix.cu is part of gpumatting and Copyright Philip G. Lee <rocketman768@gmail.com> 2013
 * all rights reserved.
 */

#include "CompressedMatrix.h"

void csmInit( CompressedMatrix* m, int rows, int cols, size_t length )
{
   m->rows = rows;
   m->cols = cols;
   m->nnz = 0;
   m->length = length;

   cudaMalloc( &(m->k), length );
   cudaMalloc( &(m->j), length );
   cudaMalloc( &(m->p), rows+1 );
   cudaMemset( m->p, 0x00, (rows+1)*sizeof(int));
}

// Thread i is responsible in each iteration for b[i + nthreads*iteration].
// sdata[ti] is used in each block to store partial result for b[i + nthreads*iteration].
__global__ void csmAxpy( float* b, CompressedMatrix const* a, float const* x, float const* y )
{
   extern __shared__ float sdata[];
   
   int nthreads = blockDim.x*gridDim.x;
   int ti = threadIdx.x;
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   
   float* mysdata = sdata+ti;
   
   b += i;
   y += i;
   float const* bend = b + a->rows;
   int const* ap   = a->p + i;
   int const* aj;
   float const* ak;
   float const* akend;
   
   while(b < bend)
   {
      ak = a->k + *ap;
      aj = a->j + *ap;
      akend = a->k + *(ap+1);
      
      *mysdata = 0.0f;
      while( ak < akend )
      {
         *mysdata += *ak * x[*aj];
         ++ak;
         ++aj;
      }
      
      // Wait so that sdata is fully populated.
      // NOTE: will this cause problems since some threads may terminate at the
      // while() condition above?
      __syncthreads();
      // Since all threads sync'd, this should result in sequential access.
      *b = *mysdata + *y;
      
      b += nthreads;
      y += nthreads;
      ap += nthreads;
   }
}
