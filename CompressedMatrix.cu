/* CompressedMatrix.cu is part of gpumatting and is 
 * Copyright 2013 Philip G. Lee <rocketman768@gmail.com>
 * 
 * gpumatting is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * gpumatting is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with gpumatting.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "CompressedMatrix.h"
#include <cuda.h>

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

__global__ void csmInit( CompressedMatrix* m, int rows, int cols, int* p, int* j, float* k, size_t nnz )
{
   m->rows = rows;
   m->cols = cols;
   m->nnz = nnz;
   m->length = nnz;
   
   m->p = p;
   m->j = j;
   m->k = k;
}

// Thread i is responsible in each iteration for b[i + nthreads*iteration].
// sdata[ti] is used in each block to store partial result for b[i + nthreads*iteration].
template<bool symmetric, bool addy>
__device__ void csmAxpy( float* b, CompressedMatrix const* a, float const* x, float const* y )
{
   extern __shared__ float sdata[];
   
   int nthreads = blockDim.x*gridDim.x;
   int ti = threadIdx.x;
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   
   float* mysdata = sdata+ti;
   
   int row = i;
   int numrows = a->rows;
   int ndx, ndxEnd;
   
   while(true)
   {
      if( row <= numrows )
      {
         ndx = a->p[row];
         ndxEnd = a->p[row+1];
         *mysdata = 0.0f;
         while( ndx < ndxEnd )
         {
            *mysdata += a->k[ndx] * x[a->j[ndx]];
            ++ndx;
         }
      }
      
      // Wait so that sdata is fully populated.
      // Since all threads sync'd, this should result in sequential access.
      __syncthreads();
      if( row <= numrows )
         b[row] = *mysdata + *y;
      
      // If any thread has fallen off, they will all fall off next iteration,
      // so end it now!
      if( __any( row > numrows ) )
         break;
      
      row += nthreads;
   }
}

template<bool symmetric>
__device__ void csmAx( float* b, CompressedMatrix const* a, float const* x )
{
   extern __shared__ float sdata[];
   
   int nthreads = blockDim.x*gridDim.x;
   int ti = threadIdx.x;
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   
   float* mysdata = sdata+ti;
   
   b += i;
   float const* bend = b + a->rows;
   int const* ap   = a->p + i;
   int const* aj;
   float const* ak;
   float const* akend;
   
   while(true)
   {
      if( b < bend )
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
         __syncthreads();
         // Since all threads sync'd, this should result in sequential access.
         *b = *mysdata;
      }
      else
      {
         // These threads have fallen off the end, so just have them sit.
         __syncthreads();
      }
      
      // If any thread has fallen off, they will all fall off next iteration,
      // so end it now!
      if( __any( b >= bend ) )
         break;
      
      b += nthreads;
      ap += nthreads;
   }
}

__global__ void csmAxpy_k( float* b, CompressedMatrix const* a, float const* x, float const* y )
{
   csmAxpy<false, true>(b, a, x, y);
}

__global__ void csmAxpy_k_symm( float* b, CompressedMatrix const* a, float const* x, float const* y )
{
   csmAxpy<true, true>(b, a, x, y);
}

__global__ void csmAxmy_k( float* b, CompressedMatrix const* a, float const* x, float const* y )
{
   csmAxpy<false, false>(b, a, x, y);
}

__global__ void csmAxmy_k_symm( float* b, CompressedMatrix const* a, float const* x, float const* y )
{
   csmAxpy<true, false>(b, a, x, y);
}
