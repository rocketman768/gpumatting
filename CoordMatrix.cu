/* CoordMatrix.cu is part of gpumatting and is 
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

#include "CoordMatrix.h"
#include <device_functions.h>

void cooInit( CoordMatrix* m, int rows, int cols, size_t length )
{
   m->rows = rows;
   m->cols = cols;
   m->nnz = 0;
   m->length = length;

   cudaMalloc( &(m->i), length );
   cudaMalloc( &(m->j), length );
   cudaMalloc( &(m->k), length );
}

void cooRelease( CoordMatrix* m )
{
   cudaFree( m->i );
   cudaFree( m->j );
   cudaFree( m->k );
   m->rows = 0;
   m->cols = 0;
   m->nnz = 0;
   m->length = 0;
}

void cooAppend( CoordMatrix* m, int* i, int* j, int* k, size_t length )
{
   cudaMemcpy(m->i, i, length*sizeof(int), cudaMemcpyHostToHost );
   cudaMemcpy(m->j, j, length*sizeof(int), cudaMemcpyHostToHost );
   cudaMemcpy(m->k, k, length*sizeof(float), cudaMemcpyHostToHost );
   m->nnz += length;
}

__global__ void cooAxpy( float* b, CoordMatrix const* a, float const* x, float const* y )
{
   // Assume blocks and grids are all 1D
   
   // Find out total number of threads N.
   int nthreads = blockDim.x*gridDim.x;
   // Find out thread index 0 <= ti < N.
   int ti = blockIdx.x*blockDim.x + threadIdx.x;
   
   // The end of the a->i array.
   int const* aiend = a->i + a->nnz;
   // The end of y.
   float const* yend = y + a->rows;
   
   int const* ai   = a->i + ti;
   int const* aj   = a->j + ti;
   float const* ak = a->k + ti;
   
   // Matrix multiplication.
   while( ai < aiend )
   {
      // ===COMPLEXITY===
      // Dereferences: 4
      // Int Adds:     4
      // Float Mults:  1
      // Float Adds:   1
      atomicAdd( b+*ai, (*ak)*x[*aj] );
      ai += nthreads;
      aj += nthreads;
      ak += nthreads;
   }
   
   // Add y to b.
   for( y += ti, b += ti; y < yend; y+=nthreads )
      *b += *y;
}
