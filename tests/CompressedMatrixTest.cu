/* CompressedMatrixTest.h is part of gpumatting and is 
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

#include <stdio.h>
#include <cuda.h>
#include "Vector.cu"
#include "CompressedMatrix.h"
#include "CompressedMatrix.cu"

int main()
{
   int N = 1e6;
   int nBlocks = 16;
   int nThreadsPerBlock = 1024;
   
   cudaError_t error;
   int i,col;
   int nnz;
   float* hx = (float*)malloc( N * sizeof(float) );
   float* hy = (float*)malloc( N * sizeof(float) );
   float* dx;
   float* dy;
   CompressedMatrix* da;
   float* hb = (float*)malloc( N * sizeof(float) );
   float* db;
   
   // Make x and y vectors.
   for( i = 0; i < N; ++i )
   {
      hx[i] = 1.f;
      hy[i] = 2.f;
   }

   // Make the matrix.
   float* k = (float*)malloc( 9*N*sizeof(float) );
   int* j = (int*)malloc( 9*N*sizeof(int) );
   int* p = (int*)malloc( (N+1)*sizeof(int) );
   float* dk;
   int* dj;
   int* dp;
   for( i = 0, nnz=0; i < N; ++i )
   {
      p[i] = nnz;
      for( col = i; col < N && col-i < 9; ++col, ++nnz )
      {
         k[nnz] = (float)nnz;
         j[nnz] = col;
      }
   }
   p[i] = nnz;
  
   printf("i\tk\tj\tp\n");
   for( i = 0; i < 20; ++i )
      printf("%d %.2e %d %d\n", i, k[i], j[i], p[i]);
   printf( "p[N-1]: %d\np[N]: %d\n", p[N-1], p[N] );

   // Tell GPU what to do.
   cudaMalloc( (void**)&dx, N*sizeof(float) );
   cudaMalloc( (void**)&dy, N*sizeof(float) );
   cudaMalloc( (void**)&db, N*sizeof(float) );
  
   cudaMalloc( (void**)&da, sizeof(CompressedMatrix) );
   cudaMalloc( (void**)&dk, 9*N*sizeof(float) );
   cudaMalloc( (void**)&dj, 9*N*sizeof(int) );
   cudaMalloc( (void**)&dp, (N+1)*sizeof(int) );
   
   cudaMemcpy( (void*)dx, (void*)hx, N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( (void*)dy, (void*)hy, N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( (void*)dk, (void*)k, 9*N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( (void*)dj, (void*)j, 9*N*sizeof(int), cudaMemcpyHostToDevice );
   cudaMemcpy( (void*)dp, (void*)p, (N+1)*sizeof(int), cudaMemcpyHostToDevice );
   
   // Copy the matrix over.
   csmInit<<<1,1>>>( da, N, N, dp, dj, dk, nnz);
   error = cudaDeviceSynchronize();
   printf("error = %d\n", error);

   // Do the damn multiplication already.
   csmAxpy_k<<<nBlocks, nThreadsPerBlock, nThreadsPerBlock*sizeof(float)>>>(db, da, dx, dy);
   
   // Wait for GPU to finish all that shit.
   cudaDeviceSynchronize();

   // Copy result vector back.
   cudaMemcpy( (void*)hb, (void const*)db, N*sizeof(float), cudaMemcpyDeviceToHost );
 
   // Free device pointers.
   cudaFree( dp );
   cudaFree( dj );
   cudaFree( dk );
   cudaFree( da );
   cudaFree( db );
   cudaFree( dy );
   cudaFree( dx );
   
   // Print result
   printf("b=\n");
   for( i = 0; i < 10; ++i )
      printf("%.2f\n", hb[i]);
   printf("...\n");
   
   // Free host pointers.
   free( p );
   free( j );
   free( k );
   free( hb );
   free( hy );
   free( hx );
   
   return 0;
}

