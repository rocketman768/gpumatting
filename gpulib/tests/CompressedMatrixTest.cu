#include <stdio.h>
#include <cuda.h>
#include "Vector.h"
#include "CompressedMatrix.h"

int main()
{
   int N = 1e6;
   int nBlocks = 16;
   int nThreadsPerBlock = 1024;
   
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
   
   // Tell GPU what to do.
   cudaMalloc( (void**)&dx, N*sizeof(float) );
   cudaMalloc( (void**)&dy, N*sizeof(float) );
   cudaMalloc( (void**)&db, N*sizeof(float) );
  
   cudaMalloc( (void**)&da, sizeof(CompressedMatrix) );
   cudaMalloc( (void**)&dk, 9*N*sizeof(float) );
   cudaMalloc( (void**)&dj, 9*N*sizeof(int) );
   cudaMalloc( (void**)&dp, (N+1)*sizeof(int) );
   
   /* 
   cudaMemcpyAsync( (void*)dx, (void*)hx, N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpyAsync( (void*)dy, (void*)hy, N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpyAsync( (void*)dk, (void*)k, 9*N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpyAsync( (void*)dj, (void*)j, 9*N*sizeof(int), cudaMemcpyHostToDevice );
   cudaMemcpyAsync( (void*)dp, (void*)p, (N+1)*sizeof(int), cudaMemcpyHostToDevice );
   
   // Copy the matrix over.
   csmInit<<<1,1>>>( da, N, N, dp, dj, dk, nnz);
   
   // Do the damn multiplication already.
   //csmAxpy<<<nBlocks, nThreadsPerBlock, nThreadsPerBlock*sizeof(float)>>>(db, da, dx, dy);
   
   // Copy result vector back.
   cudaMemcpyAsync( (void*)hb, (void const*)db, N*sizeof(float), cudaMemcpyDeviceToHost );
   */
 
   // Wait for GPU to finish all that shit.
   cudaThreadSynchronize();

   // Free device pointers.
   cudaFree( da->p );
   cudaFree( da->j );
   cudaFree( da->k );
   cudaFree( da );
   cudaFree( db );
   cudaFree( dy );
   cudaFree( dx );
   
   // Print result
   printf("b=\n");
   for( i = 0; i < 10; ++i )
      printf("%.2e\n", hb[i]);
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

