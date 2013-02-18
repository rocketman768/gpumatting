#include <stdio.h>
#include <cuda.h>
#include "BandedMatrix.h"
#include "BandedMatrix.cu"

int main()
{
   int N = 1e6;
   int nBlocks = 16;
   int nThreadsPerBlock = 1024;
   
   int i;
   
   float* hx = (float*)malloc( N * sizeof(float) );
   float* hb = (float*)malloc( N * sizeof(float) );
   BandedMatrix hA;
   
   float* dx;
   float* db;
   BandedMatrix dA;
   
   // Initialize the host's banded matrix.
   hA.nbands = 3;
   hA.rows = N;
   hA.cols = N;
   hA.a = (float*)malloc( N * hA.nbands * sizeof(float) );
   memset( hA.a, 0x00, N*hA.nbands * sizeof(float) );
   hA.bands = (int*)malloc( hA.nbands * sizeof(int) );
   hA.bands[0] = -10;
   hA.bands[1] = 0;
   hA.bands[2] = 10;
   
   for( i = 0; i < N; ++i )
   {
      hx[i] = 1.f;
      if( i - 10 >= 0 )
         hA.a[0 + i*hA.nbands] = 1;
      hA.a[1 + i*hA.nbands] = 1;
      if( i + 10 < N )
         hA.a[2 + i*hA.nbands] = 1;
   }
   
   dA = hA;
   
   // Tell GPU what to do.
   cudaMalloc( (void**)&dx, N*sizeof(float) );
   cudaMalloc( (void**)&db, N*sizeof(float) );
   cudaMemcpyAsync( (void*)dx, (void*)hx, N*sizeof(float), cudaMemcpyHostToDevice );
   
   cudaMemcpyAsync( (void*)dA.a, (void*)hA.a, N * hA.nbands * sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpyAsync( (void*)dA.bands, (void*)hA.bands, hA.nbands * sizeof(int), cudaMemcpyHostToDevice );
   
   bmAx_k<<<nBlocks, nThreadsPerBlock>>>(db, dA, dx);
   cudaMemcpyAsync( (void*)&hb, (void const*)db, N*sizeof(float), cudaMemcpyDeviceToHost );
   
   cudaFree( dA.a );
   cudaFree( dA.bands );
   cudaFree( db );
   cudaFree( dx );
   
   // Wait for GPU to finish all that shit.
   cudaThreadSynchronize();
   
   for( i = 0; i < 20; ++i )
      printf("%.2e\n", hb[i]);
   
   free(hx);
   free(hb);
   return 0;
}
