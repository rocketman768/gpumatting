#include <stdio.h>
#include <cuda.h>
#include "Vector.h"

int main()
{
   int N = 1e6;
   int nBlocks = 16;
   int nThreadsPerBlock = 1024;
   
   int i;
   float* hx = (float*)malloc( N * sizeof(float) );
   float* hy = (float*)malloc( N * sizeof(float) );
   float* dx;
   float* dy;
   float hresult = 0;
   float* dresult;
   
   for( i = 0; i < N; ++i )
   {
      hx[i] = 1.f;
      hy[i] = 2.f;
   }
   
   // Tell GPU what to do.
   cudaMalloc( (void**)&dx, N*sizeof(float) );
   cudaMalloc( (void**)&dy, N*sizeof(float) );
   cudaMalloc( (void**)&dresult, 1*sizeof(float) );
   cudaMemcpyAsync( (void*)dx, (void*)hx, N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpyAsync( (void*)dy, (void*)hy, N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpyAsync( (void*)dresult, (void*)&hresult, 1*sizeof(float), cudaMemcpyHostToDevice );
   innerProd<<<nBlocks, nThreadsPerBlock, nThreadsPerBlock*sizeof(float)>>>(dresult, dx, dy, N);
   cudaMemcpyAsync( (void*)&hresult, (void const*)dresult, 1*sizeof(float), cudaMemcpyDeviceToHost );
   cudaFree( dresult );
   cudaFree( dy );
   cudaFree( dx );
   
   // Wait for GPU to finish all that shit.
   cudaThreadSynchronize();
   
   printf("result: %.2e\n", hresult);
   return 0;
}

