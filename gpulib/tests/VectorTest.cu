#include <stdio.h>
#include <cuda.h>
#include "Vector.h"

int main()
{
   int N = 1e6;
   int nBlocks = 8;
   int nThreadsPerBlock = 32;
   
   int i;
   float* hx = (float*)malloc( N * sizeof(float) );
   float* hy = (float*)malloc( N * sizeof(float) );
   float* dx;
   float* dy;
   float hresult;
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
   cudaMemcpy( (void*)dx, (void*)hx, N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( (void*)dy, (void*)hy, N*sizeof(float), cudaMemcpyHostToDevice );
   innerProd<<<nBlocks, nThreadsPerBlock, nThreadsPerBlock*sizeof(float)>>>(dresult, dx, dy, N);
   cudaMemcpy( (void*)&hresult, (void const*)dresult, 1*sizeof(float), cudaMemcpyDeviceToHost );
   cudaFree( dresult );
   cudaFree( dy );
   cudaFree( dx );
   
   // Wait for GPU to finish all that shit.
   cudaThreadSynchronize();
   
   printf("%.2e\n", hresult);
   return 0;
}
