/* VectorTest.cu is part of gpumatting and is 
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
   innerProd_k<<<nBlocks, nThreadsPerBlock, nThreadsPerBlock*sizeof(float)>>>(dresult, dx, dy, N);
   cudaMemcpyAsync( (void*)&hresult, (void const*)dresult, 1*sizeof(float), cudaMemcpyDeviceToHost );
   cudaFree( dresult );
   cudaFree( dy );
   cudaFree( dx );
   
   // Wait for GPU to finish all that shit.
   cudaDeviceSynchronize();
   
   if( fabs(hresult-2e6) < 1e-5 )
      printf("Test PASSED\n");
   else
   {
      printf("Test FAILED\n");
      printf("result: %.2e\n", hresult);
      return 1;
   }
   
   return 0;
}
