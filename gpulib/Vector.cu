/*
 * Vector.cu is part of gpumatting and Copyright Philip G. Lee <rocketman768@gmail.com> 2013
 * all rights reserved.
 */

#include "Vector.h"

__global__ void innerProd( float* result, float const* x, float const* y, int len )
{
   extern __shared__ float sdata[];
   
   int nthreads = blockDim.x*gridDim.x;
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int ti = threadIdx.x;
   
   float* mysdata = sdata+ti;
   *mysdata = 0.f;
   
   for( ; i < len; i += nthreads )
      *mysdata += x[i]*y[i];
  
   // Doesn't work for some reason? 
   //if( i == 0 )
   //   *result = 0.f;
   
   // Wait for all the shared data to be fully populated.
   __syncthreads();
   
   reduceSequential();
   
   // Need each block to contribute its final result to the global result.
   if( ti == 0 )
      atomicAdd( result, sdata[0] );
}

__device__ void reduceSequential()
{
   extern __shared__ float sdata[];
   
   int ti = threadIdx.x;
   int stride;
   
   for( stride = blockDim.x>>1; stride > 0; stride >>= 1 )
   {
      if( ti < stride )
         sdata[ti] += sdata[ti+stride];
      __syncthreads();
   }
}
