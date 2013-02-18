/*
 * Matting.cu is part of gpumatting and Copyright Philip G. Lee <rocketman768@gmail.com> 2013
 * all rights reserved.
 */

#include <cuda.h>
#include "CompressedMatrix.h"
#include "CompressedMatrix.cu"
#include "Vector.cu"

__device__ void levinLaplacian( CompressedMatrix* L, float* b, texture<float,3,cudaReadModeNormalizedFloat> const* image, texture<float,1,cudaReadModeNormalizedFloat> const* trimap )
{
   const float gamma = 1e1;
   const int winRad = 1;
}

/*
__global__ void gradMatting( float* alpha, CompressedMatrix* L, texture<float,3,cudaReadModeNormalizedFloat> const* image, texture<float,1,cudaReadModeNormalizedFloat> const* trimap )
{
   extern __shared__ float sdata[];
   
   const int nthreads = blockDim.x*gridDim.x;
   const int ti = threadIdx.x;
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   
   float* b;
   float* d;
   float* e;
   float* f;
   float k;
   int N = L->rows;
   float tmp; // NOTE: somehow, this must be global or shared. It causes an error with atomicAdd otherwise.
   
   // Create the Laplacian.
   levinLaplacian( L, b, image, trimap );
   __syncthreads();
   
   // Do the gradient descent iteration.
   while( true )
   {
      // d := 2*L*alpha - b = gradient(alpha'*L*alpha - alpha'*b)
      vecScale( f, alpha, 2.0f, N );
      __syncthreads();
      csmAxpy<true,false>(d, L, f, b);
      __syncthreads();
      
      // If the gradient magnitude is small enough, we're done.
      innerProd(&tmp, d, d, N);

      __syncthreads();
      if( tmp < 1e-5 )
         break;
      
      // k := <d,b>
      innerProd(&k, d, b, N);
      __syncthreads();
      
      // e := H*d
      csmAx<true>(e, L, d);
      __syncthreads();
      
      // k -= 2*<e,alpha>
      innerProd( &tmp, e, alpha, N );
      __syncthreads();
      k -= 2*tmp;
      
      // k /= 2*<e,d>
      innerProd( &tmp, e, d, N );
      __syncthreads();
      k /= 2.0f*tmp;
      
      // alpha += k*d
      vecScale( d, d, k, N );
      vecAdd( alpha, alpha, d, N );
   }
}
*/