/*
 * Matting.cu is part of gpumatting and Copyright Philip G. Lee <rocketman768@gmail.com> 2013
 * all rights reserved.
 */

#include <cuda.h>
#include "BandedMatrix.h"
#include "Vector.cu"

/*!
 * \brief Generate Levin's Laplacian for a given image.
 * 
 * NOTE: requires \c levinLaplacian_image to be a globally visible
 *       texture<float4,cudaTextureType2D,cudaReadModeElementType>
 * NOTE: requires \c levinLaplacian_trimap to be a globally visible
 *       texture<float4,cudaTextureType2D,cudaReadModeElementType>
 * 
 * \param L output banded sparse Laplacian
 * \param b output right-hand side.
 */
__device__ void levinLaplacian( BandedMatrix* L, float* b )
{
   extern texture<float4,cudaTextureType2D,cudaReadModeElementType> levinLaplacian_image;
   extern texture<float,cudaTextureType2D,cudaReadModeElementType> levinLaplacian_trimap;
   
   const float gamma = 1e1;
   const int winRad = 1;
   const int nthreads = blockDim.x*gridDim.x;
   const int i = blockIdx.x*blockDim.x + threadIdx.x;
   
   float u, v;
   float4 rgba;
   // Local covariance matrix (symmetric).
   float c11, c12, c13,
              c22, c23,
                   c33;
   float cdet;
   // Local inverse of covariance matrix (symmetric).
   float d11, d12, d13,
              d22, d23,
                   d33;
   while(true)
   {
      rgba = tex2D(levinLaplacian_image, u, v);
      // Get the inverse.
      cdet = -c11*c12*c12 +
             c11*c11*c22 -
             c13*c13*c22 +
             2*c12*c13*c23 -
             c11*c23*c23;
      d11 = (c11*c22 - c23*c23)/cdet;
      d12 = (c13*c23 - c11*c12)/cdet;
      d13 = (c12*c23 - c13*c22)/cdet;
      d22 = (c11*c11 - c13*c13)/cdet;
      d23 = (c12*c13 - c11*c23)/cdet;
      d33 = (c11*c22 - c12*c12)/cdet;
   }
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