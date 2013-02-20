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
 * Needs to be called with 2D blocks, enough for 1 thread/pixel
 *
 * NOTE: requires \c levinLaplacian_image to be a globally visible
 *       texture<float4,cudaTextureType2D,cudaReadModeElementType>
 * NOTE: requires \c levinLaplacian_trimap to be a globally visible
 *       texture<float,cudaTextureType2D,cudaReadModeElementType>
 * 
 * \param L output banded sparse Laplacian
 * \param b output right-hand side.
 */
__global__ void levinLaplacian( BandedMatrix* L, float* b, float lambda, float4* im, int imH, int imW, int imPitch )
{
   const float gamma = 1e1;
   const int winRad = 1;
   //const int winSize = (2*winRad+1)*(2*winRad+1);
   
   const int u = blockIdx.x*blockDim.x + threadIdx.x;
   const int v = blockIdx.y*blockDim.y + threadIdx.y;
   
   int u1, v1, u2, v2;
   int i, j;
   int numNeighbors;
   
   float4 rgba, rgba2, white;
   // Local covariance matrix (symmetric).
   float c11, c12, c13,
              c22, c23,
                   c33;
   // Determinant of covariance matrix.
   float cdet;
   
   // Local color average.
   float4 mu;
   
   // Local inverse of covariance matrix (symmetric).
   float d11, d12, d13,
              d22, d23,
                   d33;

   // Construct local covariance matrix in the window.
   // NOTE: this will make some bad indexes into the texture.
   //       Need to find out if that is ok with texture indexing.
   c11 = c12 = c13 = c22 = c23 = c33 = 0.f;
   mu.x = mu.y = mu.z = mu.w = 0.f;
   numNeighbors = 0;
   for( v1 = v-winRad; v1 <= v+winRad; ++v1 )
   {
      for( u1 = u-winRad; u1 <= u+winRad; ++u1 )
      {
         rgba = im[u1 + v1*imPitch];
         mu.x += rgba.x; mu.y += rgba.y; mu.z += rgba.z;
         c11 += rgba.x*rgba.x;
         c12 += rgba.x*rgba.y;
         c13 += rgba.x*rgba.z;
         c22 += rgba.y*rgba.y;
         c23 += rgba.y*rgba.z;
         c33 += rgba.z*rgba.z;
         
         ++numNeighbors;
      }
   }
   mu.x /= numNeighbors; mu.y /= numNeighbors; mu.z /= numNeighbors;
   c11 = (c11+lambda)/numNeighbors - mu.x*mu.x;
   c12 = c12/numNeighbors          - mu.x*mu.y;
   c13 = c13/numNeighbors          - mu.x*mu.z;
   c22 = (c22+lambda)/numNeighbors - mu.y*mu.y;
   c23 = c23/numNeighbors          - mu.y*mu.z;
   c33 = (c33+lambda)/numNeighbors - mu.z*mu.z;
   
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
   
   for( v1 = v-winRad; v1 <= v+winRad; ++v1 )
   {
      for( u1 = u-winRad; u1 <= u+winRad; ++u1 )
      {
         i = u1 + v1*imW;
         rgba = im[u1 + v1*imPitch];
         // Get the whitened pixel
         rgba.x -= mu.x;
         rgba.y -= mu.y;
         rgba.z -= mu.z;
         white.x = d11 * rgba.x + d12 * rgba.y + d13 * rgba.z;
         white.y = d12 * rgba.x + d22 * rgba.y + d23 * rgba.z;
         white.z = d13 * rgba.x + d13 * rgba.y + d33 * rgba.z;
      }
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
