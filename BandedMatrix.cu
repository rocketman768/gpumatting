#include "BandedMatrix.h"

/*!
 * \brief b = A*x when A is a sparse banded matrix.
 *
 * Needs a.nbands * sizeof(int) shared memory.
 */
__device__ void bmAx( float* b, const BandedMatrix a, float const* x )
{
   extern __shared__ int sdata[];
   
   int nthreads = blockDim.x*gridDim.x;
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   int j;
   float bi;
   
   // Make the shared data store the band offsets.
   if( threadIdx.x < a.nbands )
      sdata[threadIdx.x] = a.bands[threadIdx.x];
   __syncthreads();
   
   while( true )
   {
      bi = 0.f;
      if( i < a.rows )
      {
         // NOTE: sometimes i+a.bands[j] is a bad index into x[], but we
         // guarantee that when it is a bad index, a[i+j*a->nbands] == 0.f,
         // so as long as there is no segfault, we don't have to branch here.
         for( j = 0; j < a.nbands; ++j )
            bi += a.a[i+j*a.apitch] * x[i+sdata[j]];
         
         b[i] = bi;
      }
      
      if( __any(i >= a.rows) )
         break;

      i += nthreads;
   }
}

/*!
 * \sa bmAx()
 */
__global__ void bmAx_k( float* b, const BandedMatrix a, float const* x )
{
   bmAx(b,a,x);
}
