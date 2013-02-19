#include "BandedMatrix.h"

/*!
 * \brief Copy a host matrix to a device matrix.
 */
void copyToDevice( BandedMatrix* dA, BandedMatrix const* hA )
{
   *dA = *hA; 

   cudaMallocPitch( (void**)&(dA->a), (size_t*)&(dA->apitch), hA->rows * sizeof(float), hA->nbands );
   cudaMalloc( (void**)&(dA->bands),  hA->nbands * sizeof(int) );
   cudaMemcpy2D(
      (void*)(dA->a),            // Destination
      dA->apitch,                // Destination pitch
      (const void*)(hA->a),      // Source
      hA->rows * sizeof(float),  // Source pitch
      hA->rows * sizeof(float),  // Source width
      hA->nbands,                // Source height
      cudaMemcpyHostToDevice
   );
   cudaThreadSynchronize();
   dA->apitch /= sizeof(float); // Want the pitch to be in float indices rather than bytes.
   cudaMemcpy( (void*)(dA->bands), (void*)(hA->bands), hA->nbands * sizeof(int), cudaMemcpyHostToDevice );
}

void deviceFree( BandedMatrix* dA )
{
   cudaFree( dA->bands );
   cudaFree( dA->a );
}

/*!
 * \brief b = A*x when A is a sparse banded matrix.
 *
 * Needs a.nbands * sizeof(int) shared memory.
 * NOTE: \c x must have 0 padding so that x[a.bands[i]] and x[N-1+a.bands[i]]
 *       are valid indices into x for all i.
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
