/* BandedMatrix.cu is part of gpumatting and is 
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

#include "BandedMatrix.h"

/*!
 * \brief Copy a host matrix to a device matrix.
 */
void bmCopyToDevice( BandedMatrix* dA, BandedMatrix const* hA )
{
   *dA = *hA; 

   cudaMallocPitch( (void**)&(dA->a), &(dA->apitch), hA->rows * sizeof(float), hA->nbands );
   cudaMalloc( (void**)&(dA->bands),  hA->nbands * sizeof(int) );
   cudaMemcpy2D(
      (void*)(dA->a),            // Destination
      dA->apitch,                // Destination pitch (bytes)
      (const void*)(hA->a),      // Source
      hA->rows * sizeof(float),  // Source pitch (bytes)
      hA->rows * sizeof(float),  // Source width (bytes)
      hA->nbands,                // Source height
      cudaMemcpyHostToDevice
   );
   cudaThreadSynchronize();
   dA->apitch /= sizeof(float); // Want the pitch to be in float indices rather than bytes.
   cudaMemcpy( (void*)(dA->bands), (void*)(hA->bands), hA->nbands * sizeof(int), cudaMemcpyHostToDevice );
}

void bmDeviceFree( BandedMatrix* dA )
{
   cudaFree( dA->bands );
   cudaFree( dA->a );
}

/*!
 * \brief Damped Jacobi iteration.
 * 
 * Does a single damped Jacobi iteration.
 * 
 * \param xx Output of the iteration.
 * \param x Input point of the iteration.
 * \param a Matrix
 * \param b Right-hand side
 * \param omega Damping ratio. 1 means no damping, and 0 means infinite
 *        damping. Usually, 2/3 is used.
 */
template<int nbands>
__global__ void jacobi_k(
   float* xx,
   float const* x,
   const BandedMatrix a,
   float const* b,
   float omega
)
{
   __shared__ int sdata[nbands];
   
   int nthreads = blockDim.x*gridDim.x;
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   int j;
   float bi;
   
   // Make the shared data store the band offsets.
   if( threadIdx.x < nbands )
      sdata[threadIdx.x] = a.bands[threadIdx.x];
   __syncthreads();
   
   while( true )
   {
      bi = b[i];
      if( i < a.rows )
      {
         for( j = 0; j < nbands; ++j )
            bi -= a.a[i+j*a.apitch] * x[i+sdata[j]];
         
         bi = x[i] + bi/a.a[i+(nbands/2)*a.apitch];
         xx[i] = omega*bi + (1.f-omega)*x[i];
      }
      else
         break;
      
      i += nthreads;
   }
}

/*!
 * \brief b = A*x when A is a sparse banded matrix.
 *
 * Needs a.nbands * sizeof(int) shared memory.
 * NOTE: \c x must have 0 padding so that x[a.bands[i]] and x[N-1+a.bands[i]]
 *       are valid indices into x for all i.
 */
template<int nbands>
__device__ void bmAx( float* b, const BandedMatrix a, float const* x )
{
   //extern __shared__ int sdata[];
   __shared__ int sdata[nbands];
   
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
      else
         break;
      
      i += nthreads;
   }
}

/*!
 * \brief b = A*x+y or A*x-y when A is a sparse banded matrix.
 *
 * Needs a.nbands * sizeof(int) shared memory.
 * NOTE: \c x must have 0 padding so that x[a.bands[i]] and x[N-1+a.bands[i]]
 *       are valid indices into x for all i.
 * 
 * \tparam add If true, do A*x+y, else A*x-y
 */
template <int nbands, bool add>
__device__ void bmAxpy( float* b, const BandedMatrix a, float const* x, float const* y )
{
   //extern __shared__ int sdata[];
   __shared__ int sdata[nbands];
   
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
         
         if( add )
            b[i] = bi + y[i];
         else
            b[i] = bi - y[i];
      }
      else
         break;

      i += nthreads;
   }
}

/*!
 * \sa bmAx()
 */
template <int nbands>
__global__ void bmAx_k( float* b, const BandedMatrix a, float const* x )
{
   bmAx<nbands>(b,a,x);
}

/*!
 * \sa bmAxpy()
 */
template <int nbands, bool add>
__global__ void bmAxpy_k( float* b, const BandedMatrix a, float const* x, float const* y )
{
   bmAxpy<nbands, add>(b,a,x,y);
}
