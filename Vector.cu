/*
 * Vector.cu is part of gpumatting and Copyright Philip G. Lee <rocketman768@gmail.com> 2013
 * all rights reserved.
 */

#ifndef VECTOR_CU
#define VECTOR_CU

#include <device_functions.h>

/*!
 * \brief Copy a host vector to a device vector.
 * \param leftPadding amount of 0-filled padding on the left of the vector.
 * \param rightPadding amount of 0-filled padding ont he right of the vector.
 */
void vecCopyToDevice(float** dx, float const* hx, int length, int leftPadding=0, int rightPadding=0 )
{
   cudaMalloc((void**)dx, sizeof(float)*(length+leftPadding+rightPadding));
   cudaThreadSynchronize();
   cudaMemset((void*)*dx, 0x00, sizeof(float)*(length+leftPadding+rightPadding));
   if( leftPadding )
      *dx += leftPadding;
   cudaMemcpy((void*)*dx, (void*)hx, sizeof(float)*length, cudaMemcpyHostToDevice);
}

void vecDeviceFree( float* dx, int leftPadding=0 )
{
   cudaFree(dx-leftPadding);
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

template <int blocksize>
__device__ void reduceUnrolled()
{
   extern __shared__ float sdata[];
   int ti = threadIdx.x;
   
   if( blocksize >= 1024 )
   {
      if( ti < 512 )
         sdata[ti] += sdata[ti+512];
      __syncthreads();
   }
   if( blocksize >= 512 )
   {
      if( ti < 256 )
         sdata[ti] += sdata[ti+256];
      __syncthreads();
   }
   if( blocksize >= 256 )
   {
      if( ti < 128 )
         sdata[ti] += sdata[ti+128];
      __syncthreads();
   }
   if( blocksize >= 128 )
   {
      if( ti < 64 )
         sdata[ti] += sdata[ti+64];
      __syncthreads();
   }
   
   // Since warp size is 32, these are guaranteed to happen synchronously,
   // so no explicity synching is needed.
   if( ti < 32 )
   {
      if( blocksize >= 64 )
         sdata[ti] += sdata[ti+32];
      if( blocksize >= 32 )
         sdata[ti] += sdata[ti+16];
      if( blocksize >= 16 )
         sdata[ti] += sdata[ti+8];
      if( blocksize >= 8 )
         sdata[ti] += sdata[ti+4];
      if( blocksize >= 4 )
         sdata[ti] += sdata[ti+2];
      if( blocksize >= 2 )
         sdata[ti] += sdata[ti+1];
   }
}

/*!
 * \brief Add vectors. Can be in-place.
 * 
 * Shared memory: 0
 * 
 * \param result output vector \c x + \c y. May be \c x or \c y.
 * \param x input vector
 * \param y second input vector
 * \param len number of elements in \c x, \c y, and \c result.
 */
__device__ void vecAdd( float* result, float const* x, float const* y, int len )
{
   int nthreads = blockDim.x*gridDim.x;
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int ti = threadIdx.x;
   
   while( i < len )
   {
      result[i] = x[i]+y[i];
      
      i += nthreads;
   }
}

__global__ void vecAdd_k( float* result, float const* x, float const* y, int len )
{
   vecAdd( result, x, y, len );
}

/*!
 * \brief Scale vector by a constant. Can be in-place.
 * 
 * Shared memory: 0
 * 
 * \param result output vector. May be \c x.
 * \param x input vector
 * \param s scaling factor
 * \param len number of elements in \c x and \c result.
 */
__device__ void vecScale( float* result, float const* x, float s, int len )
{
   int nthreads = blockDim.x*gridDim.x;
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   int ti = threadIdx.x;
   
   while( i < len )
   {
      result[i] = s*x[i];
      i += nthreads;
   }
}

__global__ void vecScaleConst_k( float* result, float const* x, float s, int len )
{
   vecScale( result, x, s, len );
}

__global__ void vecScale_k( float* result, float const* x, float* s, int len )
{
   vecScale( result, x, *s, len );
}

/*!
 * \brief Stores inner product of \c x and \c y of length \c len in \c result.
 * 
 * Shared memory: blockDim.x*sizeof(float)
 * 
 * \param result scalar inner product
 * \param x first vector
 * \param y second vector
 * \param len length of \c x and \c y
 */
__device__ void innerProd( float* result, float const* x, float const* y, int len )
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
      *result = 0.f;
   
   // Wait for all the shared data to be fully populated.
   __syncthreads();
   
   /*
   switch( blockDim.x )
   {
      case 1024:
         reduceUnrolled<1024>();
         break;
      case 512:
         reduceUnrolled<512>();
         break;
      case 256:
         reduceUnrolled<256>();
         break;
      case 128:
         reduceUnrolled<128>();
         break;
      case 64:
         reduceUnrolled<64>();
         break;
      case 32:
         reduceUnrolled<32>();
         break;
      case 16:
         reduceUnrolled<16>();
         break;
      case 8:
         reduceUnrolled<8>();
         break;
      case 4:
         reduceUnrolled<4>();
         break;
      case 2:
         reduceUnrolled<2>();
         break;
      case 1:
         reduceUnrolled<1>();
         break;
   }
   */
   
   reduceSequential();
   
   // Need each block to contribute its final result to the global result.
   if( ti == 0 )
      atomicAdd( result, sdata[0] );
}

__global__ void innerProd_k( float* result, float const* x, float const* y, int len )
{
   innerProd( result, x, y, len );
}

#endif
