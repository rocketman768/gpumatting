/*
 * Vector.h is part of gpumatting and Copyright Philip G. Lee <rocketman768@gmail.com> 2013
 * all rights reserved.
 */

#ifndef VECTOR_H
#define VECTOR_H

//! \brief Stores inner product of \c x and \c y of length \c len in \c result.
__global__ void innerProd( float* result, float const* x, float const* y, int len );

__device__ void reduceSequential();

#endif /*VECTOR_H*/
