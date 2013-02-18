#include <stdio.h>
#include <cuda.h>
#include "BandedMatrix.h"
#include "BandedMatrix.cu"

int main()
{
   int N = 1e6;
   int nBlocks = 16;
   int nThreadsPerBlock = 1024;
   
   int i;
   
   float* hx = (float*)malloc( N * sizeof(float) );
   float* hb = (float*)malloc( N * sizeof(float) );
   BandedMatrix hA;
   
   float* dx;
   float* db;
   BandedMatrix dA;
   
   // Initialize the host's banded matrix.
   hA.nbands = 3;
   hA.rows = N;
   hA.cols = N;
   hA.a = (float*)malloc( N * hA.nbands * sizeof(float) );
   memset( hA.a, 0x00, N*hA.nbands * sizeof(float) );
   hA.bands = (int*)malloc( hA.nbands * sizeof(int) );
   hA.bands[0] = -10;
   hA.bands[1] = 0;
   hA.bands[2] = 10;
   
   for( i = 0; i < N; ++i )
   {
      hx[i] = i;
      /*
      if( i - 10 >= 0 )
         hA.a[0 + i*hA.nbands] = 1;
      hA.a[1 + i*hA.nbands] = 1;
      if( i + 10 < N )
         hA.a[2 + i*hA.nbands] = 1;
      */
      
      if( i - 10 >= 0 )
         hA.a[i + 0*hA.rows] = 1;
      hA.a[i + 1*hA.rows] = 1;
      if( i + 10 < N )
         hA.a[i + 2*hA.rows] = 1;
   }
   
   dA = hA;
   
   // Tell GPU what to do.
   cudaMalloc( (void**)&dx, N*sizeof(float) );
   cudaMalloc( (void**)&db, N*sizeof(float) );
   cudaMalloc( (void**)&(dA.a),  N * hA.nbands * sizeof(float) );
   cudaMalloc( (void**)&(dA.bands),  hA.nbands * sizeof(int) );
   
   cudaMemcpy( (void*)dx, (void*)hx, N*sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( (void*)(dA.a), (void*)(hA.a), N * hA.nbands * sizeof(float), cudaMemcpyHostToDevice );
   cudaMemcpy( (void*)(dA.bands), (void*)(hA.bands), hA.nbands * sizeof(int), cudaMemcpyHostToDevice );
   
   bmAx_k<<<nBlocks, nThreadsPerBlock>>>(db, dA, dx);
   cudaMemcpy( (void*)hb, (void*)db, N*sizeof(float), cudaMemcpyDeviceToHost );
   
   cudaFree( dA.a );
   cudaFree( dA.bands );
   cudaFree( db );
   cudaFree( dx );
   
   // Wait for GPU to finish all that shit.
   cudaThreadSynchronize();
   
   //+++++++++++++++++++++++++++++++TEST+++++++++++++++++++++++++++++++++++++++
   bool passed = true;
   for( i = 0; i < 10; ++i )
   {
      if(fabs(hb[i] - (10.f+2*i)) > 1e-6)
      {
         passed = false;
         break;
      }
   }
   for( i = 10; passed && i < N-10; ++i )
   {
      if(fabs(hb[i] - (30.0+3*(i-10))) > 1e-6)
      {
         printf("b[%d]=%.5e\n", i, hb[i]);
         passed = false;
         break;
      }
   }
   for( i = N-10; passed && i < N; ++i )
   {
      if(fabs(hb[i] - (1999970.0+2*(i-(N-10)))) > 1e-6)
      {
         printf("b[%d]=%.5e\n", i, hb[i]);
         passed = false;
         break;
      }
   }
   
   if(passed)
      printf("Test PASSED\n");
   else
      printf("Test FAILED\n");
   //--------------------------------------------------------------------------
   
   free(hA.bands);
   free(hA.a);
   free(hb);
   free(hx);
   return 0;
}
