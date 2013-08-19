#include "Solve.h"

void gaussSeidel_host( float* x, BandedMatrix const& A, float const* b, size_t iterations )
{
   int i,jband;
   float tmp;
   float const* a = A.a;
   int const* bands = A.bands;
   int* jbandOff = new int[A.nbands];
   int midBand = A.nbands/2;
   
   // Pre-compute column index offsets.
   for( jband = 0; jband < A.nbands; ++jband )
      jbandOff[jband] = jband*A.apitch;
   
   for( ; iterations > 0; --iterations )
   {
      for( i = 0; i < A.rows; ++i )
      {
         tmp = 0.f;
         
         for( jband = 0; jband < midBand; ++jband )
            tmp += a[i+jbandOff[jband]] * x[i+bands[jband]];
         // Skip the 0-band (bands[jband]==0)
         for( ++jband; jband < A.nbands; ++jband )
            tmp += a[i+jbandOff[jband]] * x[i+bands[jband]];
         
         tmp = b[i] - tmp;
         x[i] = tmp/a[i+jbandOff[midBand]]; // x[i] = tmp/a_ii
      }
   }
   delete[] jbandOff;
}

void jacobi_host( float* x, BandedMatrix const& A, float const* b, size_t iterations, float omega )
{
   int i,jband;
   float tmp;
   float omega1 = 1.f-omega;
   float const* a = A.a;
   int const* bands = A.bands;
   int* jbandOff = new int[A.nbands];
   int midBand = A.nbands/2;
   
   // WARNING: this needs padding so that xx[i+bands[jband]] is always ok.
   float* xx = new float[A.rows];
   //float* xx = new float[A.rows + 4000];
   //xx += 2000;
   float* swapper;
   
   // Pre-compute column index offsets.
   for( jband = 0; jband < A.nbands; ++jband )
      jbandOff[jband] = jband*A.apitch;
   
   for( ; iterations > 0; --iterations )
   {
      for( i = 0; i < A.rows; ++i )
      {
         tmp = 0.f;
         
         for( jband = 0; jband < midBand; ++jband )
            tmp += a[i+jbandOff[jband]] * x[i+bands[jband]];
         // Skip the 0-band (bands[jband]==0)
         for( ++jband; jband < A.nbands; ++jband )
            tmp += a[i+jbandOff[jband]] * x[i+bands[jband]];
         
         tmp = b[i] - tmp;
         xx[i] = omega*tmp/a[i+jbandOff[midBand]] + omega1*x[i];
      }
      
      // Swap xx and x.
      swapper = x;
      x = xx;
      xx = swapper;
   }
   
   // WARNING: because we swap x and xx at each iteration, we may lose the
   // result of the very last iteration on returning.
   
   //xx -= 2000;
   delete[] xx;
   delete[] jbandOff;
}
