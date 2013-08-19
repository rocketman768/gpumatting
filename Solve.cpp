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
