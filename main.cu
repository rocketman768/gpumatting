#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include "ppm.h"
#include "BandedMatrix.h"
#include "BandedMatrix.cu"
#include "Matting.cu"

//! \brief Print help message and exit.
void help();
//! \brief Dump vector to stdout in %.5e format.
void dump1D( float* a, int n );
//! \brief Dump row-major matrix to stdout in %.5e format.
void dump2D( float* a, int rows, int cols, size_t pitch );

int myceildiv(int a, int b)
{
   if( a % b != 0 )
      ++a;
   return a/b;
}

int main(int argc, char* argv[])
{
   float4* im;
   float4* dIm;
   unsigned char* scribs;
   float* b;
   float* dB;
   int imW, imH;
   int scribW, scribH;
   size_t imPitch=0;
   int i;
   clock_t beg,end;
   
   if( argc < 3 )
      help();
   
   //==================HOST DATA====================
   im = ppmread_float4( argv[1], &imW, &imH );
   scribs = pgmread( argv[2], &scribW, &scribH );
   if( scribW != imW || scribH != imH )
   {
      fprintf(
         stderr,
         "ERROR: scribbles not the same size as the image.\n"
         "  %d x %d vs. %d x %d\n",
         scribW, scribH, imW, imH
      );
      exit(1);
   }
   
   BandedMatrix L;
   L.rows = imW*imH;
   L.cols = L.rows;
   // Setup bands===
   L.nbands = 17;
   L.bands = (int*)malloc(17*sizeof(int));
   L.bands[8+0] = 0;
   L.bands[8+1] = 1;
   L.bands[8+2] = 2;
   L.bands[8+3] = imW;
   L.bands[8+4] = imW+1;
   L.bands[8+5] = imW+2;
   L.bands[8+6] = 2*imW;
   L.bands[8+7] = 2*imW+1;
   L.bands[8+8] = 2*imW+2;
   L.bands[8-1] = -1;
   L.bands[8-2] = -2;
   L.bands[8-3] = -imW;
   L.bands[8-4] = -(imW+1);
   L.bands[8-5] = -(imW+2);
   L.bands[8-6] = -(2*imW);
   L.bands[8-7] = -(2*imW+1);
   L.bands[8-8] = -(2*imW+2);
   // Setup nonzeros===
   L.a = (float*)malloc( L.nbands*L.rows * sizeof(float));
   memset( L.a, 0x00, L.nbands*L.rows * sizeof(float));
   L.apitch = L.rows;
   
   b = (float*)malloc( L.rows * sizeof(float) );
   
   beg = clock();
   hostLevinLaplacian(L, b, 1e-5, im, scribs, imW, imH, imW);
   end = clock();
   //dump1D( b, L.rows );
   //return 0;
   dump2D( L.a, L.nbands, L.rows, L.rows );
   return 0;
   //printf("Laplacian generation: %.2es\n", (double)(end-beg)/CLOCKS_PER_SEC);
   //------------------------------------------------
   
   dim3 levinLapBlockSize(16,16);
   dim3 levinLapGridSize( myceildiv(imW,16), myceildiv(imH,16) );
   
   //=================GPU Time=======================
   BandedMatrix dL;
   bmCopyToDevice( &dL, &L );
   
   /*
   cudaMallocPitch( (void**)&dIm, &imPitch, imW * sizeof(float4), imH );
   cudaThreadSynchronize();
   imPitch /= sizeof(float4); // Want pitch in terms of elements, not bytes.
   cudaMemcpy2D(
      (void*)dIm,             // Destination
      imPitch*sizeof(float4), // Destination pitch (bytes)
      (const void*)im,        // Source
      imW * sizeof(float4),   // Source pitch (bytes)
      imW * sizeof(float4),   // Source width (bytes)
      imH,                    // Source height
      cudaMemcpyHostToDevice
   );
   cudaThreadSynchronize();
   levinLaplacian<<<levinLapGridSize, levinLapBlockSize>>>(dL, dB, 1e-5, dIm, imW, imH, imPitch);
   
   cudaThreadSynchronize();

   cudaMemcpy2D(
      (void*)L.a,                 // Destination
      L.apitch * sizeof(float),   // Destination pitch (bytes)
      (const void*)dL.a,          // Source
      dL.apitch * sizeof(float),  // Source pitch (bytes)
      dL.rows * sizeof(float),    // Source width (bytes)
      dL.nbands,                  // Source height
      cudaMemcpyDeviceToHost
   );
   
   cudaFree(dIm);
   */
   
   bmDeviceFree( &dL );
   
   cudaThreadSynchronize();
   //------------------------------------------------
   
   // Print any errors
   cudaError_t code = cudaGetLastError(); 
   const char* error_str = cudaGetErrorString(code);
   if( code )
      fprintf(stderr, "ERROR: %s\n", error_str);
   
   // Print some stats
   printf("Pitch: %lu, %lu\n", L.apitch, dL.apitch);
   printf("rows, nbands: %d, %d\n", dL.rows, dL.nbands);
   printf("Image Size: %d x %d\n", imW, imH );
   printf("Grid Dims  : %u x %u\n", levinLapGridSize.x, levinLapGridSize.y );
   printf("Image Pitch: %lu\n", imPitch);
   
   printf("[");
   for( i = 0; i < L.nbands; ++i )
      printf("%.5e, ", L.a[100 + i*L.apitch]);
   printf("]\n");
   
   free(b);
   free(L.a);
   free(L.bands);
   free(im);
   return 0;
}

void help()
{
   fprintf(
      stderr,
      "Usage: matting <image>.ppm <scribbles>.pgm\n"
      "  image     - An RGB image to matte\n"
      "  scribbles - Scribbles for the matte\n"
   );
   
   exit(0);
}

void dump1D( float* a, int n )
{
   int i;
   for( i = 0; i < n-1; ++i )
      printf("%.5e, ", a[i]);
   printf("%.5e\n", a[i]);
}

void dump2D( float* a, int rows, int cols, size_t pitch )
{
   int i,j;
   for( i = 0; i < rows; ++i )
   {
      for( j = 0; j < cols-1; ++j )
         printf("%.5e, ", a[j + i*pitch]);
      printf("%.5e\n", a[j + i*pitch]);
   }
}
