#include <stdio.h>
#include <cuda.h>
#include "ppm.h"
#include "BandedMatrix.h"
#include "BandedMatrix.cu"
#include "Matting.cu"

void help();

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
   float* dB;
   int imW, imH, imPitch=0;
   int i;
   
   if( argc < 2 )
      help();
   
   im = ppmread_float4( argv[1], &imW, &imH );
   
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
   
   dim3 levinLapBlockSize(16,16);
   int a = myceildiv(imW,16);
   int b = myceildiv(imH,16);
   dim3 levinLapGridSize( a, b );
   
   //=================GPU Time=======================
   BandedMatrix dL;
   bmCopyToDevice( &dL, &L );
   
   cudaMallocPitch( (void**)&dIm, (size_t*)&imPitch, imW * sizeof(float4), imH );
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
   bmDeviceFree( &dL );
   
   cudaThreadSynchronize();
   //------------------------------------------------
   
   // Print any errors
   cudaError_t code = cudaGetLastError(); 
   const char* error_str = cudaGetErrorString(code);
   if( code )
      fprintf(stderr, "ERROR: %s\n", error_str);
   
   // Print some stats
   printf("Pitch: %d, %d\n", L.apitch, dL.apitch);
   printf("rows, nbands: %d, %d\n", dL.rows, dL.nbands);
   printf("Image Size: %d x %d\n", imW, imH );
   printf("Grid Dims  : %d x %d\n", a, b );
   printf("Image Pitch: %d\n", imPitch);
   
   printf("[");
   for( i = 0; i < L.nbands; ++i )
      printf("%.3f, ", L.a[0 + i*L.apitch]);
   printf("]\n");
   
   free(L.a);
   free(L.bands);
   free(im);
   return 0;
}

void help()
{
   fprintf(
      stderr,
      "Usage: matting <image>.ppm\n"
      "  image - An RGB image to matte\n"
   );
   
   exit(0);
}
