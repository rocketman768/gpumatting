#include <stdio.h>
#include <cuda.h>
#include "ppm.h"
#include "BandedMatrix.h"
#include "BandedMatrix.cu"
#include "Matting.cu"

void help();

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
   
   //=================GPU Time=======================
   BandedMatrix dL;
   bmCopyToDevice( &dL, &L );
   
   cudaMallocPitch( (void**)&dIm, (size_t*)&imPitch, imW * sizeof(float4), imH );
   cudaThreadSynchronize();
   imPitch /= sizeof(float4); // Want pitch in terms of elements, not bytes.
   cudaMemcpy2D(
      (void*)dIm,            // Destination
      imPitch,               // Destination pitch
      (const void*)im,       // Source
      imW * sizeof(float4),  // Source pitch
      imW * sizeof(float4),  // Source width
      imH,                   // Source height
      cudaMemcpyHostToDevice
   );
   
   dim3 blocksize(32,32);
   dim3 gridsize( ceil(imW/32), ceil(imH/32) );
   levinLaplacian<<<blocksize, gridsize>>>(dL, dB, 1e-5, dIm, imW, imH, imPitch);
   
   cudaFree(dIm);
   bmDeviceFree( &dL );
   
   cudaThreadSynchronize();
   //------------------------------------------------
   
   cudaError_t code = cudaGetLastError(); 
   const char* error_str = cudaGetErrorString(code);
   if( code )
      fprintf(stderr, "ERROR: %s\n", error_str);
   
   printf("Pitch: %d\n", imPitch);
   
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
