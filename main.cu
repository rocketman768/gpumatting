#include <stdio.h>
#include <cuda.h>
#include "ppm.h"
#include "Matting.cu"

texture<float4,cudaTextureType2D,cudaReadModeElementType> levinLaplacian_image;
texture<float,cudaTextureType2D,cudaReadModeElementType> levinLaplacian_trimap;

void help();

int main(int argc, char* argv[])
{
   float* im;
   int imW, imH;
   
   if( argc < 2 )
      help();
   
   im = ppmread_float( argv[1], &imW, &imH );
   
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
