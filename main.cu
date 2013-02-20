#include <stdio.h>
#include <cuda.h>
#include "ppm.h"
#include "Matting.cu"

void help();

int main(int argc, char* argv[])
{
   float4* im;
   int imW, imH;
   
   if( argc < 2 )
      help();
   
   im = ppmread_float4( argv[1], &imW, &imH );
   
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
