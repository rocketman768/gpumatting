#include <stdio.h>
#include "ppm.h"
#include "Matting.cu"

void help()
{
}

int main(int argc, char* argv[])
{
   unsigned char* im;
   int imW, imH;
   
   if( argc < 2 )
      help();
   
   im = ppmread( argv[1], &imW, &imH );
   
   free(im);
   return 0;
}

