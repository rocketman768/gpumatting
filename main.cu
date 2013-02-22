#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include "ppm.h"
#include "BandedMatrix.h"
#include "BandedMatrix.cu"
#include "Matting.cu"
#include "Vector.cu"

//! \brief Print help message and exit.
void help();
//! \brief Dump vector to stdout in %.5e format.
void dump1D( float* a, int n );
//! \brief Dump row-major matrix to stdout in %.5e format.
void dump2D( float* a, int rows, int cols, size_t pitch );
/*!
 * \brief Solve L*alpha = b by gradient descent.
 * 
 * \param alpha device vector of size L.rows
 * \param L device banded matrix
 * \param b device vector of size L.rows
 * \param iterations number of gradient descent steps before termination
 */
void gradSolve( float* alpha, BandedMatrix L, float* b, int iterations);

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
   float* alpha;
   float* dAlpha;
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
   alpha = (float*)malloc(L.rows * sizeof(float));
   for( i = 0; i < L.rows; ++i )
      alpha[i] = 0.5f;
   
   beg = clock();
   hostLevinLaplacian(L, b, 1e-2, im, scribs, imW, imH, imW);
   end = clock();
   //dump1D( b, L.rows );
   //return 0;
   //dump2D( L.a, L.nbands, L.rows, L.rows );
   fprintf(stderr,"Laplacian generation: %.2es\n", (double)(end-beg)/CLOCKS_PER_SEC);
   //------------------------------------------------
   
   dim3 levinLapBlockSize(16,16);
   dim3 levinLapGridSize( myceildiv(imW,16), myceildiv(imH,16) );
   
   //=================GPU Time=======================
   BandedMatrix dL;
   bmCopyToDevice( &dL, &L );
   
   cudaMalloc((void**)&dB, L.rows*sizeof(float));
   cudaMemcpy((void*)dB, (void*)b, L.rows*sizeof(float), cudaMemcpyHostToDevice);
   
   cudaMalloc((void**)&dAlpha, (L.rows+L.bands[16]*2)*sizeof(float));
   cudaThreadSynchronize();
   dAlpha += L.bands[16];
   cudaMemcpy((void*)dAlpha, (void*)alpha, L.rows*sizeof(float), cudaMemcpyHostToDevice);
   
   //+++++++++++++++++++++++++++++
   //gradSolve(dAlpha, dL, dB, 1);
   //+++++++++++++++++++++++++++++
   
   cudaMemcpy( (void*)alpha, (void*)dAlpha, L.rows*sizeof(float), cudaMemcpyDeviceToHost );
   
   dAlpha -= L.bands[16];
   cudaFree(dAlpha);
   cudaFree(dB);
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
   
   pgmwrite_float("alpha.pgm", imW, imH, alpha, "", 1);
   
   free(alpha);
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

__global__ void addScalar( float* k, float* val )
{
   *k += *val;
}

__global__ void subScalar( float* k, float* val )
{
   *k -= *val;
}

__global__ void multScalar( float* k, float* val )
{
   *k *= *val;
}

__global__ void multScalarConst( float* k, float val )
{
   *k *= val;
}

__global__ void divScalar( float* k, float* val )
{
   *k /= *val;
}

void gradSolve( float* alpha, BandedMatrix L, float* b, int iterations)
{
   float* d;
   float* e;
   float* f;
   float* k;
   int N = L.rows;
   float* tmp;
   
   cudaMalloc((void**)&d, N*sizeof(float));
   cudaMalloc((void**)&e, N*sizeof(float));
   cudaMalloc((void**)&f, N*sizeof(float));
   cudaMalloc((void**)&k, 1*sizeof(float));
   cudaMalloc((void**)&tmp, 1*sizeof(float));
   
   // Do the gradient descent iteration.
   while( iterations-- > 0 )
   {
      // d := 2*L*alpha - b = gradient(alpha'*L*alpha - alpha'*b)
      vecScaleConst_k<<<16,1024>>>( f, alpha, 2.0f, N );
      /*
      bmAxpy_k<17,false><<<16,1024>>>(d, L, f, b);
      
      // If the gradient magnitude is small enough, we're done.
      //innerProd(&tmp, d, d, N);
      
      // k := <d,b>
      innerProd_k<<<16,1024>>>(k, d, b, N);
      
      // e := H*d
      bmAx_k<17><<<16,1024>>>(e, L, d);
      
      // k -= 2*<e,alpha>
      innerProd_k<<<16,1024>>>( tmp, e, alpha, N );
      multScalarConst<<<1,1>>>(tmp, 2.0f);
      subScalar<<<1,1>>>(k,tmp);
      
      // k /= 2*<e,d>
      innerProd_k<<<16,1024>>>( tmp, e, d, N );
      multScalarConst<<<1,1>>>(tmp, 2.0f);
      divScalar<<<1,1>>>(k, tmp);
      
      // alpha += k*d
      vecScale_k<<<16,1024>>>( d, d, k, N );
      vecAdd_k<<<16,1024>>>( alpha, alpha, d, N );
      */
   }
   
   cudaFree(tmp);
   cudaFree(k);
   cudaFree(f);
   cudaFree(e);
   cudaFree(d);
}