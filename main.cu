/* main.cu is part of gpumatting and is 
 * Copyright 2013 Philip G. Lee <rocketman768@gmail.com>
 * 
 * gpumatting is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * gpumatting is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with gpumatting.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include "ppm.h"
#include "BandedMatrix.h"
#include "BandedMatrix.cu"
#include "Matting.cu"
#include "Vector.cu"
#include "SLIC.h"

//! \brief Print help message and exit.
void help();
//! \brief Dump vector to stdout in %.5e format.
void dump1D( float* a, int n );
//! \brief Dump row-major matrix to stdout in %.5e format.
void dump2D( float* a, int rows, int cols, size_t pitch );
/*!
 * \brief Solve L*alpha = b by gradient descent.
 * 
 * \param alpha device vector of size L.rows padded properly to make \c L * \c alpha work.
 * \param L device banded matrix
 * \param b device vector of size L.rows
 * \param iterations number of gradient descent steps before termination
 * \param pad The size of left and right vector padding to make \c L * x work for a vector x.
 */
void gradSolve( float* alpha, BandedMatrix L, float* b, int iterations, int pad);
/*!
 * \brief Solve L*alpha = b by conjugate-gradient descent.
 * 
 * \param alpha device vector of size L.rows padded properly to make \c L * \c alpha work.
 * \param L device banded matrix
 * \param b device vector of size L.rows
 * \param pad The size of left and right vector padding to make \c L * x work for a vector x.
 * \param iterations number of steps before termination
 * \param restartInterval restart cg after this many iterations (typically about 50)
 */
void cgSolve( float* alpha, BandedMatrix L, float* b, int pad, int iterations, int restartInterval);
/*!
 * \brief Compute and display matte ground truth errors.
 *
 * \param alpha Computed alpha matte
 * \param gtAlpha Ground truth alpha matte
 * \param imW Matte width
 * \param imH Matte height
 */
void computeError( float* alpha, float* gtAlpha, int imW, int imH );

int myceildiv(int a, int b)
{
   if( a % b != 0 )
      ++a;
   return a/b;
}

int main(int argc, char* argv[])
{
   enum Solver{SOLVER_GRAD, SOLVER_CG};
   Solver solver = SOLVER_CG;
   float4* im;
   unsigned char* charIm;
   unsigned char* scribs;
   int* labels;
   //unsigned int numLabels;
   float* b;
   float* dB;
   float* alpha;
   float* dAlpha;
   int dAlpha_pad;
   float* alphaGt = 0;
   int imW, imH;
   int scribW, scribH;
   int gtW, gtH;
   int i;
   int iterations;
   clock_t beg,end;
   
   if( argc < 5 )
      help();
   
   //==================HOST DATA====================
      
   // Parse the options.
   if( strncmp(argv[1],"grad",4)==0 )
      solver = SOLVER_GRAD;
   else
      solver = SOLVER_CG;
   iterations = atoi(argv[2]);
   im = ppmread_float4( &charIm, argv[3], &imW, &imH );
   scribs = pgmread( argv[4], &scribW, &scribH );
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
   if( argc > 5 )
      alphaGt = pgmread_float( argv[5], &gtW, &gtH );
   
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
   
   labels = (int*)malloc(imW*imH*sizeof(int));
   beg = clock();
   // charIm is [r,g,b,a,r,g,b,a...], but slic wants [a,r,g,b,a,r,g,b,...], so shift the
   // charIm by 1 to appease it.
   // WARNING: this may cause a segfault since it can result in a bad dereference.
   //numLabels = slicSegmentation( labels, (unsigned int*)(charIm-1), imW, imH, 100, 10.0 );
   //end = clock();
   //fprintf(stderr,"SLIC segmentation: %.2es\n", (double)(end-beg)/CLOCKS_PER_SEC);
   
   beg = clock();
   // WARNING: regularization param < 1e-3 seems to make the Laplacian unstable.
   hostLevinLaplacian(L, b, 1e-3, im, scribs, imW, imH, imW);
   end = clock();
   fprintf(stderr,"Laplacian generation: %.2es\n", (double)(end-beg)/CLOCKS_PER_SEC);
   //------------------------------------------------
   
   // Pad alpha by a multiple of 32 that is larger than (2*imW+2).
   dAlpha_pad = ((2*imW+2)/32)*32+32;
   
   //=================GPU Time=======================
   cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
   BandedMatrix dL;
   bmCopyToDevice( &dL, &L );
   
   cudaMalloc((void**)&dB, L.rows*sizeof(float));
   cudaMemcpy((void*)dB, (void*)b, L.rows*sizeof(float), cudaMemcpyHostToDevice);
   
   vecCopyToDevice(&dAlpha, alpha, L.rows, dAlpha_pad, dAlpha_pad);
   
   //+++++++++++++++++++++++++++++
   switch( solver )
   {
      case SOLVER_GRAD:
         gradSolve(dAlpha, dL, dB, iterations, dAlpha_pad);
         break;
      case SOLVER_CG:
         cgSolve(dAlpha, dL, dB, dAlpha_pad, iterations, 101);
         break;
      default:
         break;
   }
   //+++++++++++++++++++++++++++++
   
   cudaMemcpy( (void*)alpha, (void*)dAlpha, L.rows*sizeof(float), cudaMemcpyDeviceToHost );
   
   vecDeviceFree( dAlpha, dAlpha_pad );
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
   
   if(alphaGt)
      computeError(alpha, alphaGt, imW, imH);
   
   pgmwrite_float("alpha.pgm", imW, imH, alpha, "", 1);
   
   free(alpha);
   free(b);
   free(L.a);
   free(L.bands);
   free(labels);
   free(scribs);
   free(im);
   free(charIm);
   return 0;
}

void help()
{
   fprintf(
      stderr,
      "Usage: matting <solver> <iter> <image>.ppm <scribbles>.pgm [<gt>.pgm]\n"
      "  solver    - Either \"grad\" or \"cg\" for gradient/conjugate-gradient\n"
      "  iter      - Number of iterations for the solver\n"
      "  image     - An RGB image to matte\n"
      "  scribbles - Scribbles for the matte\n"
      "  gt        - Ground truth for the matte\n"
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

__global__ void divScalar2( float* lhs, float* num, float* den )
{
   *lhs = *num / *den;
}

void gradSolve( float* alpha, BandedMatrix L, float* b, int iterations, int pad)
{
   float* d;
   float* e;
   float* f;
   float* k;
   int N = L.rows;
   float* tmp;
   
   vecDeviceMalloc(&d, N, pad, pad);
   cudaMalloc((void**)&e, N*sizeof(float));
   vecDeviceMalloc(&f, N, pad, pad);
   cudaMalloc((void**)&k, 1*sizeof(float));
   cudaMalloc((void**)&tmp, 1*sizeof(float));
   
   cudaThreadSynchronize();
   
   // Do the gradient descent iteration.
   while( iterations-- > 0 )
   {
      // d := L*alpha - b
      bmAxpy_k<17,false><<<16,1024>>>(d, L, alpha, b);
      
      // If the gradient magnitude is small enough, we're done.
      //innerProd(&tmp, d, d, N);
      
      // k := <d,b>
      innerProd_k<<<16,1024,1024*sizeof(float)>>>(k, d, b, N);
      
      // e := H*d
      bmAx_k<17><<<16,1024>>>(e, L, d);
      
      // k -= <e,alpha>
      innerProd_k<<<16,1024,1024*sizeof(float)>>>( tmp, e, alpha, N );
      subScalar<<<1,1>>>(k,tmp);
      
      // k /= <e,d>
      innerProd_k<<<16,1024,1024*sizeof(float)>>>( tmp, e, d, N );
      divScalar<<<1,1>>>(k, tmp);
      
      // alpha += k*d
      vecScale_k<<<16,1024>>>( d, d, k, N );
      vecAdd_k<<<16,1024>>>( alpha, alpha, d, N );
   }
   
   cudaFree(tmp);
   cudaFree(k);
   vecDeviceFree(f, pad);
   cudaFree(e);
   vecDeviceFree(d, pad);
}

void cgSolve( float* alpha, BandedMatrix L, float* b, int pad, int iterations, int restartInterval)
{
   float* r;
   float* p;
   float* Lp;
   float* kp;
   float* k;
   int N = L.rows;
   float* rTr;
   
   // This makes the first iteration gradient descent.
   int innerIter = 0;
   
   vecDeviceMalloc(&r, N, pad, pad);
   vecDeviceMalloc(&p, N, pad, pad);
   vecDeviceMalloc(&Lp, N, pad, pad);
   vecDeviceMalloc(&kp, N, 0, 0);
   cudaMalloc((void**)&k, 1*sizeof(float));
   cudaMalloc((void**)&rTr, 1*sizeof(float));
   
   cudaThreadSynchronize();
   
   // Do the conjugate gradient iterations.
   while( iterations-- > 0 )
   {
      if( innerIter == 0 )
      {
         // r := L*alpha - b
         bmAxpy_k<17,false><<<16,1024>>>(r, L, alpha, b);
         // p = -r
         vecScaleConst_k<<<16,1024>>>(p, r, -1.0f, N);
         
         innerIter = restartInterval-1;
      }
      else
         --innerIter;
      
      // Lp := L*p
      bmAx_k<17><<<16,1024>>>(Lp, L, p);
      
      // k = <r,r>/<p,p>_L
      innerProd_k<<<16,1024,1024*sizeof(float)>>>(rTr, r, r, N);
      innerProd_k<<<16,1024,1024*sizeof(float)>>>(k, p, Lp, N);
      divScalar2<<<1,1>>>(k,rTr,k);
      
      // alpha += k*p
      vecScale_k<<<16,1024>>>(kp, p, k, N);
      vecAdd_k<<<16,1024>>>(alpha, alpha, kp, N);
      
      // r += k*L*p
      vecScale_k<<<16,1024>>>(Lp, Lp, k, N);
      vecAdd_k<<<16,1024>>>(r, r, Lp, N);
      
      // k = <r,r>/<r_old,r_old>
      innerProd_k<<<16,1024,1024*sizeof(float)>>>(k, r, r, N);
      divScalar<<<1,1>>>(k,rTr);
      
      // p = k*p - r;
      vecScale_k<<<16,1024>>>(kp, p, k, N);
      vecSub_k<<<16,1024>>>( p, kp, r, N );
   }
   
   cudaFree(rTr);
   cudaFree(k);
   vecDeviceFree(kp, 0);
   vecDeviceFree(Lp, pad);
   vecDeviceFree(p, pad);
   vecDeviceFree(r, pad);
}

void computeError( float* alpha, float* gtAlpha, int imW, int imH )
{
   double ssd = 0.0;
   int i, j;
   
   for( i = 0; i < imH; ++i )
   {
      for( j = 0; j < imW; ++j )
      {
         if( alpha[j + i*imW] > 1.0f )
            ssd += (1.0f-gtAlpha[j+i*imW])*(1.0f-gtAlpha[j+i*imW]);
         else if( alpha[j + i*imW] < 0.0f )
            ssd += gtAlpha[j+i*imW] * gtAlpha[j+i*imW];
         else
            ssd += (alpha[j+i*imW]-gtAlpha[j+i*imW])*(alpha[j+i*imW]-gtAlpha[j+i*imW]);
      }
   }
   
   ssd /= imW*imH;
   
   printf("Ground truth SSD: %.3e\n", ssd);
}
