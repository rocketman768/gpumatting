/* matmul.cu is part of gpumatting and is 
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

// b = a*x
__global__ void matmul(float* b, float* a, float* x, int rows, int cols)
{
   int i = threadIdx.x;
   int j;
   float* a_ij = a+i*cols;
   float* x_j = x;
   float b_i = 0.f;

   for( j = 0; j < cols; ++j )
   {
      b_i += (*a_ij) * (*x_j);
      ++a_ij;
      ++x_j;
   }

   b[i] = b_i;
}

float** rowMatrix(int m, int n)
{
   int i;
   float** A = (float**)malloc(m * sizeof(float*));
   A[0] = (float*)malloc( m*n * sizeof(float) );
   
   for( i = 1; i < m; ++i )
      A[i] = A[i-1] + n;

   return A;
}

void freeRowMatrix(float** A)
{
   free(A[0]);
}

int main()
{
   float** A = rowMatrix(3,3);
   freeRowMatrix(A);
   return 0;
}
