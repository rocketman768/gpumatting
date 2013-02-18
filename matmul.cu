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
