#ifndef BANDEDMATRIX_H
#define BANDEDMATRIX_H

typedef struct BandedMatrix_s
{
   // The band data, organized in a nbands x rows dense
   // column-major matrix format
   float* a;
   // The pitch >= rows between the rows of a[].
   int apitch;
   
   // [-1,0,2] means a[i+0*apitch] is a_i(i-1),
   // a[i+1*apitch] is a_ii, and a[i+2*apitch] is a_i(i+2)
   int* bands;
   int nbands;
   
   int rows;
   int cols;
}BandedMatrix;

#endif /*BANDEDMATRIX_H*/

