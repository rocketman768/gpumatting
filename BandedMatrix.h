#ifndef BANDEDMATRIX_H
#define BANDEDMATRIX_H

typedef struct BandedMatrix_s
{
   // The band data, organized in a rows x nbands dense
   // row-major matrix format
   float* a;
   // [-1,0,2] means a[0+i*nbands] is a_i(i-1),
   // a[1+i*nbands] is a_ii, and a[2+i*nbands] is a_i(i+2)
   int* bands;
   int nbands;
   int rows;
   int cols;
}BandedMatrix;

#endif /*BANDEDMATRIX_H*/

