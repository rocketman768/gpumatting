/* BandedMatrix.h is part of gpumatting and is 
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

#ifndef BANDEDMATRIX_H
#define BANDEDMATRIX_H

typedef struct BandedMatrix_s
{
   // The band data, organized in a nbands x rows dense
   // column-major matrix format
   float* a;
   // The pitch >= rows between the rows of a[].
   size_t apitch;
   
   // [-1,0,2] means a[i+0*apitch] is a_i(i-1),
   // a[i+1*apitch] is a_ii, and a[i+2*apitch] is a_i(i+2)
   int* bands;
   int nbands;
   
   int rows;
   int cols;
}BandedMatrix;

#endif /*BANDEDMATRIX_H*/

