#ifndef SOLVE_H
#define SOLVE_H

#include "BandedMatrix.h"
#include <stddef.h>

/*!
 * \brief Gauss-Seidel relaxation
 */
void gaussSeidel_host( float* x, BandedMatrix const& A, float const* b, size_t iterations=1 );

#endif /*SOLVE_H*/
