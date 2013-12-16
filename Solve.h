#ifndef SOLVE_H
#define SOLVE_H

#include "BandedMatrix.h"
#include <stddef.h>

/*!
 * \brief Gauss-Seidel relaxation.
 * 
 * Sequential Gauss-Seidel on a single host CPU for \f$ Ax=b \f$.
 * 
 * \param x Input and output vector
 * \param A Input matrix
 * \param b Input right-hand side
 * \param iterations How many smoothing iterations to perform
 */
void gaussSeidel_host( float* x, BandedMatrix const& A, float const* b, size_t iterations=1 );

/*!
 * \brief Damped Jacobi relaxation.
 * 
 * Damped Jacobi iterations on a single host CPU for \f$ Ax=b \f$.
 * 
 * \param x Input and output vector
 * \param A Input matrix
 * \param b Input right-hand side
 * \param iterations How many smoothing iterations to perform
 * \param omega Damping ratio (in [0,1]). Default is 2/3 (typical choice).
 *        1 means no damping, 0 means infinite.
 */
void jacobi_host( float* x, BandedMatrix const& A, float const* b, size_t iterations, int xpad, float omega=2.f/3.f );

#endif /*SOLVE_H*/
