#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define RANK	2

typedef double		Vector_T[ RANK ];
typedef Vector_T	Matrix_T[ RANK ];

static inline void vRand( Vector_T vctr ) {
	for( register int k = 0; k < RANK; k++ )
		vctr[ k ] = drand48() * 1e2;
}

static inline void mRand( Matrix_T mtx ) {
	for( register int i = 0; i < RANK; i++ )
		vRand( mtx[ i ] );
}

static inline void vDiff( const Vector_T one, const Vector_T two, Vector_T res ) {
	for( register int k = 0; k < RANK; k++ )
		res[ k ] = one[ k ] - two[ k ];
}

static inline void mDiff( const Matrix_T one, const Matrix_T two, Matrix_T res ) {
	for( register int i = 0; i < RANK; i++ )
		vDiff( one[ i ], two[ i ], res[ i ] );
}

static inline double vProd( const Vector_T one, const Vector_T two ) {
	register double	res = .0;

	for( register int k = 0; k < RANK; k++ )
		res += one[ k ] * two[ k ];

	return res;
}

static inline void mTrans( const Matrix_T mtx, Matrix_T res ) {
	for( register int i = 0; i < RANK; i++ )
		for( register int j = 0; j < RANK; j++ )
			res[ j ][ i ] = mtx[ i ][ j ];
}

static inline void mProd( const Matrix_T one, const Matrix_T two, Matrix_T res ) {
	Matrix_T	tmp;

	mTrans( two, tmp );

	for( register int i = 0; i < RANK; i++ )
		for( register int j = 0; j < RANK; j++ )
			res[ i ][ j ] = vProd( one[ i ], tmp[ j ] );
}

static inline double vNorm( const Vector_T vctr ) {
	return sqrt( vProd( vctr, vctr ) );
}

static inline double mNorm( const Matrix_T mtx ) {
	register double	res = .0;

	for( register int i = 0; i < RANK; i++ )
		res += vProd( mtx[ i ], mtx[ i ] );

	return sqrt( res );
}

static inline void vPrint( const Vector_T vctr ) {
	printf( "%g", vctr[ 0 ] );

	for( register int k = 1; k < RANK; k++ )
		printf( ",\t%g", vctr[ k ] );
}

static inline void mPrint( const Matrix_T mtx ) {
	printf( "[ " );
	vPrint( mtx[ 0 ] );
	printf( "\n" );

	for( register int i = 1; i < RANK - 1; i++ ) {
		printf( "  " );
		vPrint( mtx[ i ] );
		printf( "\n" );
	}

	printf( "  " );
	vPrint( mtx[ RANK - 1 ] );
	printf( " ]\n" );
}

int main() {
	Matrix_T	A, B, C;

	srand48( ( long )time( NULL ) );

	mRand( A );
	mRand( B );

	mPrint( A );
	mPrint( B );

	mDiff( A, B, C );
	mProd( C, C, A );

	mPrint( A );

	printf( "Matrix norm is %g\n", mNorm( A ) );
	return 0;
}