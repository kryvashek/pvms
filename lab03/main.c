#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define ATTEMPTS	10			// количество замеров длительности вычислений (экспериментов)
#define RANK		500		// порядок матрицы
#define STEP		100			// шаг для оптимизации 6
#define THREADS		4			// число потоков параллельной обработки
#define QRAN		(RANK*RANK) // количество элементов в квадратной матрице порядка RANK

// vectors ====================================================================

typedef double	* vector;

double * vMake() {
	return calloc( RANK, sizeof( double ) );
}

static inline void vRand( vector vctr ) {
	for( register int k = 0; k < RANK; k++ )
		vctr[ k ] = drand48() * 1e2;
}

static inline int min( const register int a, const register int b ) {
	return a < b ? a : b;
}

static inline void vDiff( const vector one, const vector two, vector res ) {
	for( register int k = 0; k < RANK; k += STEP )
		for( register int K = k; K < min( k + STEP, RANK ); K++ )
			res[ K ] = one[ K ] - two[ K ]; // осуществлена оптимизация 6 ("разделение на блоки")
}

static inline double vProd( const vector one, const vector two ) {
	register double	res = .0;

	for( register int k = 0; k < RANK; k++ )
		res += one[ k ] * two[ k ];

	return res;
}

static inline void vPrint( const vector vctr ) {
	printf( "%g", vctr[ 0 ] );

	for( register int k = 1; k < RANK; k++ )
		printf( ",\t%g", vctr[ k ] );
}

// matrices ===================================================================

typedef double	* matrix;

double * mMake() {
	return calloc( QRAN, sizeof( double ) );
}

static inline void mRand( matrix mtx ) {
	for( register int i = 0; i < QRAN; i += RANK )
		vRand( mtx + i );
}

static inline void mDiff( const matrix one, const matrix two, matrix res ) {
	for( register int i = 0; i < QRAN; i += RANK )
		vDiff( one + i, two + i, res + i );
}

static inline void mTrans( const matrix mtx, matrix res ) {
#pragma omp parallel for
	for( register int i = 0; i < RANK; i += 4 ) {
		for( register int j = 0; j < RANK; j++ ) {
			res[ j * RANK + i ] = mtx[ i * RANK + j ]; // осуществлена оптимизация 5 ("развёртка цикла")
			res[ j * RANK + i + 1 ] = mtx[ ( i + 1 ) * RANK + j ];
			res[ j * RANK + i + 2 ] = mtx[ ( i + 2 ) * RANK + j ];
			res[ j * RANK + i + 3 ] = mtx[ ( i + 3 ) * RANK + j ];
		}
	}
}

static inline void mProd( const matrix one, const matrix two, matrix res ) {
	matrix			tmp;

	tmp = mMake();

	mTrans( two, tmp );

#pragma omp parallel for
	for( register int i = 0; i < QRAN; i += RANK ) {
		for( register int j = 0; j < RANK; j++ )
			res[ i + j ] = vProd( one + i, tmp + j * RANK ); // осуществлена оптимизация 2 ("расширение скаляра")
	}

	free( tmp );
}

static inline double mNorm( const matrix mtx ) {
	register double	sumRes = .0;
	double allRes[ THREADS ] = { 0 };

#pragma omp parallel for
	for( register int i = 0; i < QRAN; i += RANK ) {
		allRes[ omp_get_thread_num() ] += vProd( mtx + i, mtx + i );
	}

	for( register int i = 0; i < THREADS; i++ )
		sumRes += allRes[ i ];

	return sqrt( sumRes );
}

static inline void mPrint( const matrix mtx ) {
	printf( "[ " );
	vPrint( mtx );
	printf( "\n" );

	for( register int i = RANK; i < QRAN - RANK; i += RANK ) {
		printf( "  " );
		vPrint( mtx + i );
		printf( "\n" );
	}

	printf( "  " );
	vPrint( mtx + QRAN - RANK );
	printf( " ]\n" );
}

// допустима оптимизация 7 ("увеличение локализации вычислений")

int main() {
	double 			* A, * B, * C, period, result, average = .0;
//	struct timespec	start,
//					finish;
	double	start,
			finish;

	srand48( ( long )time( NULL ) );

	A = mMake(), B = mMake(), C = mMake();

	omp_set_num_threads( THREADS );

	for( int i = 1; i <= ATTEMPTS; i++ ) {
		printf( "Начат эксперимент %d\n", i );
		// создание случайных матриц
		mRand( A );

		// вывод матриц в случае малого порядка
		if( RANK < 4 ) {
			printf( "Матрица A:\n" );
			mPrint( A );
			printf( "Матрица B:\n" );
			mPrint( B );
		}

		// отметка времени начала вычислений
		// clock_gettime( CLOCK_MONOTONIC, &start );
		start = omp_get_wtime();

		mProd( A, A, B );
		mProd( B, A, C );
		result = mNorm( C ); // вычисление нормы матрицы

		// отметка времени окончания вычислений
		// clock_gettime( CLOCK_MONOTONIC, &finish );
		finish = omp_get_wtime();

		// вывод результирующей матрицы в случае малого порядка
		if( RANK < 4 ) {
			printf( "Матрица C:\n" );
			mPrint( A );
		}

		// определение длительности вычислений в микросекундах
//		period = ( double )( finish.tv_sec - start.tv_sec ) * 1e6 + ( double )( finish.tv_nsec - start.tv_nsec ) / 1e3;
		period = ( finish - start ) * 1e6;
		average += ( period - average ) / i;
		printf( "Норма: %f.\n", result );
		printf( "Длительность вычислений эксперимента %d: %f мкс.\n", i, period );
	}

	printf( "Среднее время вычислений за %d экспериментов в %d потоков: %f мкс\n", ATTEMPTS, THREADS, average );
	free( A ), free( B ), free( C );

	return 0;
}