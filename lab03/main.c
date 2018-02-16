#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define ATTEMPTS	10			// количество замеров длительности вычислений (экспериментов)
#define RANK		500		// порядок матрицы
#define THREADS		4			// число потоков параллельной обработки
#define QRAN		(RANK*RANK) // количество элементов в квадратной матрице порядка RANK

// vectors ====================================================================

typedef double	* vector;

static inline void vRand( vector vctr ) {
	for( register int k = 0; k < RANK; k++ )
		vctr[ k ] = drand48() * 1e2;
}

static inline double vProd( const vector one, const vector two ) {
	register double	res = .0;

	for( register int k = 0; k < RANK; k++ )
		res += one[ k ] * two[ k ];

	return res;
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
	matrix	tmp;

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

int main( void ) {
	double 	* A, * B, * C;
	double	period, result, average = .0;
	double	start, finish;

	srand48( ( long )time( NULL ) );

	A = mMake(), B = mMake(), C = mMake();

	// установление числа потокв
	omp_set_num_threads( THREADS );

	for( int i = 1; i <= ATTEMPTS; i++ ) {
		printf( "Начат эксперимент %d\n", i );
		// создание случайной матрицы
		mRand( A );

		// отметка времени начала вычислений
		start = omp_get_wtime();

		mProd( A, A, B );
		mProd( B, A, C );
		result = mNorm( C ); // вычисление нормы матрицы

		// отметка времени окончания вычислений
		finish = omp_get_wtime();

		// определение длительности вычислений в секундах
		period = finish - start;

		// устойчивое вычисление среднего времени выполнения
		average += ( period - average ) / i;

		printf( "Норма: %f.\n", result );
		printf( "Длительность вычислений эксперимента %d: %f с.\n", i, period );
	}

	printf( "Среднее время вычислений за %d экспериментов в %d потоков: %f с.\n", ATTEMPTS, THREADS, average );
	free( A ), free( B ), free( C );

	return 0;
}