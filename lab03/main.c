#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define ATTEMPTS	8			// количество замеров длительности вычислений (экспериментов)
#define RANK		3000		// порядок матрицы
#define THREADS		4			// число потоков параллельной обработки
#define QRAN		(RANK*RANK) // количество элементов в квадратной матрице порядка RANK

// vectors ====================================================================

typedef double	* vector;

static inline void vRand( vector vctr ) {
	for( register int k = 0; k < RANK; k++ )
		vctr[ k ] = drand48() * 1e2;
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

int main( void ) {
	double 	* A, * B, * C;
	double	period, result = .0, average = .0;
	double	start, finish;
	double 	allRes[ THREADS ] = { .0 };

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

#pragma omp parallel for
		for( register int i = 0; i < QRAN; i += RANK ) {
			for( register int j = 0; j < RANK; j++ )	// mProd( A, A, B );
				for( register int k = 0; k < RANK; k++ )
					B[ i + j ] += A[ i + k ] * A[ k * RANK + j ];

			for( register int j = 0; j < RANK; j++ )	// mProd( B, A, C );
				for( register int k = 0; k < RANK; k++ )
					C[ i + j ] += B[ i + k ] * A[ k * RANK + j ];

			for( register int k = 0; k < RANK; k++ )	// mNorm( C ); - параллельная часть
				allRes[ omp_get_thread_num() ] += C[ i + k ] * C[ i + k ];
		}

		for( register int k = 0; k < THREADS; k++ )	// mNorm( C ); - последовательная часть
			result += allRes[ k ];
		result = sqrt( result );

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