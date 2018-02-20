#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define ERROR		1.0
#define MAGNITUDE	1e1
#define SCHEMES		4
#define PARASCHEMES	3
#define THREADVARS	3
#define RANKVARS	3

// vectors ====================================================================

typedef double	* vector;

static inline void vRand( const vector vctr, const int rank ) {
	for( register int k = 0; k < rank; k++ )
		vctr[ k ] = drand48() * MAGNITUDE;
}

static inline double vProd( const vector one, const vector two, const int rank ) {
	register double	res = .0;

	for( register int k = 0; k < rank; k++ )
		res += one[ k ] * two[ k ];

	return res;
}

// matrices ===================================================================

typedef double	* matrix;

double * mMake( const int rank ) {
	return calloc( rank * rank, sizeof( double ) );
}

static inline void mRand( const matrix mtx, const int rank ) {
	const register int	qran = rank * rank;

	for( register int i = 0; i < qran; i += rank )
		vRand( mtx + i, rank );
}

static inline void mTrans( const matrix mtx, const matrix res, const int rank ) {
#pragma omp parallel for
	for( register int i = 0; i < rank; i += 4 ) {
		for( register int j = 0; j < rank; j++ ) {
			res[ j * rank + i ] = mtx[ i * rank + j ]; // осуществлена оптимизация 5 ("развёртка цикла")
			res[ j * rank + i + 1 ] = mtx[ ( i + 1 ) * rank + j ];
			res[ j * rank + i + 2 ] = mtx[ ( i + 2 ) * rank + j ];
			res[ j * rank + i + 3 ] = mtx[ ( i + 3 ) * rank + j ];
		}
	}
}

static inline void mProd( const matrix one, const matrix two, const matrix res, const int rank ) {
	const matrix	tmp = mMake( rank );
	const register int qran = rank * rank;

	mTrans( two, tmp, rank );

#pragma omp parallel for
	for( register int i = 0; i < qran; i += rank ) {
		for( register int j = 0; j < rank; j++ )
			res[ i + j ] = vProd( one + i, tmp + j * rank, rank ); // осуществлена оптимизация 2 ("расширение скаляра")
	}

	free( tmp );
}

static inline double mNorm( const matrix mtx, const int rank ) {
	const register int	qran = rank * rank,
						threads = omp_get_max_threads();
	register double	sumRes = .0;
	double allRes[ threads ];

	memset( allRes, 0, threads * sizeof( double ) );

#pragma omp parallel for
	for( register int i = 0; i < qran; i += rank ) {
		allRes[ omp_get_thread_num() ] += vProd( mtx + i, mtx + i, rank );
	}

	for( register int i = 0; i < threads; i++ )
		sumRes += allRes[ i ];

	return sqrt( sumRes );
}

// calculations ===============================================================

// вычисление с помощью реализации последовательного алгоритма (директивы '#pragma omp parallel for' игнорируются ввиду снижения числа потоков до 1)
double calcNorm0( const matrix A, const matrix B, const matrix C, const int rank, double * period ) {
	const register int	qran = rank * rank,
						threads = omp_get_max_threads();
	register double		result;
	double				start;

	// снижение числа потоков до 1 для последовательного выполнения
	omp_set_num_threads( 1 );

	// обнуление матриц с промежуточными значениями вычислений
	memset( B, 0, qran * sizeof( double ) );
	memset( C, 0, qran * sizeof( double ) );

	// отметка времени начала вычислений
	start = omp_get_wtime();

	mProd( A, A, B, rank );
	mProd( B, A, C, rank );
	result = mNorm( C, rank ); // вычисление нормы матрицы

	// определение длительности вычислений в секундах
	*period = omp_get_wtime() - start;

	// восстановление числа потоков
	omp_set_num_threads( threads );
	return result;
}

// вычисление с помощью реализации последовательного алгоритма, снабжённой директивами '#pragma omp parallel for'
double calcNorm1( const matrix A, const matrix B, const matrix C, const int rank, double * period ) {
	const register int	qran = rank * rank;
	register double		result;
	double				start;

	// обнуление матриц с промежуточными значениями вычислений
	memset( B, 0, qran * sizeof( double ) );
	memset( C, 0, qran * sizeof( double ) );

	// отметка времени начала вычислений
	start = omp_get_wtime();

	mProd( A, A, B, rank );
	mProd( B, A, C, rank );
	result = mNorm( C, rank ); // вычисление нормы матрицы

	// определение длительности вычислений в секундах
	*period = omp_get_wtime() - start;
	return result;
}

// вычисление с помощью реализации параллельного алгоритма
double calcNorm2( const matrix A, const matrix B, const matrix C, const int rank, double * period ) {
	const register int	qran = rank * rank,
						threads = omp_get_max_threads();
	register double		result;
	double				allRes[ threads ],
						start;

	// обнуление промежуточных значений вычислений
	memset( B, 0, qran * sizeof( double ) );
	memset( C, 0, qran * sizeof( double ) );
	memset( allRes, 0, threads * sizeof( double ) );
	result = .0;

	// отметка времени начала вычислений
	start = omp_get_wtime();

#pragma omp parallel for
	for( register int rowStart = 0; rowStart < qran; rowStart += rank ) {
		register int	colIndex, k;

		for( colIndex = 0; colIndex < rank; colIndex++ )	// mProd( A, A, B );
			for( k = 0; k < rank; k++ )
				B[ rowStart + colIndex ] += A[ rowStart + k ] * A[ k * rank + colIndex ];

		for( colIndex = 0; colIndex < rank; colIndex++ )	// mProd( B, A, C );
			for( k = 0; k < rank; k++ )
				C[ rowStart + colIndex ] += B[ rowStart + k ] * A[ k * rank + colIndex ];

		for( colIndex = 0; colIndex < rank; colIndex++ )	// mNorm( C ); - параллельная часть
			allRes[ omp_get_thread_num() ] += C[ rowStart + colIndex ] * C[ rowStart + colIndex ];
	}

#pragma omp barrier

	for( register int k = 0; k < threads; k++ )	// mNorm( C ); - последовательная часть
		result += allRes[ k ];
	result = sqrt( result );

	// определение длительности вычислений в секундах
	*period = omp_get_wtime() - start;
	return result;
}

// вычисление с помощью реализации параллельного алгоритма с предварительным транспонированием
double calcNorm3( const matrix A, const matrix B, const matrix C, const int rank, double * period ) {
	const register int	qran = rank * rank,
						threads = omp_get_max_threads();
	register double		result;
	const matrix		T = mMake( rank );
	double				allRes[ threads ],
						start;

	// обнуление промежуточных значений вычислений
	memset( B, 0, qran * sizeof( double ) );
	memset( C, 0, qran * sizeof( double ) );
	memset( allRes, 0, threads * sizeof( double ) );
	result = .0;

	// отметка времени начала вычислений
	start = omp_get_wtime();

#pragma omp parallel for
	for( register int rowIndex = 0; rowIndex < rank; rowIndex++ )	// mTrans( A, T ); - транспонирование матрицы A
		for( register int colIndex = 0; colIndex < rank; colIndex++ )
			T[ rowIndex * rank + colIndex ] = A[ colIndex * rank + rowIndex ];

#pragma omp barrier

#pragma omp parallel for
	for( register int rowStart = 0; rowStart < qran; rowStart += rank ) {
		register int	colIndex, k;

		for( colIndex = 0; colIndex < rank; colIndex++ )	// mProd( A, T, B );
			for( k = 0; k < rank; k++ )
				B[ rowStart + colIndex ] += A[ rowStart + k ] * T[ colIndex * rank + k ];

		for( colIndex = 0; colIndex < rank; colIndex++ )	// mProd( B, T, C );
			for( k = 0; k < rank; k++ )
				C[ rowStart + colIndex ] += B[ rowStart + k ] * T[ colIndex * rank + k ];

		for( colIndex = 0; colIndex < rank; colIndex++ )	// mNorm( C ); - параллельная часть
			allRes[ omp_get_thread_num() ] += C[ rowStart + colIndex ] * C[ rowStart + colIndex ];
	}

#pragma omp barrier

	for( register int k = 0; k < threads; k++ )	// mNorm( C ); - последовательная часть
		result += allRes[ k ];
	result = sqrt( result );

	// определение длительности вычислений в секундах
	*period = omp_get_wtime() - start;

	free( T );
	return result;
}

// устойчивое вычисление среднего времени выполнения
static inline double recalcAverage( const double average, const int number, const double value ) {
	return average + ( value - average ) / ( double )number;
}

int main( void ) {
	const int	threads[ THREADVARS ] = { 1, 2, 4 },
				rank[ RANKVARS ] = { 500, 2000, 3000 },
//				rank[ RANKVARS ] = { 300, 600, 1200 },
				attempts[ RANKVARS ][ THREADVARS ] = { { 20, 30, 40 }, // для порядка матрицы 500
								   { 10, 15, 20 }, // для порядка матрицы 2000
								   { 5, 8, 10 } }; // для порядка матрицы 3000
//				attempts[ RANKVARS ][ THREADVARS ] = { { 12, 16, 20 }, // для порядка матрицы 300
//								   { 6, 8, 10 }, // для порядка матрицы 600
//								   { 3, 4, 5 } }; // для порядка матрицы 1200
	double		result[ SCHEMES ],
				period,
				average[ SCHEMES ];
	int			diffs[ PARASCHEMES ];
	matrix 		A, B, C;

	printf( "Начата серия экспериментов по следующим схемам: \n"
				"\t- схема 0: проверочный последовательный алгоритм,\n"
				"\t- схема 1: последовательный алгоритм, снабжённый директивами '#pragma omp parallel for'\n"
				"\t- схема 2: параллельный алгоритм\n"
				"\t- схема 3: параллельный алгоритм с предварительным транспонированием\n" );
	fflush( stdout );

	srand48( ( long )time( NULL ) );

	for( register int rankIdx = 0; rankIdx < RANKVARS; rankIdx++ ) {
		// выделение памяти под матрицы
		A = mMake( rank[ rankIdx ] ), B = mMake( rank[ rankIdx ] ), C = mMake( rank[ rankIdx ] );

		for( register int threadsIdx = 0; threadsIdx < THREADVARS; threadsIdx++ ) {
			printf( "\nНачата серия экспериментов:\n"
						"\t- порядок: %d\n"
						"\t- потоков: %d\n"
						"\t- экспериментов: %d\n", rank[ rankIdx ], threads[ threadsIdx ], attempts[ rankIdx ][ threadsIdx ] );
			fflush( stdout );

			// установление числа потоков
			omp_set_num_threads( threads[ threadsIdx ] );

			// обнуление средних значений
			memset( average, 0, SCHEMES * sizeof( double ) );

			// обнуление числа отличий от эталона
			memset( diffs, 0, PARASCHEMES * sizeof( int ) );

			for( int attempt = 1; attempt <= attempts[ rankIdx ][ threadsIdx ]; attempt++ ) {
				// заполнение матрицы A случайными значениями
				mRand( A, rank[ rankIdx ] );

				// осуществление вычислений и замеров длительности по схеме 0
				result[ 0 ] = calcNorm0( A, B, C, rank[ rankIdx ], &period );
				average[ 0 ] = recalcAverage( average[ 0 ], attempt, period );

				// осуществление вычислений и замеров длительности по схеме 1
				result[ 1 ] = calcNorm1( A, B, C, rank[ rankIdx ], &period );
				average[ 1 ] = recalcAverage( average[ 1 ], attempt, period );

				// осуществление вычислений и замеров длительности по схеме 2
				result[ 2 ] = calcNorm2( A, B, C, rank[ rankIdx ], &period );
				average[ 2 ] = recalcAverage( average[ 2 ], attempt, period );

				// осуществление вычислений и замеров длительности по схеме 3
				result[ 3 ] = calcNorm3( A, B, C, rank[ rankIdx ], &period );
				average[ 3 ] = recalcAverage( average[ 3 ], attempt, period );

				// подсчёт числа отличий
				for( register int scheme = 1; scheme < 4; scheme++ )
					if( fabs( result[ scheme ] - result[ 0 ] ) >= ERROR )
						diffs[ scheme - 1 ]++;
			}

			printf( "Серия экспериментов завершена. Результаты:\n"
						"Схема Отличий Время Ускорение Эффективность\n"
						"%5d   --    %5.3f    --           --      \n", 0, average[ 0 ] );

			for( register int scheme = 1; scheme < 4; scheme++ )
				printf( "%5d %7d %5.3f %9.3f %13.3f\n", scheme, diffs[ scheme - 1 ], average[ scheme ], average[ 0 ] / average[ scheme ], average[ 0 ] / ( average[ scheme ] * threads[ threadsIdx ] ) );

			fflush( stdout );
		}

		free( A ), free( B ), free( C );
	}

	printf( "Все эксперименты завершены.\n" );
	return 0;
}
