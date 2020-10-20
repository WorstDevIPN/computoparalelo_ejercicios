/****************************************************************\
 *																*
 * Titulo			:	Ejecucion de hilos						*
 * Organizacion		:	CIC-IPN									*
 * Autor	       	:	Oswaldo Franco							*
 * e-mail			:	oswaldo1509@gmail.com					*
 * Periodo			:	Semestre B20							*
 * Dependencies		:	cudaError.h								*
 * Referencia		:	Introduccion a la programacion			*
 *						en CUDA ( Universidad de Burgos )		*
 *																*
\****************************************************************/

// includes
#include <stdio.h>
#include <stdlib.h>

// defines
#define N 16

// tamano de los vectores
#define BLOCK 5 // tamano del bloque

// declaracion de funciones
// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void suma( float *a, float *b, float *c ) {
	int myID = threadIdx.x + blockDim.x * blockIdx.x;
	// Solo trabajan N hilos
	if (myID < N) {
		c[myID] = a[myID] + b[myID];
	}
}

// MAIN: rutina principal ejecutada en el host

int main(int argc, char** argv) {
	// declaraciones
	float *vector1, *vector2, *resultado;
	float *dev_vector1, *dev_vector2, *dev_resultado;
	
	// reserva en el host
	vector1 = (float *)malloc(N*sizeof(float));
	vector2 = (float *)malloc(N*sizeof(float));
	resultado = (float *)malloc(N*sizeof(float));
	
	// reserva en el device
	cudaMalloc( (void**)&dev_vector1, N*sizeof(float));
	cudaMalloc( (void**)&dev_vector2, N*sizeof(float));
	cudaMalloc( (void**)&dev_resultado, N*sizeof(float));
	
	// inicializacion de vectores
	for (int i = 0; i < N; i++) {
		vector1[i] = (float) rand() / RAND_MAX;
		vector2[i] = (float) rand() / RAND_MAX;
	}
	
	// copia de datos hacia el device
	cudaMemcpy(dev_vector1, vector1, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vector2, vector2, N*sizeof(float), cudaMemcpyHostToDevice);
	
	// lanzamiento del kernel
	// calculamos el numero de bloques necesario para un tamaño de bloque fijo
	int nBloques = N/BLOCK;
	if (N%BLOCK != 0) {
		nBloques = nBloques + 1;
	}

	int hilosB = BLOCK;
	printf("Vector de %d elementos\n", N);
	printf("Lanzamiento con %d bloques (%d hilos)\n", nBloques, nBloques*hilosB);
	suma<<< nBloques, hilosB >>>( dev_vector1, dev_vector2, dev_resultado );

	// recogida de datos desde el device
	cudaMemcpy(resultado, dev_resultado, N*sizeof(float), cudaMemcpyDeviceToHost);
	// impresion de resultados
	printf( "> vector1:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", vector1[i]);
	}
	printf("\n");

	printf( "> vector2:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", vector2[i]);
	}
	printf("\n");

	printf( "> SUMA:\n");
	for (int i = 0; i < N; i++) {
		printf("%.2f ", resultado[i]);
	}
	printf("\n");
	
	// liberamos memoria en el device
	cudaFree( dev_vector1 );
	cudaFree( dev_vector2 );
	cudaFree( dev_resultado );
	
	// salida
	//printf("\npulsa INTRO para finalizar...");
	return 0;
}
