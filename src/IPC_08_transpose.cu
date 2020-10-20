/****************************************************************\
 *																*
 * Titulo			:	Uso de memoria constante				*
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
#include <cuda_runtime.h>

// defines
#define N 8

// CUDA constants
__constant__ float dev_A[N][N];

// declaracion de funciones

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void traspuesta( float *dev_B) {
	// kernel lanzado con un solo bloque y NxN hilos
	int columna = threadIdx.x;
	int fila = threadIdx.y;
	int pos = columna + N*fila;
	// cada hilo coloca un elemento de la matriz final
	dev_B[pos] = dev_A[columna][fila];
}

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv) {
	// declaraciones
	float *hst_A, *hst_B;;
	float *dev_B;
	
	// reserva en el host
	hst_A = (float*)malloc(N*N*sizeof(float));
	hst_B = (float*)malloc(N*N*sizeof(float));

	// reserva en el device
	cudaMalloc( (void**)&dev_B, N*N*sizeof(float));
	// inicializacion
	for(int i=0; i<N*N; i++) {
		hst_A[i]= (float)i;
	}// copia de datos
	cudaMemcpyToSymbol( dev_A, hst_A, N*N*sizeof(float));

	// dimensiones del kernel
	dim3 Nbloques(1);

	dim3 hilosB(N,N);

	// llamada al kernel bidimensional de NxN hilos
	traspuesta<<<Nbloques,hilosB>>>(dev_B);

	// recogida de datos
	cudaMemcpy( hst_B, dev_B, N*N*sizeof(float), cudaMemcpyDeviceToHost );

	// impresion de resultados

	printf("Resultado:\n");
	printf("ORIGINAL:\n");
	for(int i=0; i<N; i++) {
		for(int j=0; j<N;j++) {
			printf("%2.0f ",hst_A[j + i*N]);
		}
		printf("\n");
	}
	
	printf("TRASPUESTA:\n");
	for(int i=0;i<N;i++) {
		for(int j=0; j<N; j++) {
			printf("%2.0f ",hst_B[j + i*N]);
		}
		printf("\n");
	}

	return 0;
}
