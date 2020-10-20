/****************************************************************\
 *																*
 * Titulo			:	Ejecucion de hilos multidimensionales	*
 * Organizacion		:	CIC-IPN									*
 * Autor	       	:	Oswaldo Franco							*
 * e-mail			:	oswaldo1509@gmail.com					*
 * Periodo			:	Semestre B20							*
 * Dependencies		:	cudaError.h								*
 *																*
\****************************************************************/

#include<stdio.h>
#include<stdio.h>
#include"cudaError.h"

#define	N 16

__device__ int index(int columna, int fila){
	return columna + fila * N;
}

__global__ void kernel(int *org, int *mod){

	int columna = threadIdx.x;
	int fila = threadIdx.y;
	int id = columna + fila * blockDim.x;

	//printf("global %d \t", global);

	if (columna%N>0 && columna%N<N-1 && fila%N>0 && fila<N-1){

		mod[id] = org[index(columna-1, fila)] + org[index(columna+1, fila)] + org[index(columna,fila-1)] + org[index(columna, fila+1)];
	}
	else{
		mod[id] = org[id];
	}
}

int main(){
	int *hst_matrixA, *hst_matrixB;
	int *dev_matrixA, *dev_matrixB;
	dim3 Nbloques(1);
	dim3 hilosB(N,N);
	cudaEvent_t start, stop;
	float elapsedTime;
	
	srand(time(NULL));

	hst_matrixA = (int*)malloc(N*N*sizeof(int));
	hst_matrixB = (int*)malloc(N*N*sizeof(int));
	cudaMalloc((void**)&dev_matrixA,N*N*sizeof(int));
	cudaMalloc((void**)&dev_matrixB,N*N*sizeof(int));

	for(int i=0; i<N*N; i++)
		hst_matrixA[i] = rand() % 2;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaMemcpy(dev_matrixA, hst_matrixA, N*N*sizeof(int), cudaMemcpyHostToDevice);
	check_CUDA_Error("ERROR de Memcpy");

	cudaEventRecord(start,0);

	kernel<<<Nbloques, hilosB >>>(dev_matrixA, dev_matrixB);
	check_CUDA_Error("ERROR de Kernel");

	cudaEventRecord(stop,0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);


	cudaMemcpy(hst_matrixB, dev_matrixB, N*N*sizeof(int), cudaMemcpyDeviceToHost);
	check_CUDA_Error("ERROR de Memcpy");

	printf("Tiempo de ejecucion %.8f\n",elapsedTime);

	for(int i=0; i<N*N; i++){
		if (i%N == 0) printf("\n");
		printf("%d \t", hst_matrixA[i]);
	}
	printf("\n");

	for(int i=0; i<N*N; i++){
		if (i%N == 0) printf("\n");
		printf("%d \t", hst_matrixB[i]);
	}
	printf("\n");
	
	cudaFree(dev_matrixA);
	cudaFree(dev_matrixB);

	return 0;
}
