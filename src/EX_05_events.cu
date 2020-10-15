/************************************************************\
 *															*
 * Titulo			:	Creación de eventos					*
 * Organizacion		:	CIC-IPN								*
 * Autor	       	:	Oswaldo Franco						*
 * e-mail			:	oswaldo1509@gmail.com				*
 * Periodo			:	Semestre B20						*
 * Dependencies		:	cudaError.h							*
 *															*
\************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include"cudaError.h"

#define nHilos	4

__global__ void reverse(int *org, int *rev, int N){
	int global = threadIdx.x + blockDim.x * blockIdx.x;
	rev[(N-1)-global] = org[global];
}

int main(){
	int N=0;
	int *hst_arr, *hst_rev;
	int *dev_arr, *dev_rev;
	float elapsedTime;
	cudaEvent_t start, stop;
	
	printf("Ingresa N ");
	scanf("%d", &N);

	hst_arr = (int*)malloc( N*sizeof(float));
	hst_rev = (int*)malloc( N*sizeof(float));

	cudaMalloc( (void**)&dev_arr, N*sizeof(int) );
	cudaMalloc( (void**)&dev_rev, N*sizeof(int) );

	srand(time(NULL));

	for(int i=0; i<N; i++){
		hst_arr[i] = rand() % 9;
	}
	
	cudaMemcpy(dev_arr, hst_arr, N*sizeof(int),cudaMemcpyHostToDevice);
	check_CUDA_Error("ERROR EN cudaMemcpy");

	int nBloques = N / nHilos;

	nBloques = (N%nHilos==0) ? nBloques : nBloques+1;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);
	reverse<<< nBloques, nHilos >>>(dev_arr, dev_rev, N);
	check_CUDA_Error("ERROR EN reverse");
	cudaEventRecord(stop,0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Tiempo de ejecucion en GPU %f ms\n",elapsedTime);

	cudaMemcpy(hst_rev, dev_rev, N*sizeof(int),cudaMemcpyDeviceToHost);
	check_CUDA_Error("ERROR EN cudaMemcpy");

	printf("arreglo original\n");
	for(int i=0; i<N; i++){
		printf("%d\t",hst_arr[i]);
	}
	printf("\n");

	printf("arreglo en reversa\n");
	for(int i=0; i<N; i++){
		printf("%d\t",hst_rev[i]);
	}
	printf("\n");

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(dev_arr);
	cudaFree(dev_rev);
	free(hst_arr);
	free(hst_rev);
	return 0;
}
