/****************************************************************\
 *																*
 * Titulo			:	Uso de memoria constante				*
 * Organizacion		:	CIC-IPN									*
 * Autor	       	:	Oswaldo Franco							*
 * e-mail			:	oswaldo1509@gmail.com					*
 * Periodo			:	Semestre B20							*
 * Dependencies		:	cudaError.h								*
 *																*
\****************************************************************/

#include<stdio.h>
#include<stdlib.h>

#define N 4

__constant__ float dev_A[N][N];
__constant__ float dev_B[N][N];


__global__ void matmul(float *C){
	int columna = threadIdx.x;
	int fila = threadIdx.y;
	int pos = columna + fila * N;
	
	for (int i=0; i<N; i++){
		C[pos] += dev_A[fila][i] * dev_B[i][columna];
	}
}

int main(){
	float *hst_A, *hst_B, *hst_C;
	float *dev_C;
	dim3 Nbloques(1);
	dim3 hilosB(N,N);
	cudaDeviceProp myGPU;
	cudaEvent_t start, stop;
	float elapsedTime;
	size_t memLibre, memTotal;
	

	srand(time(NULL));

	hst_A = (float*)malloc(N*N*sizeof(float));
	hst_B = (float*)malloc(N*N*sizeof(float));
	hst_C = (float*)malloc(N*N*sizeof(float));

	cudaMalloc ((void**)&dev_C, N*N*sizeof(float));

	for(int i=0; i<N*N; i++){
		hst_A[i] = (float) (rand()%2);
		hst_B[i] = (float) (rand()%2);
	}

	cudaGetDeviceProperties(&myGPU, 0);
	printf("totalConstMem: %zu\n", myGPU.totalConstMem);

	cudaMemcpyToSymbol(dev_A, hst_A, N*N*sizeof(float));
	cudaMemcpyToSymbol(dev_B, hst_B, N*N*sizeof(float));

	cudaMemGetInfo(&memLibre, &memTotal);
	printf("memLibre: %zu\n", memLibre);


	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start,0);

	matmul<<< Nbloques, hilosB >>>(dev_C);

	cudaEventRecord(stop,0);

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Elapsed Time: %.2f\n",elapsedTime);

	cudaMemcpy(hst_C, dev_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			printf("%.2f\t",hst_A[j + i*N]);
		}
		printf("\n");
	}
	printf("\n");

	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			printf("%.2f\t",hst_B[j + i*N]);
		}
		printf("\n");
	}
	printf("\n");

	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			printf("%.2f\t",hst_C[j + i*N]);
		}
		printf("\n");
	}
	printf("\n");

	cudaFree(dev_C);
	return 0;
}
