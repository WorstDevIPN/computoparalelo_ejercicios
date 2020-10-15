/************************************************************\
 *															*
 * Titulo			:	Ejecucion de hilos					*
 * Organizacion		:	CIC-IPN								*
 * Autor	       	:	Oswaldo Franco						*
 * e-mail			:	oswaldo1509@gmail.com				*
 * Periodo			:	Semestre B20						*
 * Dependencies		:										*
 *															*
\************************************************************/

#include<stdio.h>

#define N 24

__global__ void kernel(int *thread, int *block, int *global){
	int myId = threadIdx.x;
	int myBlock = blockIdx.x;
	int myGlobal = threadIdx.x + blockDim.x * blockIdx.x;

	thread [myGlobal] = myId;
	block [myGlobal] = myBlock;
	global [myGlobal] = myGlobal;
}

int main(){
	int *hst_thread, *hst_block, *hst_global;
	int *dev_thread, *dev_block, *dev_global;
	int option;

	hst_thread = (int*) malloc(N*sizeof(int));
	hst_block = (int*) malloc(N*sizeof(int));
	hst_global = (int*) malloc(N*sizeof(int));

	cudaMalloc( (void**)&dev_thread, N*sizeof(int) );
	cudaMalloc( (void**)&dev_block, N*sizeof(int) );
	cudaMalloc( (void**)&dev_global, N*sizeof(int) );
	printf("Ingresa opcion: \n\
		0: 1B24H\n\
		1: 24B1H\n\
		2: 4B6H\n");
	scanf("%d",&option);
	printf("Opcion %d:", option);
	switch (option){
		case 1:
			kernel<<<N,1>>>(dev_thread, dev_block, dev_global);
			printf("24 bloques 1 hilo \n");
			break;
		case 2:
			kernel<<<4,6>>>(dev_thread, dev_block, dev_global);
			printf("4 bloques 6 hilos \n");
			break;
		default:
			kernel<<<1,N>>>(dev_thread, dev_block, dev_global);
			printf("1 bloque 24 hilos \n");
			break;
	}

	cudaMemcpy(hst_thread, dev_thread, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_block, dev_block, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_global, dev_global, N*sizeof(int), cudaMemcpyDeviceToHost);

	printf("indice de hilo\n");
	for (int i=0; i<N; i++)	printf("%d\t", hst_thread[i]);
	printf("\n");
	printf("indice de bloque\n");
	for (int i=0; i<N; i++)	printf("%d\t", hst_block[i]);
	printf("\n");
	printf("indice global\n");
	for (int i=0; i<N; i++)	printf("%d\t", hst_global[i]);
	printf("\n");

	cudaFree(dev_thread);
	cudaFree(dev_block);
	cudaFree(dev_global);

	return 0;
}
