/************************************************************\
 *															*
 * Titulo			:	Memoria de la GPU					*
 * Organizacion		:	CIC-IPN								*
 * Autor	       	:	Oswaldo Franco						*
 * e-mail			:	oswaldo1509@gmail.com				*
 * Periodo			:	Semestre B20						*
 * Dependencies		:										*
 *															*
\************************************************************/

#include<stdio.h>
#include<time.h>

#define N 8

int main(){
	float *hst_matrixA;
	float *hst_matrixB;

	float *dev_matrixA;
	float *dev_matrixB;
	
	size_t memLibre, memTotal;

	hst_matrixA = (float *)malloc( N*N*sizeof(float) );
	hst_matrixB = (float *)malloc( N*N*sizeof(float) );
	cudaMalloc ( (void**)&dev_matrixA, N*N*sizeof(float) );
	cudaMalloc ( (void**)&dev_matrixB, N*N*sizeof(float) );

	srand( (int)time(NULL) );

	for(int i=0; i<N*N; i++){
		hst_matrixA[i] = (float)(rand()%10);
	}

	cudaMemcpy(dev_matrixA, hst_matrixA, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matrixB, dev_matrixA, N*N*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(hst_matrixB, dev_matrixB, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	printf("======== Matriz Hst A ===========\n");

	for(int i=0; i<N*N; i++){
		if (i%8 == 0) printf("\n");
		printf("%.2f \t ", hst_matrixA[i]);
	}
	printf("\n");

	printf("======== Matriz Hst B ===========\n");

	for(int i=0; i<N*N; i++){
		if (i%8 == 0) printf("\n");
		printf("%.2f \t ", hst_matrixB[i]);
	}

	printf("\n");

	cudaMemGetInfo( &memLibre, &memTotal);
	printf("Libre %zu MB Total %zu MB \n", memLibre/1024, memTotal/1024);

	cudaFree(dev_matrixA);
	cudaFree(dev_matrixB);

	return 0;
}
