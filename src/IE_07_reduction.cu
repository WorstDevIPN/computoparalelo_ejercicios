/****************************************************************\
 *																*
 * Titulo			:	Uso de memoria compartida				*
 * Organizacion		:	CIC-IPN									*
 * Autor	       	:	Oswaldo Franco							*
 * e-mail			:	oswaldo1509@gmail.com					*
 * Periodo			:	Semestre B20							*
 * Dependencies		:	cudaError.h								*
 * Referencia		:	Introduccion a la programacion			*
 *						en CUDA ( Universidad de Burgos )		*
 *																*
\****************************************************************/

#include<stdio.h>
#include<stdlib.h>

#define N 16

__global__ void reduction( float *vector, float *suma){
	__shared__ float temporal[N];

	int myID = threadIdx.x;
	temporal[myID] = vector[myID];
	__syncthreads();

	int salto = N/2;

	while(salto){
		if(myID< salto){
			temporal[myID] = temporal[myID] + temporal[myID+salto];
		}
		__syncthreads();
		salto = salto/2;
	}

	if(myID==0)	*suma = temporal[myID];
}

int main(){

	float *hst_vector;
	float hst_suma;
	float *dev_vector;
	float *dev_suma;


	hst_vector = (float*)malloc(N*sizeof(float));


	cudaMalloc( (void**)&dev_vector, N*sizeof(float));
	cudaMalloc( (void**)&dev_suma, sizeof(float));

	srand(time(NULL));

	for(int i=0; i<N; i++)	hst_vector[i] = (float) (rand()%16);

	cudaMemcpy( dev_vector, hst_vector, N*sizeof(float), cudaMemcpyHostToDevice);

	reduction<<<1,N>>>(dev_vector, dev_suma);

	cudaMemcpy( &hst_suma, dev_suma, sizeof(float), cudaMemcpyDeviceToHost);

	for (int i=0; i<N; i++){
		printf("%.2f\t", hst_vector[i]);
	}
	printf("\n");

	printf("suma %.2f\n", hst_suma);
	cudaFree(dev_vector);
	cudaFree(dev_suma);

	return 0;
}
