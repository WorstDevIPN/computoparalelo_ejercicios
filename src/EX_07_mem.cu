/****************************************************************\
 *																*
 * Titulo			:	Uso de memoria compartida				*
 * Organizacion		:	CIC-IPN									*
 * Autor	       	:	Oswaldo Franco							*
 * e-mail			:	oswaldo1509@gmail.com					*
 * Periodo			:	Semestre B20							*
 * Dependencies		:	cudaError.h								*
 *																*
\****************************************************************/

#include<stdio.h>
#include<math.h>
#define N 32
__global__ void kernelshared( float *aprox_pi){
	__shared__ float vector[N];
	int myID = threadIdx.x;

	vector[myID] = 1.0/((myID + 1)*(myID + 1));
	__syncthreads();

	int salto = N/2;

	while(salto){
		if(myID<salto){
			vector[myID] = vector[myID] + vector[myID + salto];
		}
		__syncthreads();
		salto = salto >> 1;
	}
	if(myID==0) *aprox_pi = vector[myID];

}


__global__ void kernelglobal( float *vector, float *aprox_pi){

	int myID = threadIdx.x;
	vector[myID] = 1.0/((myID + 1)*(myID + 1));
	__syncthreads();

	int salto = N/2;

	while(salto){
		if(myID<salto){
			vector[myID] = vector[myID] + vector[myID + salto];
		}
		__syncthreads();
		salto = salto >> 1;
	}

	__syncthreads();

	if(myID==0) *aprox_pi = vector[myID];

}

int main(){

	float hst_aprox_pi;
	float *dev_aprox_pi;
	cudaDeviceProp myGPU;
	cudaEvent_t start, stop;
	cudaEvent_t startG, stopG;
	float elapsedTime, elapsedTimeG;
	float *dev_vector;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&startG);
	cudaEventCreate(&stopG);


	cudaGetDeviceProperties( &myGPU, 0);
	printf("maxThreadPerBlock: %d\n", myGPU.maxThreadsPerBlock);
	printf("sharedMemPerBlock: %zu\n", myGPU.sharedMemPerBlock/1024);
	printf("Launched threads: %d\n", N);

	cudaMalloc( (void**)&dev_aprox_pi, sizeof(float) );

	cudaEventRecord(start,0);
	kernelshared<<<1,N>>>(dev_aprox_pi);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	cudaMemcpy(&hst_aprox_pi, dev_aprox_pi, sizeof(float), cudaMemcpyDeviceToHost);

	printf("Elapsed Time: %.3f\n",elapsedTime);
	printf("Valor de Pi %.8f\n", sqrt(hst_aprox_pi*6));

	cudaFree(dev_aprox_pi);
	
	cudaMalloc( (void**)&dev_aprox_pi, sizeof(float) );
	cudaMalloc( (void**)&dev_vector, N*sizeof(float) );

	cudaEventRecord(startG,0);
	kernelglobal<<<1,N>>>(dev_vector, dev_aprox_pi);
	cudaEventRecord(stopG,0);
	cudaEventSynchronize(stopG);
	cudaEventElapsedTime(&elapsedTimeG, startG, stopG);

	cudaMemcpy(&hst_aprox_pi, dev_aprox_pi, sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("Elapsed Time: %.3f\n",elapsedTimeG);
	printf("Valor de Pi %.8f\n", sqrt(hst_aprox_pi*6));
	
	cudaFree(dev_aprox_pi);

	return 0;
}
