/************************************************************\
 *															*
 * Titulo			:	Propiedades de GPU					*
 * Organizacion		:	CIC-IPN								*
 * Autor	       	:	Oswaldo Franco						*
 * e-mail			:	oswaldo1509@gmail.com				*
 * Periodo			:	Semestre B20						*
 * Dependencies		:										*
 *															*
\************************************************************/

#include<stdio.h>

int main() {
	int count;
	int device;
	cudaDeviceProp GPU;

	printf("Hola Mundo Cuda\n");
	cudaGetDeviceCount(&count);

	printf("total GPGPUS: %d\n",count);

	device = (count==1) ? 0:-1;

	cudaGetDeviceProperties(& GPU, device);
	printf("GPU name: %s \
		\n GPU global memory(GB) %u \
		\n GPU shared memory(KB) %u \
		\n GPU warpsize %d \
		\n GPU maxThreadsPerBlock %d \
		\n GPU multiprocesadores %d \
	       	\n"\
		, GPU.name, (unsigned int)(GPU.totalGlobalMem/(1024*1000*1000))\
		, (unsigned int) GPU.sharedMemPerBlock/1024 ,GPU.warpSize\
		, GPU.maxThreadsPerBlock, GPU.multiProcessorCount);

	return 0;
}
