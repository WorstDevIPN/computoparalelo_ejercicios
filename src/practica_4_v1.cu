/********************************************************
 *														*
 * Autor: Programacion_en_CUDA-Moises Arreola Zamora	*
 *														*
 * Fecha:12/10/2020										*
 * 														*
 * Lanzamiento de Kernel con multiples hilos y bloques	*
 *														*
 * 														*
 * compilacion con: 									*
 * nvcc practica_4.cu -o practica_4.o				 	*
*********************************************************/
//librerias

#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

//definiciones
#define NUM_MATRICES	3
#define PROFUNDIDAD		24

#define PRIMER_SEG		24
#define	SEGUNDO_SEG		48
#define TERCER_SEG		72


__global__ void rellenado( int *array_threads, int *array_blocks, int *array_global) {
	int id_global 	= threadIdx.x + (blockDim.x * blockIdx.x);
	int id_thread 	= threadIdx.x;
	int id_block 	= blockIdx.x;
	
	//no usar for, apoyarse del no de bloques o el no de threads
	array_threads[id_global] = id_thread;
	array_blocks[id_global] = id_block;
	array_global[id_global] = id_global;

}

__host__ void desplegar_datos(int *thread, int *block, int *global){
	int j;
	
	for (j=0; j < NUM_MATRICES*PROFUNDIDAD; j++){
		if (j < PRIMER_SEG){
			printf("Arreglo_hilo [%d] \t%d\n", j, thread[j]);
		}
		else if(PRIMER_SEG <= j && j < SEGUNDO_SEG){
			if (j == PRIMER_SEG) printf("\n");
			printf("Arreglo_bloque [%d] \t%d\n", j, block[j-24]);
		}
		else if(j >= SEGUNDO_SEG && j < TERCER_SEG){
			if (j == SEGUNDO_SEG) printf("\n");
			printf("Array_global [%d] \t%d\n", j, global[j-48]);
		}
	}
}

// MAIN:
int main(int argc, char** argv)
{
	// declaracion var host
	int i;
	int bloques = 0;
	int hilos 	= 0;
	int *host_thread;
	int *host_block;
	int *host_global;
	//~ cudaDeviceProp myGPU;
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//var device
	int *dev_array_global;
	int *dev_array_threads;
	int *dev_array_blocks;


	//reserva al host
	host_thread = (int*)malloc(PROFUNDIDAD*sizeof(int));
	host_block = (int*)malloc(PROFUNDIDAD*sizeof(int));
	host_global= (int*)malloc(PROFUNDIDAD*sizeof(int));

	// reserva en el device
	cudaMalloc( (void**)&dev_array_global, PROFUNDIDAD*sizeof(int) );
	cudaMalloc( (void**)&dev_array_threads, PROFUNDIDAD*sizeof(int) );
	cudaMalloc( (void**)&dev_array_blocks, PROFUNDIDAD*sizeof(int) );

	//ingreso de valores 
	printf("Ejecutando Practica 4\n");

	//invocacion kernel
	cudaEventRecord(start,0);
	for (i = 0; i < NUM_MATRICES; i++){
		if(i == 0){
			bloques	=	1;
			hilos	=	24;
			//~ rellenado<<1,24>>(dev_array);
		}
		else if( i == 1){
			bloques	=	24;
			hilos	=	1;
			//~ rellenado<<bloques,hilos>>(dev_array);
		}
		else {
			bloques	=	4;
			hilos	=	6;
			//~ rellenado<<bloques,hilos>>(dev_array);
		}
		printf("\nbloques %d hilos %d\n\n", bloques,hilos);
		
		rellenado<<<bloques,hilos>>>(dev_array_threads, dev_array_blocks, dev_array_global);
		
		//~ //copia de device a host
		//~ cudaMemcpy(host_x, dev_array_threads, NUM_MATRICES*PROFUNDIDAD*sizeof(int),cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		//copia de device a host
		cudaMemcpy(host_thread, dev_array_threads, PROFUNDIDAD*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(host_block, dev_array_blocks, PROFUNDIDAD*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(host_global, dev_array_global, PROFUNDIDAD*sizeof(int),cudaMemcpyDeviceToHost);
		//~ // salida	
		desplegar_datos(host_thread, host_block, host_global);
	}
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	printf("Elapsed Time: %.3f\n",elapsedTime);

	//memory dump
	cudaFree(dev_array_threads);
	cudaFree(dev_array_blocks);
	cudaFree(dev_array_global);
	
	free(host_thread);
	free(host_block);
	free(host_global);
	
	printf("\nFin de Programa \n");

	fflush(stdin);
	return 0;
}

