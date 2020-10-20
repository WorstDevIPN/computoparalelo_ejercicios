/********************************************************
 *														*
 * Autor: Programacion_en_CUDA-Moises Arreola Zamora	*
 *														*
 * Fecha:12/10/2020										*
 * 														*
 * Lanzamiento de Kernel								*
 *														*
 * 														*
 * compilacion con: 									*
 * nvcc fractal.cu -o fractal.out -lglut -lGL -lGLU		*
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


__global__ void rellenado( int *array) {
	int id_global 	= threadIdx.x + blockDim.x * blockIdx.x;
	int id_thread 	= threadIdx.x;
	int id_block 	= blockIdx.x;
	//no usar for, apoyarse del no de bloques o el no de threads	
	array[id_global] = id_thread;
	array[id_global+24] = id_block;
	array[id_global+48] = id_global;
	
}

// MAIN:
int main(int argc, char** argv)
{
// declaracion var host
int i;
int bloques = 0;
int hilos 	= 0;
int *host_x;

//var device
int *dev_array;

//reserva al host
host_x = (int*)malloc(NUM_MATRICES*PROFUNDIDAD*sizeof(int));

// reserva en el device
cudaMalloc( (void**)&dev_array, NUM_MATRICES*PROFUNDIDAD*sizeof(int) );

//ingreso de valores 
printf("Ejecutando Practica 4\n");

//invocacion kernel
for (i = 0; i < NUM_MATRICES; i++){
	if(i == 0){
		bloques	=	1;
		hilos	=	24;
		rellenado<<<1,24>>>(dev_array);
	}
	else if( i == 1){
		bloques	=	24;
		hilos	=	1;
		rellenado<<<bloques,hilos>>>(dev_array);
	}
	else {
		bloques	=	4;
		hilos	=	6;
		rellenado<<<bloques,hilos>>>(dev_array);
	}
	printf("\nbloques %d hilos %d\n", bloques,hilos);
	//copia de device a host
	cudaMemcpy(host_x, dev_array, NUM_MATRICES*PROFUNDIDAD*sizeof(float),cudaMemcpyDeviceToHost);
	printf("hilo\n");
	for(int i = 0; i<PROFUNDIDAD; i++){
		printf("%d \t",host_x[i]);
	}
	printf("\nbloque\n");
	for(int i = 0; i<PROFUNDIDAD; i++){
		printf("%d \t",host_x[i+24]);
	}
	printf("\ngloabal\n");
	for(int i = 0; i<PROFUNDIDAD; i++){
		printf("%d \t",host_x[i+48]);
	}

}

cudaFree(dev_array);
//~ free(host_x);
printf("\nFin de Programa \n");

fflush(stdin);
return 0;
}

