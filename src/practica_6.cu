/********************************************************
 *														*
 * Autor: Programacion_en_CUDA-Moises Arreola Zamora	*
 *														*
 * Fecha:19/10/2020										*
 * 														*
 * Lanzamiento de Kernel con multiples hilos y bloques:	*
 * Manipulacion de los datos usando arreglos multidim	*
 * con el manejo de las variables dim3.					*
 * Realizar la sustitucion de valores de una matriz de 	*
 * 16 por 16 elementos manejando el mismo no de hilos	*
 * 														*
 * compilacion con: 									*
 * nvcc practica_6.cu -o practica_6.o				 	*
*********************************************************/
//librerias

#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

//definiciones
#define NUM_HILOS	16
//~ #define TAM_MIN		3


__global__ void modificar_matriz(int *matriz_ppal, int *matriz_mod) {
	//~ int id_globalx 	= threadIdx.x + (blockDim.x * blockIdx.x);
	//~ int id_globaly 	= threadIdx.y + (blockDim.y * blockIdx.y);
	int id_threadx 	= threadIdx.x ;
	int id_thready 	= threadIdx.y ;
	printf("Hilo id_globalx[%d] id_globaly[%d] = %d\n",id_threadx, id_thready,matriz_ppal[id_threadx+id_thready*NUM_HILOS]);
	
	//discriminar bordes
	if (id_threadx == 0 || id_thready == 0 || id_threadx == NUM_HILOS-1 || id_thready == NUM_HILOS-1 ){
		matriz_mod[id_threadx+id_thready*NUM_HILOS] = matriz_ppal[id_threadx+id_thready*NUM_HILOS];
		return;
	}
	else{
	//vaceado de datos
		matriz_mod[id_threadx+id_thready*NUM_HILOS] = matriz_ppal[(id_threadx-1)+id_thready*NUM_HILOS] +
				matriz_ppal[id_threadx+(id_thready-1)*NUM_HILOS] + matriz_ppal[(id_threadx+1)+id_thready*NUM_HILOS] +
				matriz_ppal[id_threadx+(id_thready+1)*NUM_HILOS];
	}
}

__host__ void check_CUDA_Error(const char *mensaje){
	cudaError_t error;
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	if(error != cudaSuccess){
		printf("ERROR %d: %s (%s)\n", error, cudaGetErrorString(error),
		mensaje );
		printf("\npulsa INTRO para finalizar...");
		fflush(stdin);
		char tecla =getchar();
		exit(-1);
	}
}


// MAIN:
int main(int argc, char** argv)
{
	// declaracion var host
	int i;
	int contador = 1;

	int *host_matriz_original;
	int *host_matriz_modificada;
	
	//Temporizacion
	//~ cudaDeviceProp myGPU;
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//var device
	int *dev_matriz;
	int *dev_matriz_ret;
	
	//ingreso de valores 
		
	//variables de control
		
	//reserva al host
	host_matriz_original = (int*)malloc(NUM_HILOS*NUM_HILOS*sizeof(int));
	host_matriz_modificada = (int*)malloc(NUM_HILOS*NUM_HILOS*sizeof(int));

	// reserva en el device
	cudaMalloc( (void**)&dev_matriz, NUM_HILOS*NUM_HILOS*sizeof(int) );
	cudaMalloc( (void**)&dev_matriz_ret, NUM_HILOS*NUM_HILOS*sizeof(int) );
	check_CUDA_Error("Error en cudaMalloc\n");
	
	//rellenado de la matriz
	printf("\nLos valores del arreglo son: \n");
	srand ( (int)time(NULL) );
	
	for (i=0; i < NUM_HILOS*NUM_HILOS; i++){
		
		if (i == (contador * NUM_HILOS)) {
			printf("\n");
			contador+=1;
		}
		host_matriz_original[i] = (int)( rand() % 2 );
		printf("Matriz_Inicial[%d] = %d \n", i,
					host_matriz_original[i] );
	}
	contador = 1;
	// copia de datos
	cudaMemcpy(dev_matriz, host_matriz_original, NUM_HILOS*NUM_HILOS*sizeof(int), 
		cudaMemcpyHostToDevice);
	check_CUDA_Error("Error en cudaMemcpy H2D\n");
	
	//dimensiones del kernel
	dim3 Nbloques(1);
	dim3 Nhilos(NUM_HILOS,NUM_HILOS); 	//16*16

	//invocacion kernel
	cudaEventRecord(start,0);
	
	modificar_matriz<<<Nbloques,Nhilos>>>(dev_matriz, dev_matriz_ret);
	check_CUDA_Error("Error en modificar_matriz\n");
	
	//copia de device a host
	//~ cudaDeviceSynchronize();
	cudaMemcpy(host_matriz_modificada, dev_matriz_ret, NUM_HILOS*NUM_HILOS*sizeof(int),
		cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error en cudaMemcpy D2H\n");
	
	//parando cronometro
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	//~ // salida	
	//~ desplegar_datos(host_arreglo_reordenado);
	printf("\nLos valores del arreglo reordenados son:\n");
	for (i=0; i < NUM_HILOS*NUM_HILOS; i++){
		if (i == (contador * NUM_HILOS)) {
			printf("\n");
			contador+=1;
		}
		printf("Matriz_Modificada[%d] = %d \n", i,
					host_matriz_modificada[i] );
	}
	
	printf("\nTiempo transcurrido en Device : %.3f\n",elapsedTime);
	//memory dump
	cudaFree(dev_matriz);
	cudaFree(dev_matriz_ret);
	free(host_matriz_original);
	free(host_matriz_modificada);
	
	printf("\nFin de Programa \n");

	fflush(stdin);
	return 0;
}

