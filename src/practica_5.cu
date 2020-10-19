/********************************************************
 *														*
 * Autor: Programacion_en_CUDA-Moises Arreola Zamora	*
 *														*
 * Fecha:19/10/2020										*
 * 														*
 * Lanzamiento de Kernel con multiples hilos y bloques:	*
 * los hilos por bloque son fijos a 4 y el numero de 	*
 * bloques depende del tamaño del arreglo de elementos	*
 * Cada hilo puede invertir como maximo un par de 		*
 * elementos											*
 * 														*
 * compilacion con: 									*
 * nvcc practica_5.cu -o practica_5.o				 	*
*********************************************************/
//librerias

#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

//definiciones
#define NUM_HILOS	4
#define TAM_MIN		3


__global__ void reordena(int N, int punto_medio, int bandera, int *arreglo_ppal) {
	int id_global 	= threadIdx.x + (blockDim.x * blockIdx.x);
	int aux_swap;
	//~ printf("tamaño = %d",N);
	if (N <= TAM_MIN){
		//~ printf("Arreglo de tamaño minimo\n");
		if (id_global == 0){
			if (N == 3){
				aux_swap 		= arreglo_ppal[0];
				arreglo_ppal[0] = arreglo_ppal[2];
				arreglo_ppal[2]	= aux_swap;
				return;
			}
			else if(N == 2){
				aux_swap 		= arreglo_ppal[0];
				arreglo_ppal[0] = arreglo_ppal[1];
				arreglo_ppal[1] = aux_swap;
				return;
			}
			else{
				printf("Tamaño invalido\n");
				return;
			}
		}
		else
			return;
	}
	else{
		if ( (id_global + 1) == punto_medio && bandera > 0){
			//~ printf("Se llego a punto medio y se termina el swap\n");
			return;
		} 
		else {
			aux_swap 					= arreglo_ppal[id_global];
			arreglo_ppal[id_global] 	= arreglo_ppal[N - (id_global+1)];
			arreglo_ppal[N - (id_global+1)] 	= aux_swap;
		}
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
	int tam_vector;
	int bloques = 0; 		//dependiente del tamaño del arreglo
		
	int *host_arreglo_ordenado;
	int *host_arreglo_reordenado;
	
	//Temporizacion
	//~ cudaDeviceProp myGPU;
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//var device
	int *dev_arreglo_ordenado;
	
	//ingreso de valores 
	printf("Ejecutando Practica 5\t ");
	printf("Ingresa el tamaño del Vector \n");
	scanf("%d",&tam_vector);
	if(tam_vector <= 1){
		printf("El tamaño es invalido, saliendo del programa \n");
		return 0;
	}
	
	//variable de control
	bloques 	= tam_vector/4;
	if(bloques < 1) bloques = 1;
	int media 	= tam_vector/2; 	//limite para habilitar hilos
	int modulo	= tam_vector%2;
	
	//~ printf("Los valores de las variables de control:\nNo. de bloques: %d\t Punto medio: %d\t Modulo: %d \n",
			//~ bloques, media, modulo);

	//reserva al host
	host_arreglo_ordenado = (int*)malloc(tam_vector*sizeof(int));
	host_arreglo_reordenado = (int*)malloc(tam_vector*sizeof(int));

	// reserva en el device
	cudaMalloc( (void**)&dev_arreglo_ordenado, tam_vector*sizeof(int) );
	check_CUDA_Error("Error en cudaMalloc\n");
	
	//rellenado del arreglo
	printf("\nLos valores del arreglo son: \n");
	srand ( (int)time(NULL) );
	
	for (i=0; i < tam_vector; i++){
		host_arreglo_ordenado[i] = (int)( rand() % 10 );
		printf("Arreglo_Ordenado[%d] = %d \n", i,
				host_arreglo_ordenado[i] );
	}
	// copia de datos
	cudaMemcpy(dev_arreglo_ordenado, host_arreglo_ordenado, tam_vector*sizeof(int), 
		cudaMemcpyHostToDevice);
	check_CUDA_Error("Error en cudaMemcpy H2D\n");

	//invocacion kernel
	cudaEventRecord(start,0);
	reordena<<<bloques,NUM_HILOS>>>(tam_vector, media, modulo, dev_arreglo_ordenado);
	check_CUDA_Error("Error en reordena\n");
	
	//copia de device a host
	//~ cudaDeviceSynchronize();
	cudaMemcpy(host_arreglo_reordenado, dev_arreglo_ordenado, tam_vector*sizeof(int),
		cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error en cudaMemcpy D2H\n");
	
	//parando cronometro
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	//~ // salida	
	//~ desplegar_datos(host_arreglo_reordenado);
	printf("\nLos valores del arreglo reordenados son:\n");
	for (i=0; i < tam_vector; i++){
		printf("Arreglo_ReOrdenado[%d] = %d \n", i,
				host_arreglo_reordenado[i] );
	}
	
	printf("\nTiempo transcurrido en Device : %.3f\n",elapsedTime);
	//memory dump
	cudaFree(dev_arreglo_ordenado);
	free(host_arreglo_ordenado);
	free(host_arreglo_reordenado);
	
	printf("\nFin de Programa \n");

	fflush(stdin);
	return 0;
}

