/********************************************************
 *														*
 * Autor: Programacion_en_CUDA-Moises Arreola Zamora	*
 *														*
 * Fecha:21/10/2020										*
 * 														*
 * Lanzamiento de Kernel con multiples hilos:			*
 * Calculo de la aproximacion de pi 					*
 *														*
 * compilacion con: 									*
 * nvcc practica_7.cu -o practica_7.o				 	*
*********************************************************/
//librerias

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>

//definiciones
#define NUM_HILOS	128

__global__ void calcular_pi_shared(float *matriz_ppal) {
	//reservado de memoria compartida de hilos
	__shared__ float array [NUM_HILOS];
	int id_threadx 	= threadIdx.x ;
	int hilo		= threadIdx.x +1;
	//calculo de la casilla y sincroniza
	array[id_threadx] = 1/(hilo*hilo);
	__syncthreads();
	//reduccion paralela para ir sumando terminos
	int salto = NUM_HILOS/2;
	while (salto){
		if(id_threadx < salto){
			array[id_threadx] = array[id_threadx] + array[id_threadx+salto];
		}
		__syncthreads();
		salto = salto>>1;
	}
	if (id_threadx == 0){
		//~ printf("\nEl valor aprox de pi es %f\n",sqrtf(array[id_threadx]*6));
		*matriz_ppal = array[id_threadx];
	}
}

__global__ void calcular_pi(float *matriz_ppal, float *arreglo) {
	//~ float array[NUM_HILOS];
	int id_threadx 	= threadIdx.x ;
	int hilo		= threadIdx.x +1;
	//calculo de la casilla y sincroniza
	arreglo[id_threadx] = 1/(hilo*hilo);
	__syncthreads();
	//reduccion paralela para ir sumando terminos
	int salto = NUM_HILOS/2;
	while (salto){
		if(id_threadx < salto){
			arreglo[id_threadx] = arreglo[id_threadx] + arreglo[id_threadx+salto];
		}
		__syncthreads();
		salto = salto/2;
	}
	if (id_threadx == 0){
		//~ printf("\nEl valor aprox de pi es %f\n",sqrtf(arreglo[id_threadx]*6));
		*matriz_ppal = arreglo[id_threadx];
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
	float host_resultado;
	
	//Temporizacion
	//~ cudaDeviceProp myGPU;
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//var device
	float *dev_matriz;
	float *dev_matriz_global;

	cudaMalloc( (void**)&dev_matriz, sizeof(float) );
	cudaMalloc( (void**)&dev_matriz_global, NUM_HILOS*sizeof(float) );
	check_CUDA_Error("Error en cudaMalloc\n");
	
	//invocacion kernel
	cudaEventRecord(start,0);
	
	calcular_pi_shared<<<1,NUM_HILOS>>>(dev_matriz);
	check_CUDA_Error("Error en calcular_pi\n");
		//copia de device a host
	cudaMemcpy(&host_resultado, dev_matriz, sizeof(float),
		cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error en cudaMemcpy D2H\n");
	//~ // salida	
	printf("\nLa aproximacion de Pi por memoria compartida es: %.9f\n",sqrt(6*host_resultado));
	
	cudaFree(dev_matriz);

	cudaMalloc( (void**)&dev_matriz, sizeof(float) );
	check_CUDA_Error("Error en cudaMalloc\n");
	
	calcular_pi<<<1,NUM_HILOS>>>(dev_matriz,dev_matriz_global);
	check_CUDA_Error("Error en calcular_pi\n");
		//copia de device a host
	cudaMemcpy(&host_resultado, dev_matriz, sizeof(float),
		cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error en cudaMemcpy D2H\n");
	//~ // salida	
	printf("\nLa aproximacion de Pi por memoria global es: %.9f\n",sqrt(6*host_resultado));
	
	
	//parando cronometro
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	
	
	printf("\nTiempo transcurrido en Device : %.3f\n",elapsedTime);
	//memory dump
	cudaFree(dev_matriz);
	cudaFree(dev_matriz);
	cudaFree(dev_matriz_global);
	//~ cudaFree(dev_matriz_ret);
	//~ free(host_resultado);
	//~ free(host_matriz_modificada);
	
	printf("\nFin de Programa \n");

	fflush(stdin);
	return 0;
}

