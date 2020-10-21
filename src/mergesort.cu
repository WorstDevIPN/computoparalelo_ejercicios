/********************************************************
 *														*
 * Autor: Programacion_en_CUDA-Moises Arreola Zamora	*
 *														*
 * Fecha:21/10/2020										*
 * 														*
 * Lanzamiento de Kernel con multiples hilos y bloques:	*
 * 														*
 * Algoritmo de ordenamiento mergesort 					*
 * 														*
 * compilacion con (revisar por guiones):				*
 * nvcc -rdc=true −arch=compute_35 mergesort.cu −o sort	*
 * nvprof --unified-memory-profiling off ./sort			*
*********************************************************/
//librerias

#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"
//~ #include "../include/common/book.h"
#include <cuda_runtime.h>

//definiciones
#define N 	16 //

__global__ void merge(float *lado_izq, float *lado_der, float *unido, int sizes) {
	int i = 0;
	int j = 0;
	int k = 0;

	while ( i<sizes/2 && j<sizes/2 ){
		if ( lado_izq[i] <= lado_der[j] ){
			unido[k] = lado_izq[i];
			i ++;
		}
		else{
			unido[k] = lado_der[j];
			j ++;
		}
		k++;
	}
	while ( i<sizes/2 ){
		unido[k] = lado_izq[i];
		i ++;
		k ++;
	}
	while ( j<sizes/2 ){
		unido[k] = lado_der[j];
		j ++;
		k ++;
	}	
}

__global__ void merge_sort(float *L, float *R, int size){
	//~ int id_threadx 	= threadIdx.x ;
	//Obtencion del tamaño de arreglos
	int i;
	int tamano = size /2;
	
	//~ printf("El id del thread es %d : %d\n",id_threadx, blockDim.x);
	if (size == 1){
		//~ printf("Condicion de salida de recursion\n");
		return;
	}
	else{
		float *Lft;
		float *Rgt;
		
		cudaMalloc((void**)&Lft, sizeof(float)*tamano);
		cudaMalloc((void**)&Rgt, sizeof(float)*tamano);
		
		if (threadIdx.x == 0){			
			//intercambio de posiciones
			memcpy(Lft, L, tamano*sizeof(float));
			memcpy(Rgt, &L[tamano], tamano*sizeof(float));
			//recursion
			merge_sort<<<1,2>>>(Lft, Rgt, tamano);
			merge<<<1,1>>>(Lft, Rgt, L, size);		
		}
		else if(threadIdx.x == 1){
			//intercambio de posiciones
			memcpy(Lft, R, tamano*sizeof(float));
			memcpy(Rgt, &R[tamano], tamano*sizeof(float));
			//recursion
			merge_sort<<<1,2>>>(Lft, Rgt, tamano);
			merge<<<1,1>>>(Lft, Rgt, R, size);		
		}
	}
	__syncthreads();
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

	float *host_array;
	float *host_array_L;
	float *host_array_R;
	
	//Temporizacion
	//~ cudaDeviceProp myGPU;
	cudaEvent_t start, stop;
	float elapsedTime;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//var device
	float *dev_array;
	float *dev_array_L;
	float *dev_array_R;
	
	//ingreso de valores 
		
	//variables de control
		
	//reserva al host
	host_array	 = (float*)malloc(N*sizeof(float));
	host_array_L = (float*)malloc(N/2*sizeof(float));
	host_array_R = (float*)malloc(N/2*sizeof(float));

	// reserva en el device
	cudaMalloc( (void**)&dev_array, N*sizeof(float) );
	cudaMalloc( (void**)&dev_array_L, N/2*sizeof(float) );
	cudaMalloc( (void**)&dev_array_R, N/2*sizeof(float) );//*2
	check_CUDA_Error("Error en cudaMalloc\n");
	
	//rellenado de la matriz
	printf("\nLos valores del arreglo son: \n");
	srand ( (float)time(NULL) );
	
	for (i=0; i < N; i++){
		host_array[i] = (float)( rand() %N+1);
		printf("Arreglo_Desordenado[%d] = %.0f \n", i,host_array[i] );
	}
	
	//rellenado de las matrices  izq y der
	for (i=0; i < N/2; i++){
		host_array_L[i] = host_array[i];
		host_array_R[i] = host_array[N/2+i];		
	}
	printf("Valores a ordenar Derecha:\n");
	for (i=0; i < N/2; i++){
		printf("D[%d]:%.0f\t",i,host_array_R[i]);	
	}
	printf("\nValores a ordenar Izquierda:\n");
	for (i=0; i < N/2; i++){
		printf("I[%d]:%.0f\t",i,host_array_L[i]);	
	}

	// copia de datos a device
	cudaMemcpy(dev_array_L, host_array_L, N/2*sizeof(float), 
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_array_R, host_array_R, N/2*sizeof(float), 
		cudaMemcpyHostToDevice);
	check_CUDA_Error("Error en cudaMemcpy H2D\n");
	
	cudaEventRecord(start,0);
	//invocacion kernel
	merge_sort<<<1,2>>>(dev_array_L, dev_array_R, N/2); // /2
	//respaldo	
	cudaMemcpy(host_array_L, dev_array_L, N/2*sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array_R, dev_array_R, N/2*sizeof(float),
		cudaMemcpyDeviceToHost);
	//
	merge<<<1,1>>>(dev_array_L, dev_array_R, dev_array, N);//*2 
	check_CUDA_Error("Error en modificar_matriz\n");
	
	//copia de device a host
	//~ cudaDeviceSynchronize();
	//~ cudaMemcpy(host_array_L, dev_array_L, N*sizeof(float),
		//~ cudaMemcpyDeviceToHost);
	//~ cudaMemcpy(host_array_R, dev_array_R, N*sizeof(float),
		//~ cudaMemcpyDeviceToHost);
	cudaMemcpy(host_array, dev_array, N*sizeof(float),
		cudaMemcpyDeviceToHost);
	check_CUDA_Error("Error en cudaMemcpy D2H\n");
	
	//parando cronometro
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	//~ // salida	
	//~ desplegar_datos(host_arreglo_reordenado);
	printf("\nLos valores del arreglo reordenados son:\n");
	for (i=0; i < N; i++){
		printf("Arreglo_ordenado[%d] = %.0f \n", i,
					host_array[i] );
	}
	
	printf("\nTiempo transcurrido en Device : %.3f\n",elapsedTime);
	//memory dump
	cudaFree(dev_array_L);
	cudaFree(dev_array_R);
	cudaFree(dev_array);
	
	free(host_array);
	free(host_array_L);
	free(host_array_R);
	
	printf("\nFin de Programa \n");

	fflush(stdin);
	return 0;
}

