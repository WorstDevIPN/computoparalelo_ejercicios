/********************************************************
 *														*
 * Autor: Programacion_en_CUDA-Moises Arreola Zamora	*
 *														*
 * Fecha:28/10/2020										*
 * 														*
 * Lanzamiento de Kernel con multiples hilos y bloques:	*
 * Generacion de imagenes por medio de bloques de 		*
 * multiples hilos (16x16).						 		*
 * Dibujar un tablero de ajedrez de dim de 8 x 8  		*
 * 														*
 * compilacion con: 									*
 * nvcc practica_9.cu -o practica_9.o -lglut -lGL -lGLU	*
*********************************************************/
//librerias

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/common/cpu_bitmap.h"
#include "device_launch_parameters.h"


//definiciones
#define DIM 1024 // Dimensiones del Bitmap 1024

__global__ void kernel( unsigned char *imagen )
{
	//bloques
	// coordenada horizontal
	int bx = blockIdx.x;
	// coordenada vertical
	int by = blockIdx.y;
	//hilos
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	// coordenada vertical
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	// coordenada global de cada pixel
	int pixel = x + y * blockDim.x * gridDim.x;
	// cada bloque pinta un pixel con un color arbitrario
	if((bx+by)%256 == 0){
		imagen[pixel *4 + 0] = 0;// canal R
		imagen[pixel *4 + 1] = 0;// canal G
		imagen[pixel *4 + 2] = 0;// canal B
		imagen[pixel *4 + 3] = 255;								// canal alfa
	}
	else{
		imagen[pixel *4 + 0] = 255;// canal R
		imagen[pixel *4 + 1] = 255;// canal G
		imagen[pixel *4 + 2] = 255;// canal B
		imagen[pixel *4 + 3] = 255;	
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
	// declaracion del bitmap
	CPUBitmap bitmap( DIM, DIM );
	// tamaño en bytes
	size_t size = bitmap.image_size();
	// reserva en el host
	unsigned char *host_bitmap = bitmap.get_ptr();
	// reserva en el device
	unsigned char *dev_bitmap;
	cudaMalloc( (void**)&dev_bitmap, size );
	// generamos el bitmap
	dim3 Nbloques(DIM/16,DIM/16);
	dim3 hilosB(16,16);
	kernel<<<Nbloques,hilosB>>>( dev_bitmap );
	// recogemos el bitmap desde la GPU para visualizarlo
	cudaMemcpy( host_bitmap, dev_bitmap, size, cudaMemcpyDeviceToHost );
	// liberacion de recursos
	cudaFree( dev_bitmap );
	// visualizacion y salida
	bitmap.display_and_exit();
	printf("\n...Ejecucion Terminada...");
	
	return 0;
}

