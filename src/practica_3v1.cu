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

// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void chicharron(int a, int b, int c, float *dev_x) {
	//distincion de raiz imaginaria
	int ima_x = (b*b)-(4*a*c);
	
	if (ima_x < 0){
		ima_x = -1 * ima_x;
		dev_x[0] = ((sqrtf(ima_x))/(2*a));
		dev_x[1] = (-1)*((sqrtf(ima_x))/(2*a));
		return;
	}
	else{
		//~ real_p = (float)0;
		dev_x[0] = ((-b)/(2*a))+((sqrtf(ima_x))/(2*a));
		dev_x[1] = ((-b)/(2*a))-((sqrtf(ima_x))/(2*a));
		return;
	}

}


// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
// declaracion var host
int coef_a;
int coef_b;
int coef_c;
float host_real;

float *host_x;

//var device
float *dev_x;

//reserva al host
host_x = (float*)malloc(2*sizeof(float));

// reserva en el device
cudaMalloc( (void**)&dev_x, 2*sizeof(float) );

//ingreso de valores 
printf("Ingresa los coeficientes de la ecuacion ax^2 + bx + c = 0\n");
printf("Coeficiente a\t");
scanf("%d",&coef_a);
printf("Coeficiente b\t");
scanf("%d",&coef_b);        
printf("Coeficiente c\t");
scanf("%d",&coef_c);
printf("\n");
host_real =(-(float)coef_b)/(2*(float)coef_a);

//invocacion kernel		
chicharron<<<1, 1>>> (coef_a, coef_b, coef_c, dev_x);	

//copia de device a host
cudaMemcpy(host_x, dev_x, 2*sizeof(float),cudaMemcpyDeviceToHost);

// salida
if((coef_b*coef_b)-(4*coef_a*coef_c) < 0){
	
	printf("Las soluciones son:\n x1 = %f %f i\t x2 = %f %f i", host_real, host_x[0], host_real, host_x[1]);
}
else
	printf("Las soluciones son:\n x1 = %f \t x2 = %f ", host_x[0], host_x[1]);
	
cudaFree( dev_x );
free(host_x);
printf("\nFin de Programa \n");

fflush(stdin);
return 0;
}
