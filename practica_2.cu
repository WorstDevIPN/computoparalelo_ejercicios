// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#define N 8

//~ __global__ void copy(float* dev_A_matriz, float* dev_B_matriz) {
   //~ cudaMemcpy(dev_B_matriz, dev_A_matriz, N*N*sizeof(float),  //operaciones de memoria 
															//no se permiten en devices
		//~ cudaMemcpyDeviceToDevice);
//~ }


// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
// declaracion
float *hst_A_matriz;
float *hst_B_matriz;
float *dev_A_matriz;
float *dev_B_matriz;
size_t 	memlibre = 0;
size_t 	memtotal = 0;
// reserva en el host
hst_A_matriz = (float*)malloc( N*N*sizeof(float) );
hst_B_matriz = (float*)malloc( N*N*sizeof(float) );
// reserva en el device
cudaMalloc( (void**)&dev_A_matriz, N*N*sizeof(float) );
cudaMalloc( (void**)&dev_B_matriz, N*N*sizeof(float) );
// inicializacion de datos
srand ( (int)time(NULL) );
for (int i=0; i<N*N; i++){
	hst_A_matriz[i] = (float)( rand() % 2 );
}
// copia de datos
cudaMemcpy(dev_A_matriz, hst_A_matriz, N*N*sizeof(float), 
		cudaMemcpyHostToDevice);

// copia de datos entre dispositivos
//~ copy<<<1, 1>>> (dev_A_matriz, dev_B_matriz);
cudaMemcpy(dev_B_matriz, dev_A_matriz, N*N*sizeof(float), 
		cudaMemcpyDeviceToDevice);

//copia de device a host
cudaMemcpy(hst_B_matriz, dev_B_matriz, N*N*sizeof(float), 
		cudaMemcpyDeviceToHost);
// salida
for (int i=0; i<N*N; i++){
	if (hst_A_matriz[i]-hst_B_matriz[i]==0){
		printf("El valor Matriz A[%d] es igual a B[%d] %.2f - %.2f = %.2f \n",
			i,i,hst_A_matriz[i],hst_B_matriz[i],hst_A_matriz[i]-hst_B_matriz[i]);
	}
}
cudaMemGetInfo(&memlibre,&memtotal);
printf("La memoria libre es de %zu y la memoria total es de %zu",memlibre,memtotal);
cudaFree( dev_A_matriz );
cudaFree( dev_B_matriz );
printf("\npulsa INTRO para finalizar...");
fflush(stdin);
char tecla = getchar();
return 0;
}
