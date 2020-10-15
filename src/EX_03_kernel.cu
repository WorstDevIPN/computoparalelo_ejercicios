/****************************************************************\
 *								*
 * Titulo		:	Lanzamiento de kernel		*
 * Organizacion		:	CIC-IPN				*
 * Autor	       	:	Oswaldo Franco			*
 * e-mail		:	oswaldo1509@gmail.com		*
 * Periodo		:	Semestre B20			*
 * Dependencies		:					*
 *								*
\****************************************************************/

#include<stdio.h>
#include<math.h>

struct cuComplex {
        float   r;
        float   i;
        cuComplex( float a, float b ) : r(a), i(b) {}
};


__global__ void chicharronera( float a, float b, float c, cuComplex *x){
	// calculo de chicharronera
	// falta manejo de imaginarios a + bi
	float det;
	det = b*b - 4*a*c;
	if (det >= 0){
		x[0].r = (-b/2*a) + (sqrtf(det) / 2*a);
		x[0].i = 0.0;
		x[1].r = (-b/2*a) - (sqrtf(det) / 2*a);
		x[1].i = 0.0;
	}
	else{
		x[0].r = (-b/2*a);
		x[0].i = (sqrtf(-det) / 2*a);
                x[1].r = (-b/2*a);
		x[1].i = (sqrtf(-det) / 2*a);
	}
}

int main(){
	int a, b, c;
	cuComplex *hst_x;
	cuComplex *dev_x;

	// reserva de memoria en host y device para resultados
	hst_x = (cuComplex*)malloc( 2*sizeof(cuComplex));
	cudaMalloc( (void**)&dev_x, 2*sizeof(cuComplex));
	
	// ingreso de datos
	printf("Ingresa los coeficientes a, b y c de la ecuacion ax^2 + bx + c = 0\n");
	scanf("%d %d %d", &a, &b, &c);

	// lanzamiento de kernel
	chicharronera<<<1,1>>>((float)a,(float)b,(float)c,dev_x);

	// recuperar resultados de Device a Host
	cudaMemcpy(hst_x, dev_x, 2*sizeof(cuComplex), cudaMemcpyDeviceToHost);

	//  mostrar resultados
	printf("\nx1 = %.2f + %.2fi \t x2 = %.2f - %.2fi\n", hst_x[0].r, hst_x[0].i, hst_x[1].r, hst_x[1].i);

	// liberar memoria en host y device
	cudaFree(dev_x);
	free(hst_x);

	return 0;

}
