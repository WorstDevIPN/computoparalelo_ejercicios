// bug en ordenamiento
// Data to merge: {0,0,1,4,4,3,1,3,},{2,2,0,0,6,0,1,1,}
// KernelOnGPU Time elapsed 0.000007 sec
// Merged Data: {0,0,0,0,1,1,1,1,0,2,2,3,3,4,4,6,}

#include "common.h"
#include<stdio.h>
#include<stdlib.h>

#include"book.h"
#define N 16

__device__ int mr[2*N];

__global__ void merge(int *left, int *right, int *merged, int size)	{

	int i = 0, j = 0, k = 0;

	while ( i<size/2 && j<size/2 ) {
		if ( left[i] <= right[j] ) {
			mr[k] = left[i];
			i ++;
		}
		else {
			mr[k] = right[j];
			j ++;
		}
		k++;
	}
	while ( i<size/2 ) {
		mr[k] = left[i];
		i ++;
		k ++;
	}
	while ( j<size/2 ){
		mr[k] = right[j];
		j ++;
		k ++;
	}

	for (int i=0; i<2*size; i++){
		merged[i] = mr[i];
	}

}


__global__ void mergesort (int *left, int *right,  int size)	{
	int tid = threadIdx.x;
	int dsize = size >> 1;

	if ( size == 1) {
		return;
	}
	else {	
        int *lleft;
        int *lright;
		cudaMalloc((void**)&lleft, sizeof(int) * dsize);
		cudaMalloc((void**)&lright, sizeof(int) * dsize);
		if (tid == 0){
            for(int i = 0; i < dsize; i++) {
                lleft[i] = left[i];
                lright[i] = left[dsize + i];
            }          
			mergesort<<<1,2>>>(lleft, lright, dsize);
			cudaDeviceSynchronize();
			merge<<<1,1>>>(lleft, lright, left, size);
		}
		else if (tid == 1){
            for(int i = 0; i < dsize; i++) {
                lleft[i] = right[i];
                lright[i] = right[dsize + i];
            }          
			mergesort<<<1,2>>>(lleft, lright, dsize);
			cudaDeviceSynchronize();
			merge<<<1,1>>>(lleft, lright, right, size);
		}
	}
}


int main (void)	{
	int *l;
	int *r;
	int *m;
	int *dev_l, *dev_r, *dev_m;
	int hi=2*N, lo=0;
    double iStart, iElaps;
    
    
    l = (int*)malloc(sizeof(int) * N);
    r = (int*)malloc(sizeof(int) * N);
    m = (int*)malloc(sizeof(int) * 2 * N);
    
	
	srand(time(NULL));

	HANDLE_ERROR( cudaMalloc( (void**)&dev_l, N*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_r, N*sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_m, 2*N*sizeof(int) ) );

    for(int i = 0; i < N; i++) {
        l[i] = rand() % (hi - lo) + 1;
        r[i] = rand() % (hi - lo) + 1;
    }

	printf("Data to merge: {");
	for (int i=0; i<N; i++)	{
		printf("%d,", l[i]);
	}
	printf("},{");
	for (int i=0; i<N; i++)	{
		printf("%d,", r[i]);
	}
	printf("}\n");


	HANDLE_ERROR( cudaMemcpy( dev_l, l, N*sizeof(int), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy( dev_r, r, N*sizeof(int), cudaMemcpyHostToDevice) );

    iStart = seconds();
	mergesort<<<1,2>>>(dev_l, dev_r, N);
	cudaDeviceSynchronize();
	merge<<<1,1>>>(dev_l, dev_r, dev_m, 2*N);
    iElaps = seconds() - iStart;
	
	HANDLE_ERROR( cudaMemcpy( m, dev_m, 2*N*sizeof(int),cudaMemcpyDeviceToHost) );

	printf("Merged Data: {");
	for (int i=0; i<N; i++)	{
		printf("%d,%d,", m[2*i], m[2*i+1]);
	}
	printf("}\n");

    printf("KernelOnGPU Time elapsed %f sec\n",iElaps);
	
	free(l);
	free(r);
	free(m);

	cudaFree( dev_l );
	cudaFree( dev_r );
	cudaFree( dev_m );
	cudaDeviceSynchronize();
}

