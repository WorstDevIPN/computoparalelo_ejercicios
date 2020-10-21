#include "common.h"
#include<stdio.h>
#include<stdlib.h>

#define N 32

__global__ void merge(int *left, int *right, int *merged, int size)	{

	int i = 0, j = 0, k = 0;

	while ( i<size/2 && j<size/2 ) {
		if ( left[i] <= right[j] ) {
			merged[k] = left[i];
			i ++;
		}
		else {
			merged[k] = right[j];
			j ++;
		}
		k++;
	}
	while ( i<size/2 ) {
		merged[k] = left[i];
		i ++;
		k ++;
	}
	while ( j<size/2 ){
		merged[k] = right[j];
		j ++;
		k ++;
	}

}


__global__ void mergesort (int *LEFT, int *RIGHT,  int size)	{
	int tid = threadIdx.x;
	int dsize = size >> 1;

	if ( size == 1) {
		return;
	}
	else {	
	        int *left;
        	int *right;
			cudaMalloc((void**)&left, sizeof(int) * dsize);
			cudaMalloc((void**)&right, sizeof(int) * dsize);
		if (tid == 0){                
			memcpy(left, LEFT, dsize*sizeof(int));
			memcpy(right, &LEFT[dsize], dsize*sizeof(int));
			mergesort<<<1,2>>>(left, right, dsize);
			merge<<<1,1>>>(left, right, LEFT, size);
		}
		else if (tid == 1){
			memcpy(left, RIGHT, dsize*sizeof(int));
			memcpy(right, &RIGHT[dsize], dsize*sizeof(int));
			mergesort<<<1,2>>>(left, right, dsize);
			merge<<<1,1>>>(left, right, RIGHT, size);
		}
	}
	__syncthreads();
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

	cudaMalloc( (void**)&dev_l, N*sizeof(int) ) ;
	cudaMalloc( (void**)&dev_r, N*sizeof(int) );
	cudaMalloc( (void**)&dev_m, 2*N*sizeof(int) );

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


	cudaMemcpy( dev_l, l, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy( dev_r, r, N*sizeof(int), cudaMemcpyHostToDevice);
	
	iStart = seconds();

	mergesort<<<1,2>>>(dev_l, dev_r, N);

    cudaMemcpy( l, dev_l, N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy( r, dev_r, N*sizeof(int),cudaMemcpyDeviceToHost);

	merge<<<1,1>>>(dev_l, dev_r, dev_m, 2*N);
	
	iElaps = seconds() - iStart;
	
	cudaMemcpy( m, dev_m, 2*N*sizeof(int),cudaMemcpyDeviceToHost);

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

