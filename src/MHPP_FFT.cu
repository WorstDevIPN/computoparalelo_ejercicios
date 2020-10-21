#include <stdio.h>
#include <stdlib.h>
#include "complex.h"

#define N 4

cuComplex * fft(cuComplex *, int);

int main(){
	cuComplex *hst_array;
	hst_array = (cuComplex*)malloc(N*sizeof(cuComplex));

	srand(time(NULL));

	for(int i=0; i<N; i++){
		hst_array[i] = cuComplex((float)i+1.0,0.0);
		printf("array %d = (%.2f,%.2fi)\n",i, hst_array[i].r, hst_array[i].i);
	}

	cuComplex *hst_fft = fft(hst_array, N);

	for(int i=0; i<N; i++){
		printf("array %d = (%.2f,%.2fi)\n",i, hst_fft[i].r, hst_fft[i].i);
	}
	
	return 0;
}

    /**
     * Performs the FFT to an array of complex values.
     * @param x The Complex[] array of values.
     * @return Complex[] representing the FFT of x.
     */


cuComplex * fft(cuComplex *x, int n) {
        if (n == 1) {
		return x;
        }
        // Even terms
        int len = (n % 2 != 0) ? (n + 1) / 2 : n / 2;

        //Complex[] even = new Complex[len];
	cuComplex *even;
	even = (cuComplex*)malloc(len*sizeof(cuComplex));

        for (int k = 0; k < len; k++) {
            even[k] = x[2 * k];
        }
        //Complex[] e = fft(even);
	cuComplex *e = fft(even, len);
	
        
        // Odd terms
        len = (n % 2 != 0) ? (n - 1) / 2 : n / 2;
        //Complex[] odd = new Complex[len];
	cuComplex *odd;
	odd = (cuComplex*)malloc(len*sizeof(cuComplex));
        for (int k = 0; k < len; k++) {
            odd[k] = x[2 * k + 1];
        }
        //Complex[] o = fft(odd);
	cuComplex *o = fft(odd, len);
        
        // Combine
        len = (n % 2 != 0) ? (n - 1) / 2 : n / 2;
        //Complex[] y = new Complex[n];
	
	cuComplex *y;
	y = (cuComplex*)malloc(len*sizeof(cuComplex));

        for (int k = 0; k < n/2; k ++) {
		double kth = -2 * k * M_PI / n;
		cuComplex wk = cuComplex(cos(kth), sin(kth));
		y[k] = e[k] + (wk*o[k]);
		y[k + n/2] = e[k]-(wk*o[k]);
        }

        return y;
}
  

cuComplex* ifft(cuComplex *x, int n) {
	cuComplex *y;
	y = (cuComplex*)malloc(n*sizeof(cuComplex));

	// Take conjugate
	for (int i = 0; i < n; i++) {
		y[i] = x[i].conjugate();
	}
        
	// Compute forward FFT
	y = fft(y, n);
        
	// Take conjugate again
	for (int i = 0; i < n; i++) {
		y[i] = y[i].conjugate();
	}
        
	// Divide by n
	for (int i = 0; i < n; i++) {
		y[i] = y[i].scale(1.0 / n);
	}
	return y;
}
