#include <stdio.h>
#include <math.h>

struct cuComplex {
        float   r;
        float   i;
        
        cuComplex( float a, float b ) : r(a), i(b) {}
        
        cuComplex operator*(const cuComplex& a){
                return cuComplex(r*a.r - i*a.i, i*a.r+r*a.i);
        }
        
        cuComplex operator+(const cuComplex& a){
                return cuComplex(r+a.r, i+a.i);
        }
        
        cuComplex operator-(const cuComplex& a){
                return cuComplex(r-a.r, i-a.i);
        }
        
        float magnitude( void ){
                return r*r + i*i;
        }
        
        float phase(){
        	return (float) atan2((double) i,(double)r);
        }
        
        cuComplex conjugate(){
        	return cuComplex(r,-i);
        }
        
        cuComplex scale(float n){
        	return cuComplex(n*r, n*i);
        }
};

