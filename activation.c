#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "activation.h"


int ACTIVATION_softmax( double * input, int inputLength, double * output )
{
	// We want to calculate the maximum value in the input
	// in order to prevent possible issues later on in the event
	// that the values become sufficiently large.
	// See: https://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
	double maxInput = input[ 0 ];
	for( int i = 1; i < inputLength; i++ )
		if( input[ i ] > maxInput )
			maxInput = input[ i ];


	double sum = 0;
	for( int i = 0; i < inputLength; i++ )
		sum += exp( input[ i ] - maxInput );


	double offset = maxInput + log( sum );
	for( int i = 0; i < inputLength; i++ )
		output[ i ] = exp( input[ i ] - offset );

	return 0;
}

