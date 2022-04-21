#ifndef LAYERS_H_
#define LAYERS_H_

#include <inttypes.h>
#include "activation.h"

/**** Layer Struct Definitions ****/
typedef struct 
{
	uint16_t numClasses;
	double * weights;
	double * bias;
	int  (*activation)(double*, int, double*);
	void * transfer;
} LAYERS_activation;

typedef struct LAYERS_Conv2D
{
	uint16_t inFilters;
	uint16_t outFilters;

	int16_t strideX;
	int16_t strideY;

	int16_t kernelWidth;
	int16_t kernelHeight;

	uint8_t padded;

	double **** weights;	// Shape = (outFilters)x(inFilters)x(kernelHeight)x(kernelWidth)
	double *   bias;		// Shape = (inFiltesr * outFilters)x1

	int (*forward)(double****, double***, uint16_t, uint16_t, 
				   uint16_t, struct LAYERS_Conv2D);

	int (*activation)(double*, int, double*);
} LAYERS_Conv2D;


/**** Helper Functions ****/
int LAYERS_load_weights( double ***** weights, double ** bias, 
    uint16_t inC, uint16_t outC, 
    uint16_t filterHeight, uint16_t filterWidth,
    char * filename );


/**** Forward Pass Functions ****/
/****
 	*	@description
****/ 
int LAYERS_convolution_2d( double **** output, double *** input, uint16_t inY,
						   uint16_t inX, uint16_t inC, LAYERS_Conv2D layer );

#endif