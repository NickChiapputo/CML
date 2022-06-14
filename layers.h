#ifndef LAYERS_H_
#define LAYERS_H_

#include <inttypes.h>
#include "activation.h"

/**** Layer Struct Definitions ****/
typedef struct 
{
	uint16_t numClasses;
	DATATYPE * weights;
	DATATYPE * bias;
	int  (*activation)(DATATYPE*, int, DATATYPE*);
	void * transfer;
} LAYERS_activation;

typedef struct LAYERS_PreLU
{
	uint16_t outFilters;
	DATATYPE * weights;
	int (*forward)(DATATYPE****, DATATYPE***, uint16_t, uint16_t, 
		struct LAYERS_PreLU);
} LAYERS_PreLU;

typedef struct LAYERS_ReLU
{
	uint16_t outFilters;
	int (*forward)(DATATYPE****, DATATYPE***, uint16_t, uint16_t, 
		struct LAYERS_ReLU);
} LAYERS_ReLU;

typedef struct LAYERS_Conv2D
{
	uint16_t inFilters;
	uint16_t outFilters;

	int16_t strideX;
	int16_t strideY;

	int16_t kernelWidth;
	int16_t kernelHeight;

	uint8_t padded;

	DATATYPE **** weights;	// Shape = (outFilters)x(inFilters)x(kernelHeight)x(kernelWidth)
	DATATYPE *   bias;		// Shape = (inFiltesr * outFilters)x1

	int (*forward)(DATATYPE****, DATATYPE***, uint16_t, uint16_t, 
				   uint16_t, struct LAYERS_Conv2D);

	void * nextLayer;
} LAYERS_Conv2D;


/**** Layer Constructors ****/
int LAYERS_conv2d_constructor( LAYERS_Conv2D * layer, 
	uint16_t inFilters, uint16_t outFilters,
	uint16_t kernelWidth, uint16_t kernelHeight,
    uint16_t strideX, uint16_t strideY,
	uint16_t padded, 
	char * weightsFile,
	int (*forward)(DATATYPE****, DATATYPE***, uint16_t, uint16_t, uint16_t, LAYERS_Conv2D) );

int LAYERS_prelu_constructor( LAYERS_PreLU * layer,
    uint16_t outFilters, char * weightsFile,
    int (*forward)(DATATYPE****, DATATYPE***, uint16_t, uint16_t, LAYERS_PreLU) );

int LAYERS_relu_constructor( LAYERS_ReLU * layer, uint16_t outFilters,
    int (*forward)(DATATYPE****, DATATYPE***, uint16_t, uint16_t, LAYERS_ReLU) );

/**** Helper Functions ****/
int LAYERS_load_weights( DATATYPE ***** weights, DATATYPE ** bias, 
    uint16_t inC, uint16_t outC, 
    uint16_t filterHeight, uint16_t filterWidth,
    char * filename );

int LAYERS_prelu_load_weights( DATATYPE ** weights, uint16_t outC, 
	char * filename );

int LAYERS_conv2d_free_weights( LAYERS_Conv2D * layer );

int LAYERS_prelu_free_weights( LAYERS_PreLU * layer );


/**** Forward Pass Functions ****/
/****
 	*	@description
****/ 
int LAYERS_convolution_2d( DATATYPE **** output, DATATYPE *** input, uint16_t inY,
						   uint16_t inX, uint16_t inC, LAYERS_Conv2D layer );

int LAYERS_prelu_forward( DATATYPE **** output, DATATYPE *** input, 
						  uint16_t inY, uint16_t inX, LAYERS_PreLU layer );

int LAYERS_relu_forward( DATATYPE **** output, DATATYPE *** input, 
                          uint16_t inY, uint16_t inX, LAYERS_ReLU layer );
#endif