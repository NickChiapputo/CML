#ifndef LAYERS_H_
#define LAYERS_H_

#include <inttypes.h>
#include "activation.h"

typedef struct 
{
	uint16_t numClasses;
	double * weights;
	double * bias;
	int  (*activation)(double*, int, double*);
	void * transfer;
} layer;

#endif