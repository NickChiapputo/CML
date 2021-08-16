#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "loss.h"

double LOSS_crossEntropy( double ** predictions, uint8_t ** targets, int numPredictions, int numClasses )
{
    double ce = 0;
    int prediction, class;

    // Since the targets are one-hot arrays, we only need to search for the target that is not zero
    // and then break out of the inner loop to save extra iterations. Averaging and multiplying by one
    // is done after the summation.
    for( prediction = 0; prediction < numPredictions; prediction++ )
    {
        for( class = 0; class < numClasses; class++ )
        {
            if( targets[ prediction ][ class ] )
            {
                ce += log( predictions[ prediction ][ class ] );
            }
        }
    }

    ce *= -( (double)1 / numPredictions );
    return ce;
}
