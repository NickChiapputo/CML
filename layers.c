#include <stdio.h>

#include "debug.h"
#include "layers.h"

int LAYERS_convolution_2d( double **** output, double *** input, uint16_t inY,
                           uint16_t inX, uint16_t inC, LAYERS_Conv2D layer )
{
    // Calculate the output size.
    // Fast ceiling division: https://stackoverflow.com/a/14878734/2898057
    uint16_t outY, outX;
    if( layer.padded )
    {
        outY = inY / layer.strideY + ( inY % layer.strideY != 0 );
        outX = inX / layer.strideX + ( inX % layer.strideX != 0 );
    }
    else
    {
        uint16_t diff = inY - layer.kernelHeight + 1;
        outY = diff / layer.strideY + ( diff % layer.strideY != 0 );

        diff = inX - layer.kernelWidth + 1;
        outX = diff / layer.strideX + ( diff % layer.strideX != 0 );
    }

    #if DEBUG
        printf( DEBUG_PRESTRING "Input Size:  (%d, %d, %d)\n"
                DEBUG_PRESTRING "Output Size: (%d, %d, %d)\n"
                DEBUG_PRESTRING "\n", 
                inY, inX, inC, outY, outX, layer.outFilters );
    #endif

    // Calculate the output
    uint16_t i, j, k,   // Iterate over kernel.
             y, x,      // Iterate over output.
             yy, xx;    // Index input.
    double sum;         // Calculate convolution.

    uint16_t heightPadding = ( layer.kernelHeight - layer.strideY ) > 0 ? 
        ( layer.kernelHeight - layer.strideY ) : 0;
    uint16_t widthPadding = ( layer.kernelWidth - layer.strideX ) > 0 ? 
        ( layer.kernelWidth - layer.strideX ) : 0; 

    uint16_t topPadding = heightPadding / 2;
    // uint16_t bottomPadding = heightPadding - topPadding;
    uint16_t leftPadding = widthPadding / 2;
    // uint16_t rightPadding = widthPadding - leftPadding;


    for( k = 0; k < inC; k++ )
    {
        for( y = 0; y < outY; y += layer.strideY )
        {
            for( x = 0; x < outX; x += layer.strideX )
            {
                sum = 0.0;

                for( i = 0; i < layer.kernelHeight; i++ )
                {
                    for( j = 0; j < layer.kernelWidth; j++ )
                    {
                        yy = y + i - topPadding;
                        xx = x + j - leftPadding;

                        if( yy < 0 )
                            yy = 0;
                        else if( yy >= inY )
                            yy = inY - 1;

                        if( xx < 0 )
                            xx = 0;
                        else if( xx >= inX )
                            xx = inX - 1;

                        sum += input[ yy ][ xx ][ k ] *
                               layer.weights[ k ][ i ][ j ];
                    }
                }

                (*output)[ y ][ x ][ k ] = sum + layer.bias[ k ];
            }
        }
    }


    return 0;
}