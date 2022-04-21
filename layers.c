#include <stdlib.h>
#include <stdio.h>

#include "debug.h"
#include "layers.h"

int LAYERS_load_weights( double ***** weights, double ** bias, 
    uint16_t inC, uint16_t outC, 
    uint16_t filterHeight, uint16_t filterWidth,
    char * filename )
{
    FILE * weightsfp = fopen( filename, "r" );
    if( weightsfp == NULL )
    {
        return 1;
    }

    /****
        * To print weights from Python using Tensorflow, first store 
        * weights and bias in list 'weights'. Then, use the following to print:

for z in range( outC ):
  for k in range( inC ):
    for i in range( kernelHeight ):
      for j in range( kernelWidth ):
        print( f"{weights[ 0 ][ i ][ j ][ k ][ z ]}", end=' ' if j != kernelWidth - 1 else '\n' )
  print( f"{weights[ 1 ][ z ]}" )

    ****/ 

    // Allocate kernels for weights and biases for each output channel.
    (*weights) = (double****) malloc( outC * sizeof( double*** ) );
    (*bias) = (double*) malloc( outC * sizeof( double ) );
    for( uint32_t z = 0; z < outC; z++ )
    {
        // Allocate kernel weights for each input channel.
        (*weights)[ z ] = (double***) malloc( inC * sizeof( double** ) );
        for( uint32_t k = 0; k < inC; k++ )
        {
            // Allocate rows for each kernel.
            (*weights)[ z ][ k ] = (double**) malloc( filterHeight * sizeof( double* ) );
            for( uint16_t i = 0; i < filterHeight; i++ )
            {
                // Allocate values for each kernel.
                (*weights)[ z ][ k ][ i ] = (double*) malloc( filterWidth * sizeof( double ) );

                for( uint16_t j = 0; j < filterWidth; j++ )
                {
                    // Read in kernel weights for current input/output channel combination.
                    fscanf( weightsfp, "%lf", &(*weights)[ z ][ k ][ i ][ j ] );
                }
            }
        }

        // Read in bias value for current output channel.
        fscanf( weightsfp, "%lf", &(*bias)[ z ] );
    }

    fclose( weightsfp );

    return 0;
}

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
    int16_t i, j, k,   // Iterate over kernel.
            y, x, z,   // Iterate over output.
            yy, xx;    // Index input.
    double sum;         // Calculate convolution.


    // Calculate necessary padding (if needed).
    uint16_t heightPadding = layer.padded ? 
        ( layer.kernelHeight - layer.strideY ) > 0 ? 
            ( layer.kernelHeight - layer.strideY ) : 0 : 
        0;
    uint16_t widthPadding = layer.padded ?
        ( layer.kernelWidth - layer.strideX ) > 0 ? 
            ( layer.kernelWidth - layer.strideX ) : 0 :
        0;

    uint16_t topPadding = heightPadding / 2;
    // uint16_t bottomPadding = heightPadding - topPadding;
    uint16_t leftPadding = widthPadding / 2;
    // uint16_t rightPadding = widthPadding - leftPadding;

    for( z = 0; z < layer.outFilters; z++ ) // Output channels.
    {
        printf( "Output Channel %i:\n", z+1 );

        for( k = 0; k < inC; k++ )  // Input channels.
        {
            printf( "  Input Channel %i:\n", k+1 );

            for( i = 0; i < outY; i += layer.strideY )  // Output rows.
            {
                printf( "  " );
                for( j = 0; j < outX; j += layer.strideX )  // Output columns.
                {
                    sum = 0.0;

                    for( y = 0; y < layer.kernelHeight; y++ )   // Kernel rows.
                    {
                        
                        for( x = 0; x < layer.kernelWidth; x++ )    // Kernel columns.
                        {
                            // Calculate row/column position in input.
                            yy = i + y - topPadding;
                            xx = j + x - leftPadding;
                            
                            /**** The following is only valid if padding
                                * is defined as expanding the input.
                            ****/
                            /*
                            // Check if padding is needed.
                            if( yy < 0 )
                                yy = 0;
                            else if( yy >= inY )
                                yy = inY - 1;

                            if( xx < 0 )
                                xx = 0;
                            else if( xx >= inX )
                                xx = inX - 1;


                            // Calculate value.
                            sum += input[ k ][ yy ][ xx ] *
                                   layer.weights[ z ][ k ][ y ][ x ];
                            */

                            /**** Otherwise, zero padding is used
                            ****/ 
                            // Check if padding is needed.
                            if( yy < 0 || yy >= inY ||
                                xx < 0 || xx >= inX )
                            {
                                sum += 0;
                            }
                            else
                            {
                                // Calculate value.
                                sum += input[ k ][ yy ][ xx ] *
                                       layer.weights[ z ][ k ][ y ][ x ];
                            }
                        }
                    }

                    printf( "  % 5.2f", sum );
                    (*output)[ z ][ i ][ j ] += sum;
                }

                printf( "\n" );
            }
        }

        printf( "\n" );
    }


    for( z = 0; z < layer.outFilters; z++ )
        for( y = 0; y < outY; y++ )
            for( x = 0; x < outX; x++ )
                (*output)[ z ][ y ][ x ] += layer.bias[ z ];


    return 0;
}