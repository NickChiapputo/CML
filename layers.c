#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "debug.h"
#include "layers.h"


int LAYERS_conv2d_constructor( LAYERS_Conv2D * layer, 
    uint16_t inFilters, uint16_t outFilters,
    uint16_t kernelWidth, uint16_t kernelHeight,
    uint16_t strideX, uint16_t strideY,
    uint16_t padded, 
    char * weightsFile,
    int (*forward)(DATATYPE****, DATATYPE***, uint16_t, uint16_t, uint16_t, LAYERS_Conv2D) )
{
    // TODO: Check for input validity.


    // Set the input and output filter numbers.
    (*layer).inFilters = inFilters;
    (*layer).outFilters = outFilters;

    // Set the kernel size.
    (*layer).kernelWidth = kernelWidth;
    (*layer).kernelHeight = kernelHeight;

    // Set the kernel stride.
    (*layer).strideX = strideX;
    (*layer).strideY = strideY;

    // Set flag to determine if input is padded or not.
    // If zero, no padding. 
    // If one, padding is used (output is same size as input).
    (*layer).padded = padded;

    // If weights file is provided, load in weights.
    if( weightsFile != NULL )
    {
        // Intialize the kernel weights and biases.
        LAYERS_load_weights( &(*layer).weights, &(*layer).bias,
            inFilters, outFilters,
            kernelHeight, kernelWidth,
            weightsFile );
    }

    // Define the forward function.
    (*layer).forward = forward;

    return 0;
}


int LAYERS_prelu_constructor( LAYERS_PreLU * layer,
    uint16_t outFilters, char * weightsFile,
    int (*forward)(DATATYPE****, DATATYPE***, uint16_t, uint16_t, LAYERS_PreLU) )
{
    // TODO: Check for input validity.

    
    // Set the output filter numbers.
    (*layer).outFilters = outFilters;

    // If weights file is provided, load in weights.
    if( weightsFile != NULL )
    {
        // Intialize the ReLU weights.
        LAYERS_prelu_load_weights( &(*layer).weights, outFilters, weightsFile );
    }    

    // Define the forward function.
    (*layer).forward = forward;

    return 0;
}


int LAYERS_relu_constructor( LAYERS_ReLU * layer, uint16_t outFilters,
    int (*forward)(DATATYPE****, DATATYPE***, uint16_t, uint16_t, LAYERS_ReLU) )
{
    // TODO: Check for input validity.

    
    // Set the output filter numbers.
    (*layer).outFilters = outFilters;  

    // Define the forward function.
    (*layer).forward = forward;

    return 0;
}


int LAYERS_load_weights( DATATYPE ***** weights, DATATYPE ** bias, 
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

        layer = model.model.layers[ 1 ]

        with open( f"{save_location}{first_conv_layer.name}", "wb" ) as f:
            # Original shape: kernelHeight x kernelWidth x inputChannels x outputChannels
            # New shape:      kernelHeight x outputChannels x inputChannels x kernelWidth
            # This is done so that when the structs are written to the file, it does
            # so across each row (e.g., all columns in first row, all columns in 
            # second row, and so on). Prints floats in little-endian 
            # IEEE.754 32-bit format.
            weights = np.swapaxes( layer[ 0 ], 1, 3 )

            for z in range( outC ):
                for k in range( inC ):
                    for i in range( kernelHeight ):
                        f.write( struct.pack( 'f' * len( weights[ i ][ z ][ k ] ), 
                            *weights[ i ][ z ][ k ] ) )

                # Write biases to file.
                f.write( struct.pack( 'f' * len( np.array( [layer[ 1 ][ z ]] ) ), 
                    *np.array( [layer[ 1 ][ z ]] ) ) )

    ****/ 

    // Allocate kernels for weights and biases for each output channel.
    (*weights) = (DATATYPE****) malloc( outC * sizeof( DATATYPE*** ) );
    (*bias) = (DATATYPE*) malloc( outC * sizeof( DATATYPE ) );
    float val;
    for( uint32_t z = 0; z < outC; z++ )
    {
        // Allocate kernel weights for each input channel.
        (*weights)[ z ] = (DATATYPE***) malloc( inC * sizeof( DATATYPE** ) );
        for( uint32_t k = 0; k < inC; k++ )
        {
            // Allocate rows for each kernel.
            (*weights)[ z ][ k ] = (DATATYPE**) malloc( filterHeight * sizeof( DATATYPE* ) );
            for( uint16_t i = 0; i < filterHeight; i++ )
            {
                // Allocate values for each kernel.
                (*weights)[ z ][ k ][ i ] = (DATATYPE*) malloc( filterWidth * sizeof( DATATYPE ) );

                for( uint16_t j = 0; j < filterWidth; j++ )
                {
                    // Read in kernel weights for current input/output channel 
                    // combination. Weights are stored in 32-bit floating point
                    // little-endian format. 
                    fread( (void*) (&val), sizeof( val ), 1, weightsfp );
                    (*weights)[ z ][ k ][ i ][ j ] = (DATATYPE)val;
                }
            }
        }

        // Read in bias value for current output channel.
        fread( (void*) (&val), sizeof( val ), 1, weightsfp );
        (*bias)[ z ] = val;
    }

    fclose( weightsfp );

    return 0;
}


int LAYERS_prelu_load_weights( DATATYPE ** weights, uint16_t outC, 
    char * filename )
{
    FILE * weightsfp = fopen( filename, "r" );
    if( weightsfp == NULL )
    {
        return 1;
    }

    (*weights) = (DATATYPE*) malloc( outC * sizeof( DATATYPE ) );
    float val;
    for( uint16_t z = 0; z < outC; z++ )
    {
        fread( (void*) (&val), sizeof( val ), 1, weightsfp );
        (*weights)[ z ] = (DATATYPE)val;
    }

    fclose( weightsfp );

    return 0;
}


int LAYERS_conv2d_free_weights( LAYERS_Conv2D * layer )
{
    for( uint16_t z = 0; z < (*layer).outFilters; z++ )
    {
        for( uint16_t k = 0; k < (*layer).inFilters; k++ )
        {
            for( uint16_t y = 0; y < (*layer).kernelHeight; y++ )
            {
                free( (*layer).weights[ z ][ k ][ y ] );
            }
            free( (*layer).weights[ z ][ k ] );
        }
        free( (*layer).weights[ z ] );
    }

    free( (*layer).weights );
    free( (*layer).bias );

    return 0;
}


int LAYERS_prelu_free_weights( LAYERS_PreLU * layer )
{
    free( (*layer).weights );

    return 0;
}

int LAYERS_convolution_2d( DATATYPE **** output, DATATYPE *** input, uint16_t inY,
                           uint16_t inX, uint16_t inC, LAYERS_Conv2D layer )
{
    // TODO: Convert data to flattened array. Check if time saved due to
    // reducing address calculation.

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
    int16_t i, j, k, a, b, // Iterate over kernel.
            y, x, z,       // Iterate over output.
            yy, xx;        // Index input.
    DATATYPE sum;          // Calculate convolution.


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


    for( z = 0; z < layer.outFilters; z++ )
        for( y = 0; y < outY; y++ )
            for( x = 0; x < outX; x++ )
                (*output)[ z ][ y ][ x ] = layer.bias[ z ];

    // TODO: Separate convolution from completely inside image and convolution
    // that requires padding.

    for( z = 0; z < layer.outFilters; z++ ) // Output channels.
    {
        #if DEBUG
            printf( "Output Channel %i:\n", z+1 );
        #endif

        float *** weightOutputFilter = layer.weights[ z ];
        float ** outputFilter = (*output)[ z ];
        for( k = 0; k < inC; k++ )  // Input channels.
        {
            #if DEBUG
                printf( "  Input Channel %i:\n", k+1 );
            #endif

            float ** inputFilter = input[ k ];
            float ** weightInputFilter = weightOutputFilter[ k ];
            for( i = 0, a = 0; i < outY; i += layer.strideY, a++ )  // Output rows.
            {
                float * outputRow = outputFilter[ a ];

                #if DEBUG
                    printf( "  " );
                #endif
                for( j = 0, b = 0; j < outX; j += layer.strideX, b++ )  // Output columns.
                {
                    sum = 0.0;

                    for( y = 0; y < layer.kernelHeight; y++ )   // Kernel rows.
                    {
                        // Reduce calculations for each pixel.
                        yy = i + y - topPadding;

                        // If zero padding is used, we don't need to check any
                        // values in this row.
                        if( yy < 0 || yy >= inY )
                            continue;

                        float * weightRow = weightInputFilter[ y ];
                        float * inputRow = inputFilter[ yy ];

                        for( x = 0; x < layer.kernelWidth; x++ )    // Kernel columns.
                        {
                            // Calculate row/column position in input.
                            // yy = in_y_center;
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
                            if( xx < 0 || xx >= inX )
                            {
                                // sum += 0
                                continue;
                            }
                            else
                            {
                                // Calculate value.
                                sum += inputRow[ xx ] *
                                       weightRow[ x ];
                            }
                        }
                    }

                    #if DEBUG
                        printf( "  % 5.2f", sum );
                    #endif
                    // a = i / layer.strideY
                    // b = j / layer.strideX
                    outputRow[ b ] += sum;
                }

                #if DEBUG
                    printf( "\n" );
                #endif
            }
        }

        #if DEBUG
            printf( "\n" );
        #endif
    }

    return 0;
}


int LAYERS_prelu_forward( DATATYPE **** output, DATATYPE *** input, 
                          uint16_t inY, uint16_t inX, LAYERS_PreLU layer )
{
    for( uint16_t z = 0; z < layer.outFilters; z++ )
    {
        for( uint16_t y = 0; y < inY; y++ )
        {
            for( uint16_t x = 0; x < inX; x++ )
            {
                (*output)[ z ][ y ][ x ] = input[ z ][ y ][ x ] > 0 ? 
                    input[ z ][ y ][ x ] : 
                    input[ z ][ y ][ x ] * layer.weights[ z ];
            }
        }
    }

    return 0;
}


int LAYERS_relu_forward( DATATYPE **** output, DATATYPE *** input, 
                          uint16_t inY, uint16_t inX, LAYERS_ReLU layer )
{
    for( uint16_t z = 0; z < layer.outFilters; z++ )
    {
        for( uint16_t y = 0; y < inY; y++ )
        {
            for( uint16_t x = 0; x < inX; x++ )
            {
                if( input[ z ][ y ][ x ] < 0 )
                    (*output)[ z ][ y ][ x ] = 0;
            }
        }
    }

    return 0;
}
