#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "debug.h"

#include "activation.h"
#include "colors.h"
#include "mnist.h"
#include "loss.h"
#include "layers.h"


#define TRAIN_DATA_PATH     "./data/mnist/t10k-images.idx3-ubyte"
#define TRAIN_LABEL_PATH    "./data/mnist/t10k-labels.idx1-ubyte"
#define TEST_DATA_PATH      "./data/mnist/train-images.idx3-ubyte"
#define TEST_LABELS_PATH    "./data/mnist/train-labels.idx1-ubyte"

#define BUFFER_LENGTH 64

// Output formatting.
#define ERROR   B_RED "ERROR: " RESET     // Error message color.
// #define LINE_SEPARATOR B_BLUE "============================================================\n" RESET
#define LINE_SEPARATOR "\n"


// Global function definitions.
int testDataParsing();
int testCrossEntropyLoss();
int testSoftmaxActivation();
int test2DConvolution();
int testLoadWeights();


// Global variable definitions.
LAYERS_activation FC_SOFTMAX_LAYER;


int main( int argc, char ** argv )
{
    /**** Test MNIST Data Parsing ****/ 
    /*printf( "[" B_GREEN "MNIST Data Parsing" RESET "]\n" );
    
    if( testDataParsing() )
        return 1;

    printf( LINE_SEPARATOR );*/
    /*********************************/

 
    /**** Test Cross Entropy Loss ****/
    /*printf( "[" B_GREEN "Test Cross Entropy Loss" RESET "]\n" );
    
    int numClasses = 5, numPredictions = 1;
    if( testCrossEntropyLoss( numClasses, numPredictions ) ) return 1;

    printf( LINE_SEPARATOR );*/
    /*********************************/


    /**** Test Softmax Activation Function ****/
    /*printf( "[" B_GREEN "Test Softmax Activation" RESET "]\n" );
    numClasses = 4;
    double * input = (double*) malloc( numClasses * sizeof( double ) );
    double * output = (double*) malloc( numClasses * sizeof( double ) );
    input[ 0 ] = -1; input[ 1 ] = 0; input[ 2 ] = 3; input[ 3 ] = 5;
    FC_SOFTMAX_LAYER.activation = ACTIVATION_softmax;

    if( testSoftmaxActivation( input, numClasses, output ) ) return 1;

    free( input );
    free( output );

    printf( LINE_SEPARATOR );*/
    /******************************************/


    /**** Test 2D Convolutional Layer ****/
    printf( "[" B_GREEN "Test 2D Convolution" RESET "]\n" );
    if( test2DConvolution() ) return 1;

    printf( LINE_SEPARATOR );
    /******************************************/


    /**** Test Loading Convolutional Weights and Biases from File ****/
    /*printf( "[" B_GREEN "Test Weights Loading from File" RESET "]\n" );

    if( testLoadWeights() ) return 1;

    printf( LINE_SEPARATOR );*/
    /******************************************/

    return 0;
}


int testDataParsing()
{
    uint32_t magicNumber = 0, numImages = 0, rows = 0, cols = 0;

    if( MNIST_parseHeader( TRAIN_DATA_PATH, &magicNumber, &numImages, &rows, &cols, 1 ) )
    {
        printf( ERROR "Unable to parse training data header.\n" );
        return 1;
    }

    printf( "TRAIN DATA SET:\n"
            "   Magic Number:      %d\n"
            "   Number of Images:  %d\n"
            "   Number of Rows:    %d\n"
            "   Number of Columns: %d\n\n", magicNumber, numImages, rows, cols ); 

    magicNumber = 0, numImages = 0;
    if( MNIST_parseHeader( TRAIN_LABEL_PATH, &magicNumber, &numImages, NULL, NULL, 0 ) )
    {
        printf( ERROR "Unable to parse training data header.\n" );
        return 1;
    }

    printf( "TRAIN DATA LABELS:\n"
            "   Magic Number:      %d\n"
            "   Number of Images:  %d\n\n", magicNumber, numImages ); 

    magicNumber = 0, numImages = 0, rows = 0, cols = 0;
    if( MNIST_parseHeader( TEST_DATA_PATH, &magicNumber, &numImages, &rows, &cols, 1 ) )
    {
        printf( ERROR "Unable to parse training data header.\n" );
        return 1;
    }

    printf( "TEST DATA SET:\n"
            "   Magic Number:      %d\n"
            "   Number of Images:  %d\n"
            "   Number of Rows:    %d\n"
            "   Number of Columns: %d\n\n", magicNumber, numImages, rows, cols ); 

    magicNumber = 0, numImages = 0;
    if( MNIST_parseHeader( TEST_LABELS_PATH, &magicNumber, &numImages, NULL, NULL, 0 ) )
    {
        printf( ERROR "Unable to parse training data header.\n" );
        return 1;
    }

    printf( "TEST DATA LABELS:\n"
            "   Magic Number:      %d\n"
            "   Number of Images:  %d\n\n", magicNumber, numImages );


    uint8_t ** data;
    uint8_t * labels;
    int numRead = 1;

    data = (uint8_t**) malloc( numRead * sizeof( uint8_t* ) );
    labels = (uint8_t*) malloc( numRead * sizeof( uint8_t ) );
    for( int i = 0; i < numRead; i++ )
    {
        data[ i ] = (uint8_t*) calloc( 784, sizeof( uint8_t ) );
    }

    if( MNIST_readDataFile( TRAIN_DATA_PATH, TRAIN_LABEL_PATH, data, labels, numRead ) )
    {
        printf( ERROR "Unable to read %d images from train dataset.\n", numRead );
        return 1;
    }


    for( int i = 0; i < numRead; i++ )
    {
        printf( "\nData %d: %d\n", i + 1, labels[ i ] );
        for( int j = 0; j < 784; j++ )
            printf( "%s%s", data[ i ][ j ] ? "00" : "--", ( j + 1 ) % 28 ? " " : "\n" );
        free( data[ i ] );
    }
    free( data );
    free( labels );

    return 0;
}


int testCrossEntropyLoss( int numClasses, int numPredictions )
{
    double ** predictions = (double**) malloc( numPredictions * sizeof( double* ) );
    uint8_t ** target = (uint8_t**) malloc( numPredictions * sizeof( uint8_t* ) );
    for( int i = 0; i < numPredictions; i++ )
    {
        predictions[ i ] = (double*) malloc( numClasses * sizeof( double ) );
        target[ i ] = (uint8_t*) malloc( numClasses * sizeof( uint8_t ) );


        predictions[ 0 ][ 0 ] = 0.1;
        predictions[ 0 ][ 1 ] = 0.5;
        predictions[ 0 ][ 2 ] = 0.1;
        predictions[ 0 ][ 3 ] = 0.1;
        predictions[ 0 ][ 4 ] = 0.2;

        target[ 0 ][ 0 ] = 1;
        target[ 0 ][ 1 ] = 0;
        target[ 0 ][ 2 ] = 0;
        target[ 0 ][ 3 ] = 0;
        target[ 0 ][ 4 ] = 0;
    }

    printf( "Cross Entropy Loss = %f\n", LOSS_crossEntropy( predictions, target, numPredictions, numClasses ) );
    
    for( int i = 0; i < numPredictions; i++ )
    {
        free( predictions[ i ] );
        free( target[ i ] );
    }
    free( predictions );
    free( target );

    return 0;
}


int testSoftmaxActivation( double * input, int numClasses, double * output )
{
    FC_SOFTMAX_LAYER.activation( input, numClasses, output );
    // ACTIVATION_softmax( input, numClasses, output );
    printf( "Softmax Activation:\n" );
    for( int i = 0; i < numClasses; i++ )
        printf( "    %lf\n", output[ i ] );

    return 0;
}


int test2DConvolution()
{
    // Create the 2D convolution layer.
    LAYERS_Conv2D layer;

    // Set the input and output filter numbers.
    layer.inFilters = 2;
    layer.outFilters = 2;

    // Set the kernel stride.
    layer.strideX = 1;
    layer.strideY = 1;

    // Set the kernel size.
    layer.kernelWidth = 3;
    layer.kernelHeight = 3;

    // Set flag to determine if input is padded or not.
    // If zero, no padding. 
    // If one, padding is used (output is same size as input).
    layer.padded = 1;

    // Intialize the kernel weights and biases.
    uint16_t i, j, k, 
             x, y, z;
    LAYERS_load_weights( &layer.weights, &layer.bias, 
        layer.inFilters, layer.outFilters, 
        layer.kernelHeight, layer.kernelWidth,
        "test_conv_weights.txt" );
    /*
    layer.weights = (double***) malloc( layer.outFilters * sizeof( double** ) );
    for( i = 0; i < layer.outFilters; i++ )
    {
        layer.weights[ i ] = (double**) malloc( layer.kernelHeight * sizeof( double* ) );

        for( j = 0; j < layer.kernelHeight; j++ )
        {
            layer.weights[ i ][ j ] = (double*) malloc( layer.kernelWidth * sizeof( double ) );

            layer.weights[ i ][ j ][ 0 ] = 0.0;
            layer.weights[ i ][ j ][ 1 ] = 1.0;
            layer.weights[ i ][ j ][ 2 ] = 0.0;
        }
    }

    layer.bias = (double*) malloc( layer.outFilters * sizeof( double ) );
    for( i = 0; i < layer.outFilters; i++ )
    {
        layer.bias[ i ] = 0.0;
    }
    */

    // Define the forward function.
    layer.forward = (*LAYERS_convolution_2d);

    // Print out the weights and biases.
    #if DEBUG
        for( k = 0; k < layer.outFilters; k++ )
        {
            printf( "Output %i Filters:\n", k+1 );
            for( z = 0; z < layer.inFilters; z++ )
            {
                printf( "  Input %i Filters:\n", z+1 );

                for( i = 0; i < layer.kernelHeight; i++ )
                {
                    printf( "    " );
                    for( j = 0; j < layer.kernelWidth; j++ )
                    {
                        printf( "% 5.2f ", layer.weights[ k ][ z ][ i ][ j ] );
                    }
                    printf( "\n" DEBUG_PRESTRING );
                }
                printf( "\n" );
            }

            printf( "  Bias: % 5.2f\n" DEBUG_PRESTRING "\n", layer.bias[ k ] );
        }
    #endif


    /**** Define and allocate the test input.
        *   Input 1:
        *     0, 0, 0, 1, 1, 0, 0, 0
        *     0, 0, 0, 1, 1, 0, 0, 0
        *     0, 0, 0, 1, 1, 0, 0, 0
        *     0, 0, 0, 1, 1, 0, 0, 0
        *     0, 0, 0, 1, 1, 0, 0, 0
        *     0, 0, 0, 1, 1, 0, 0, 0
        *     0, 0, 0, 1, 1, 0, 0, 0
        *     0, 0, 0, 1, 1, 0, 0, 0
        *   Input 2:
        *     0, 0, 0, 0, 0, 0, 0, 0
        *     0, 0, 0, 0, 0, 0, 0, 0
        *     0, 0, 0, 0, 0, 0, 0, 0
        *     1, 1, 1, 1, 1, 1, 1, 1
        *     1, 1, 1, 1, 1, 1, 1, 1
        *     0, 0, 0, 0, 0, 0, 0, 0
        *     0, 0, 0, 0, 0, 0, 0, 0
        *     0, 0, 0, 0, 0, 0, 0, 0
    ****/
    uint16_t inY = 8, inX = 8;
    uint16_t outY = 8, outX = 8;
    double *** input = (double***) malloc( layer.inFilters * sizeof( double** ) );
    for( k = 0; k < layer.inFilters; k++ )
    {
        input[ k ] = (double**) malloc( inY * sizeof( double* ) );
        for( i = 0; i < inY; i++ )
        {
            input[ k ][ i ] = (double*) malloc( inX * sizeof( double ) );
            for( j = 0; j < inX; j++ )
            {
                if( k == 0 )
                {
                    // Input 1
                    if( j == 3 || j == 4 )
                        input[ k ][ i ][ j ] = 1;   // Middle columns.
                    else
                        input[ k ][ i ][ j ] = 0;
                }
                else
                {
                    // Input 2
                    if( i == 3 || i == 4 )
                        input[ k ][ i ][ j ] = 1;   // Middle rows.
                    else
                        input[ k ][ i ][ j ] = 0;
                }
            }
        }
    }


    #if DEBUG
        for( z = 0; z < layer.inFilters; z++ )
        {
            printf( DEBUG_PRESTRING "Input Channel %d:\n" DEBUG_PRESTRING, z+1 );
            for( y = 0; y < inY; y++ )
            {
                for( x = 0; x < inX; x++ )
                {
                    printf( "  % 5.2f ", input[ z ][ y ][ x ] );
                }
                printf( "\n" DEBUG_PRESTRING );
            }
            printf( "\n" );
        }
    #endif

    // Allocate output.
    double *** output = (double***) malloc( layer.outFilters * sizeof( double** ) );
    for( k = 0; k < layer.outFilters; k++ )
    {
        output[ k ] = (double**) malloc( outY * sizeof( double* ) );
        for( i = 0; i < outY; i++ )
        {
            output[ k ][ i ] = (double*) malloc( outX * sizeof( double ) );
            for( j = 0; j < outX; j++ )
            {
                output[ k ][ i ][ j ] = 0;
            }
        }
    }

    // Call the forward function.
    layer.forward( &output, input, inY, inX, layer.inFilters, layer );

    #if DEBUG
        for( z = 0; z < layer.outFilters; z++ )
        {
            printf( DEBUG_PRESTRING "Output Channel: %d\n" DEBUG_PRESTRING, z+1 );
            for( y = 0; y < outY; y++ )
            {
                for( x = 0; x < outX; x++ )
                {
                    printf( "  % 5.2f", output[ z ][ y ][ x ] );
                }
                printf( "\n" DEBUG_PRESTRING );
            }
            printf( "\n" );
        }
    #endif

    // Free output, input, weights, and biases.
    for( z = 0; z < layer.outFilters; z++ )
    {
        for( y = 0; y < outY; y++ )
        {
            free( output[ z ][ y ] );
        }
        free( output[ z ] );
    }

    for( k = 0; k < layer.inFilters; k++ )
    {
        for( i = 0; i < inY; i++ )
        {
            free( input[ k ][ i ] );
        }
        free( input[ k ] );
    }

    for( z = 0; z < layer.outFilters; z++ )
    {
        for( k = 0; k < layer.inFilters; k++ )
        {
            for( i = 0; i < layer.kernelHeight; i++ )
            {
                free( layer.weights[ z ][ k ][ i ] );
            }
            free( layer.weights[ z ][ k ] );
        }
        free( layer.weights[ z ] );
    }

    free( input );
    free( output );
    free( layer.weights );
    free( layer.bias );

    return 0;
}


int testLoadWeights()
{
    char * filename = "test_weights.txt";

    // Open weights file.
    FILE * weightsfp = fopen( filename, "r" );
    if( weightsfp == NULL )
    {
        printf( ERROR "Can't open file." );
        return 1;
    }

    // Parse weights file.
    uint16_t x = 3, y = 3, c = 2;
    double *** weights; // Hold kernel weights.
    double * bias;      // Hold filter biases.


    weights = (double***) malloc( c * sizeof( double** ) );
    bias = (double*) malloc( c * sizeof( double ) );
    for( uint16_t k = 0; k < c; k++ )
    {
        weights[ k ] = (double**) malloc( y * sizeof( double* ) );
        printf( "Filter %d:\n", k+1 );
        for( uint16_t i = 0; i < y; i++ )
        {
            weights[ k ][ i ] = (double*) malloc( x * sizeof( double ) );
            for( uint16_t j = 0; j < x; j++ )
            {
                fscanf( weightsfp, "%lf", &weights[ k ][ i ][ j ] );
                printf( "  % 5.2f", weights[ k ][ i ][ j ] );
            }
            printf( "\n" );
        }
        fscanf( weightsfp, "%lf", &bias[ k ] );
        printf( "  Bias: % 5.2f\n\n", bias[ k ] );
    }

    fclose( weightsfp );

    return 0;
}
