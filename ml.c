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
#define LINE_SEPARATOR B_BLUE "============================================================\n" RESET


// Global function definitions.
int testDataParsing();
int testCrossEntropyLoss();
int testSoftmaxActivation();
int test2DConvolution();


// Global variable definitions.
LAYERS_activation FC_SOFTMAX_LAYER;


int main( int argc, char ** argv )
{
    /**** Test MNIST Data Parsing ****/ 
    printf( "[" B_GREEN "MNIST Data Parsing" RESET "]\n" );
    
    if( testDataParsing() )
        return 1;

    printf( LINE_SEPARATOR );
    /*********************************/

 
    /**** Test Cross Entropy Loss ****/
    printf( "[" B_GREEN "Test Cross Entropy Loss" RESET "]\n" );
    
    int numClasses = 5, numPredictions = 1;
    if( testCrossEntropyLoss( numClasses, numPredictions ) ) return 1;

    printf( LINE_SEPARATOR );
    /*********************************/


    /**** Test Softmax Activation Function ****/
    printf( "[" B_GREEN "Test Softmax Activation" RESET "]\n" );
    numClasses = 4;
    double * input = (double*) malloc( numClasses * sizeof( double ) );
    double * output = (double*) malloc( numClasses * sizeof( double ) );
    input[ 0 ] = -1; input[ 1 ] = 0; input[ 2 ] = 3; input[ 3 ] = 5;
    FC_SOFTMAX_LAYER.activation = ACTIVATION_softmax;

    if( testSoftmaxActivation( input, numClasses, output ) ) return 1;

    free( input );
    free( output );

    printf( LINE_SEPARATOR );
    /******************************************/


    /**** Test 2D Convolutional Layer ****/
    printf( "[" B_GREEN "Test 2D Convolution" RESET "]\n" );
    if( test2DConvolution() ) return 1;

    printf( LINE_SEPARATOR );
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
    layer.inFilters = 1;
    layer.outFilters = 1;

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
    uint16_t i, j, k;
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

    // Define the forward function.
    layer.forward = (*LAYERS_convolution_2d);

    // Print out the weights and biases.
    #if DEBUG
        for( i = 0; i < layer.outFilters; i++ )
        {
            printf( DEBUG_PRESTRING "Filter %i:\n" DEBUG_PRESTRING, i+1 );

            for( j = 0; j < layer.kernelHeight; j++ )
            {
                printf( "  " );
                for( k = 0; k < layer.kernelWidth; k++ )
                {
                    printf( "% 5.2f ", layer.weights[ i ][ j ][ k ] );
                }
                printf( "\n" DEBUG_PRESTRING );
            }
            printf( "  Bias: % 5.2f\n\n", layer.bias[ i ] );
        }
    #endif


    /**** Define the test input.
        *   0, 0, 0, 1, 1, 0, 0, 0
        *   0, 0, 0, 1, 1, 0, 0, 0
        *   0, 0, 0, 1, 1, 0, 0, 0
        *   0, 0, 0, 1, 1, 0, 0, 0
        *   0, 0, 0, 1, 1, 0, 0, 0
        *   0, 0, 0, 1, 1, 0, 0, 0
        *   0, 0, 0, 1, 1, 0, 0, 0
        *   0, 0, 0, 1, 1, 0, 0, 0
    ****/
    uint16_t inY = 8, inX = 8, inC = 1;
    uint16_t outY = 8, outX = 8, outC = 1;
    double *** input = (double***) malloc( inY * sizeof( double** ) );
    for( i = 0; i < inY; i++ )
    {
        input[ i ] = (double**) malloc( inX * sizeof( double* ) );
        for( j = 0; j < inX; j++ )
        {
            input[ i ][ j ] = (double*) malloc( inC * sizeof( double ) );
            for( k = 0; k < inC; k++ )
            {
                if( j == 3 || j == 4 )
                    input[ i ][ j ][ k ] = 1;
                else
                    input[ i ][ j ][ k ] = 0;
            }
        }
    }


    #if DEBUG
        for( k = 0; k < inC; k++ )
        {
            printf( DEBUG_PRESTRING "Input Channel %d:\n" DEBUG_PRESTRING, k+1 );
            for( i = 0; i < inY; i++ )
            {
                for( j = 0; j < inX; j++ )
                {
                    printf( "  % 5.2f ", input[ i ][ j ][ k ] );
                }
                printf( "\n" DEBUG_PRESTRING );
            }
            printf( "\n" );
        }
    #endif

    // Allocate output.
    double *** output = (double***) malloc( outY * sizeof( double** ) );
    for( i = 0; i < outY; i++ )
    {
        output[ i ] = (double**) malloc( outX * sizeof( double* ) );
        for( j = 0; j < outX; j++ )
        {
            output[ i ][ j ] = (double*) malloc( outC * sizeof( double ) );
            for( k = 0; k < outC; k++ )
            {
                output[ i ][ j ][ k ] = 0;
            }
        }
    }

    // Call the forward function.
    layer.forward( &output, input, inY, inX, layer.inFilters, layer );

    #if DEBUG
        for( k = 0; k < layer.outFilters; k++ )
        {
            printf( DEBUG_PRESTRING "Output Channel: %d\n" DEBUG_PRESTRING, k+1 );
            for( i = 0; i < outY; i++ )
            {
                for( j = 0; j < outX; j++ )
                {
                    printf( "  % 5.2f", output[ i ][ j ][ k ] );
                }
                printf( "\n" DEBUG_PRESTRING );
            }
            printf( "\n" );
        }
    #endif

    return 0;
}

