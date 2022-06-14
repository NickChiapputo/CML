#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

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
int testTinyPSSR();


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
    DATATYPE * input = (DATATYPE*) malloc( numClasses * sizeof( DATATYPE ) );
    DATATYPE * output = (DATATYPE*) malloc( numClasses * sizeof( DATATYPE ) );
    input[ 0 ] = -1; input[ 1 ] = 0; input[ 2 ] = 3; input[ 3 ] = 5;
    FC_SOFTMAX_LAYER.activation = ACTIVATION_softmax;

    if( testSoftmaxActivation( input, numClasses, output ) ) return 1;

    free( input );
    free( output );

    printf( LINE_SEPARATOR );*/
    /******************************************/


    /**** Test 2D Convolutional Layer ****/
    /*printf( "[" B_GREEN "Test 2D Convolution" RESET "]\n" );
    if( test2DConvolution() ) return 1;

    printf( LINE_SEPARATOR );*/
    /******************************************/


    /**** Test Loading Convolutional Weights and Biases from File ****/
    /*printf( "[" B_GREEN "Test Weights Loading from File" RESET "]\n" );

    if( testLoadWeights() ) return 1;

    printf( LINE_SEPARATOR );*/
    /******************************************/


    /**** Test TinyPSSR ****/
    uint8_t iterations = 10;
    uint16_t i = 0;
    for( i = 0; i < iterations; i++ )
    {
        printf( "[" B_GREEN "Test TinyPSSR" RESET "]\n" );
        if( testTinyPSSR() ) return 1;

        printf( LINE_SEPARATOR );
    }
    /******************************************/

    return 0;
}


/*int testDataParsing()
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
    DATATYPE ** predictions = (DATATYPE**) malloc( numPredictions * sizeof( DATATYPE* ) );
    uint8_t ** target = (uint8_t**) malloc( numPredictions * sizeof( uint8_t* ) );
    for( int i = 0; i < numPredictions; i++ )
    {
        predictions[ i ] = (DATATYPE*) malloc( numClasses * sizeof( DATATYPE ) );
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


int testSoftmaxActivation( DATATYPE * input, int numClasses, DATATYPE * output )
{
    FC_SOFTMAX_LAYER.activation( input, numClasses, output );
    // ACTIVATION_softmax( input, numClasses, output );
    printf( "Softmax Activation:\n" );
    for( int i = 0; i < numClasses; i++ )
        printf( "    %lf\n", output[ i ] );

    return 0;
}*/


int test2DConvolution()
{
    // Create the 2D convolution layer.
    LAYERS_Conv2D layer;

    // Set the input and output filter numbers.
    layer.inFilters = 1;
    layer.outFilters = 16;

    // Set the kernel stride.
    layer.strideX = 1;
    layer.strideY = 1;

    // Set the kernel size.
    layer.kernelWidth = 5;
    layer.kernelHeight = 5;

    // Set flag to determine if input is padded or not.
    // If zero, no padding. 
    // If one, padding is used (output is same size as input).
    layer.padded = 1;

    // Intialize the kernel weights and biases.
    uint16_t i, j, k, 
             y, z;
    LAYERS_load_weights( &layer.weights, &layer.bias, 
        layer.inFilters, layer.outFilters, 
        layer.kernelHeight, layer.kernelWidth,
        "models/1356-rgb/weights/conv2d.txt" );

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
    FILE * inF = fopen( TRAIN_DATA_PATH, "rb" );
    fseek( inF, 16L, SEEK_SET );
    uint16_t inY = 28, inX = 28;
    uint16_t outY = 28, outX = 28;
    DATATYPE *** input = (DATATYPE***) malloc( layer.inFilters * sizeof( DATATYPE** ) );
    for( k = 0; k < layer.inFilters; k++ )
    {
        input[ k ] = (DATATYPE**) malloc( inY * sizeof( DATATYPE* ) );
        for( i = 0; i < inY; i++ )
        {
            input[ k ][ i ] = (DATATYPE*) malloc( inX * sizeof( DATATYPE ) );
            for( j = 0; j < inX; j++ )
            {
                input[ k ][ i ][ j ] = getc( inF ) / 255.0;
            }
        }
    }


    #if DEBUG
        for( k = 0; k < layer.inFilters; k++ )
        {
            printf( DEBUG_PRESTRING "Input Channel %d:\n" DEBUG_PRESTRING, k+1 );
            for( i = 0; i < inY; i++ )
            {
                for( j = 0; j < inX; j++ )
                {
                    printf( "  % 5.2f ", input[ k ][ i ][ j ] );
                }
                printf( "\n" DEBUG_PRESTRING );
            }
            printf( "\n" );
        }
    #endif

    // Allocate output.
    DATATYPE *** output = (DATATYPE***) malloc( layer.outFilters * sizeof( DATATYPE** ) );
    for( k = 0; k < layer.outFilters; k++ )
    {
        output[ k ] = (DATATYPE**) malloc( outY * sizeof( DATATYPE* ) );
        for( i = 0; i < outY; i++ )
        {
            output[ k ][ i ] = (DATATYPE*) malloc( outX * sizeof( DATATYPE ) );
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


/*int testLoadWeights()
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
    DATATYPE *** weights; // Hold kernel weights.
    DATATYPE * bias;      // Hold filter biases.


    weights = (DATATYPE***) malloc( c * sizeof( DATATYPE** ) );
    bias = (DATATYPE*) malloc( c * sizeof( DATATYPE ) );
    for( uint16_t k = 0; k < c; k++ )
    {
        weights[ k ] = (DATATYPE**) malloc( y * sizeof( DATATYPE* ) );
        printf( "Filter %d:\n", k+1 );
        for( uint16_t i = 0; i < y; i++ )
        {
            weights[ k ][ i ] = (DATATYPE*) malloc( x * sizeof( DATATYPE ) );
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
}*/


int testTinyPSSR()
{
    uint16_t i, j, k, 
             y, z;

    /**** LAYERS ****/
    /** conv2d **/
    LAYERS_Conv2D conv2d;
    if( LAYERS_conv2d_constructor( &conv2d,
            1, 16,                                  // Input/Output Filters
            5, 5,                                   // Kernel Size
            1, 1,                                   // Kernel Stride
            1,                                      // Padded
            "models/1356-rgb/weights/conv2d",       // Weights and Biases File
            (*LAYERS_convolution_2d)                // Forward Function
        ) 
    )
    {
        return 1;
    }

    /** p_re_lu **/
    LAYERS_PreLU prelu;
    if( LAYERS_prelu_constructor( &prelu,
            conv2d.outFilters,
            "models/1356-rgb/weights/p_re_lu",
            (*LAYERS_prelu_forward)
        ) 
    )
    {
        return 1;
    }

    /** conv2d_1 **/
    LAYERS_Conv2D conv2d_1;
    if( LAYERS_conv2d_constructor( &conv2d_1,
            16, 4,                                  // Input/Output Filters
            1, 1,                                   // Kernel Size
            1, 1,                                   // Kernel Stride
            1,                                      // Padded
            "models/1356-rgb/weights/conv2d_1",     // Weights and Biases File
            (*LAYERS_convolution_2d)                // Forward Function
        ) 
    )
    {
        return 1;
    }


    /** p_re_lu_1 **/
    LAYERS_PreLU prelu_1;
    if( LAYERS_prelu_constructor( &prelu_1,
            conv2d_1.outFilters,
            "models/1356-rgb/weights/p_re_lu_1",
            (*LAYERS_prelu_forward)
        ) 
    )
    {
        return 1;
    }


    /** conv2d_2 **/
    LAYERS_Conv2D conv2d_2;
    if( LAYERS_conv2d_constructor( &conv2d_2,
            4, 8,                                   // Input/Output Filters
            5, 5,                                   // Kernel Size
            1, 1,                                   // Kernel Stride
            1,                                      // Padded
            "models/1356-rgb/weights/conv2d_2",     // Weights and Biases File
            (*LAYERS_convolution_2d)                // Forward Function
        ) 
    )
    {
        return 1;
    }


    /** p_re_lu_2 **/
    LAYERS_PreLU prelu_2;
    if( LAYERS_prelu_constructor( &prelu_2,
            conv2d_2.outFilters,
            "models/1356-rgb/weights/p_re_lu_2",
            (*LAYERS_prelu_forward)
        ) 
    )
    {
        return 1;
    }


    /** conv2d_3 **/
    LAYERS_Conv2D conv2d_3;
    if( LAYERS_conv2d_constructor( &conv2d_3,
            8, 4,                                   // Input/Output Filters
            1, 1,                                   // Kernel Size
            1, 1,                                   // Kernel Stride
            1,                                      // Padded
            "models/1356-rgb/weights/conv2d_3",     // Weights and Biases File
            (*LAYERS_convolution_2d)                // Forward Function
        ) 
    )
    {
        return 1;
    }


    /** relu **/
    LAYERS_ReLU relu;
    if( LAYERS_relu_constructor( &relu,
            conv2d_3.outFilters,
            (*LAYERS_relu_forward)
        ) 
    )
    {
        return 1;
    }




    /**** MEMORY ALLOCATION ****/
    // Read the input image.
    uint8_t numImages = 14;
    char* filenames[ 14 ] = {
        "./data/set14/0_2x_LR",
        "./data/set14/1_2x_LR",
        "./data/set14/2_2x_LR",
        "./data/set14/3_2x_LR",
        "./data/set14/4_2x_LR",
        "./data/set14/5_2x_LR",
        "./data/set14/6_2x_LR",
        "./data/set14/7_2x_LR",
        "./data/set14/8_2x_LR",
        "./data/set14/9_2x_LR",
        "./data/set14/10_2x_LR",
        "./data/set14/11_2x_LR",
        "./data/set14/12_2x_LR",
        "./data/set14/13_2x_LR"
    };


    float timeSum = 0.0, layerTime = 0.0;
    uint8_t fileIdx = 0;
    for( fileIdx = 0; fileIdx < numImages; fileIdx++ )
    {
        FILE * inF = fopen( filenames[ fileIdx ], "rb" );
        if( inF == NULL ){ 
            // printf( "ERROR: Unable to open file '%s'.", filenames[ fileIdx ] );
            return 1;
        }

        // Read in input shape.
        uint16_t inY = getc( inF ) | getc( inF ) << 8;
        uint16_t inX = getc( inF ) | getc( inF ) << 8;
        uint16_t inC = getc( inF ) | getc( inF ) << 8;
        uint16_t maxC = 16;

        // Allocate space for the input image.
        DATATYPE *** input_image = (DATATYPE***) malloc( inC * sizeof( DATATYPE** ) );
        for( k = 0; k < inC; k++ )
        {
            input_image[ k ] = (DATATYPE**) malloc( inY * sizeof( DATATYPE* ) );
            for( i = 0; i < inY; i++ )
            {
                input_image[ k ][ i ] = (DATATYPE*) malloc( inX * sizeof( DATATYPE ) );
                for( j = 0; j < inX; j++ )
                {
                    input_image[ k ][ i ][ j ] = getc( inF ) / 255.0;
                }
            }
        }

        fclose( inF );

        // Allocate space for input matrices.
        DATATYPE *** input = (DATATYPE***) malloc( maxC * sizeof( DATATYPE** ) );
        for( k = 0; k < maxC; k++ )
        {
            input[ k ] = (DATATYPE**) malloc( inY * sizeof( DATATYPE* ) );
            for( i = 0; i < inY; i++ )
            {
                input[ k ][ i ] = (DATATYPE*) malloc( inX * sizeof( DATATYPE ) );
                memset( input[ k ][ i ], 0, inX );
            }
        }

        // Allocate space for output matrices.
        uint16_t outY = inY, outX = inX;
        DATATYPE *** output = (DATATYPE***) malloc( maxC * sizeof( DATATYPE** ) );
        for( z = 0; z < maxC; z++ )
        {
            output[ z ] = (DATATYPE**) malloc( outY * sizeof( DATATYPE* ) );
            for( y = 0; y < outY; y++ )
            {
                output[ z ][ y ] = (DATATYPE*) malloc( outX * sizeof( DATATYPE ) );
                memset( output[ z ][ y ], 0, outX * sizeof( DATATYPE ) );
            }
        }


        // printf( "[% 2d]:\n", fileIdx + 1 );
        clock_t start = clock();
        /**** FORWARD PASS ****/
            /** Pass first channel through conv2d layer. **/
            // clock_t tmp = clock();
            conv2d.forward( &output, input_image, inY, inX, conv2d.inFilters, conv2d );
            // layerTime = (float)(clock() - tmp) / CLOCKS_PER_SEC;
            // printf( "  %.4f\n", layerTime );

            /** p_re_lu **/
            // tmp = clock();
            prelu.forward( &input, output, inY, inX, prelu );
            // layerTime = (float)(clock() - tmp) / CLOCKS_PER_SEC;
            // printf( "  %.4f\n", layerTime );

            /** conv2d_1 **/
            // tmp = clock();
            conv2d_1.forward( &output, input, inY, inX, conv2d_1.inFilters, conv2d_1 );
            // layerTime = (float)(clock() - tmp) / CLOCKS_PER_SEC;
            // printf( "  %.4f\n", layerTime );

            /** p_re_lu_1 **/
            // tmp = clock();
            prelu_1.forward( &input, output, inY, inX, prelu_1 );
            // layerTime = (float)(clock() - tmp) / CLOCKS_PER_SEC;
            // printf( "  %.4f\n", layerTime );

            /** conv2d_2 **/
            // tmp = clock();
            conv2d_2.forward( &output, input, inY, inX, conv2d_2.inFilters, conv2d_2 );
            // layerTime = (float)(clock() - tmp) / CLOCKS_PER_SEC;
            // printf( "  %.4f\n", layerTime );

            /** p_re_lu_2 **/
            // tmp = clock();
            prelu_2.forward( &input, output, inY, inX, prelu_2 );
            // layerTime = (float)(clock() - tmp) / CLOCKS_PER_SEC;
            // printf( "  %.4f\n", layerTime );

            /** conv2d_3 **/
            // tmp = clock();
            conv2d_3.forward( &output, input, inY, inX, conv2d_3.inFilters, conv2d_3 );
            // layerTime = (float)(clock() - tmp) / CLOCKS_PER_SEC;
            // printf( "  %.4f\n", layerTime );

            /** relu **/
            // tmp = clock();
            relu.forward( &input, output, inY, inX, relu );
            // layerTime = (float)(clock() - tmp) / CLOCKS_PER_SEC;
            // printf( "  %.4f\n", layerTime );
        /**** /FORWARD PASS ****/
        clock_t end = clock();
        timeSum += (float)(end - start) / CLOCKS_PER_SEC;
        // Must Beat: 0.0419s
        // (for layers conv2d through conv2d_3 on one channel)
        // printf( "%.4f  ", (float)(end - start) / CLOCKS_PER_SEC );


        // Free input image.
        for( k = 0; k < inC; k++ )
        {
            for( i = 0; i < inY; i++ )
            {
                free( input_image[ k ][ i ] );
            }
            free( input_image[ k ] );
        }
        free( input_image );


        // Free input.
        for( k = 0; k < maxC; k++ )
        {
            for( i = 0; i < inY; i++ )
            {
                free( input[ k ][ i ] );
            }
            free( input[ k ] );
        }
        free( input );


        // Free output.
        for( z = 0; z < maxC; z++ )
        {
            for( y = 0; y < outY; y++ )
            {
                free( output[ z ][ y ] );
            }
            free( output[ z ] );
        }
        free( output );
    }


    printf( "\nTotal:   %.4f\n"
            "Average: %.4f\n", timeSum, timeSum / numImages );

    /**** FREE MEMORY ****/
    // Free weights and biases.
    LAYERS_conv2d_free_weights( &conv2d );
    LAYERS_prelu_free_weights( &prelu );
    LAYERS_conv2d_free_weights( &conv2d_1 );
    LAYERS_prelu_free_weights( &prelu_1 );
    LAYERS_conv2d_free_weights( &conv2d_2 );
    LAYERS_prelu_free_weights( &prelu_2 );
    LAYERS_conv2d_free_weights( &conv2d_3 );

    /*
    // Free input image.
    for( k = 0; k < inC; k++ )
    {
        for( i = 0; i < inY; i++ )
        {
            free( input_image[ k ][ i ] );
        }
        free( input_image[ k ] );
    }
    free( input_image );


    // Free input.
    for( k = 0; k < maxC; k++ )
    {
        for( i = 0; i < inY; i++ )
        {
            free( input[ k ][ i ] );
        }
        free( input[ k ] );
    }
    free( input );


    // Free output.
    for( z = 0; z < maxC; z++ )
    {
        for( y = 0; y < outY; y++ )
        {
            free( output[ z ][ y ] );
        }
        free( output[ z ] );
    }
    free( output );
    */

    return 0;
}
