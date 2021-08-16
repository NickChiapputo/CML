#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "activation.h"
#include "colors.h"
#include "mnist.h"
#include "loss.h"


#define TRAIN_DATA_PATH     "./data/mnist/t10k-images.idx3-ubyte"
#define TRAIN_LABEL_PATH    "./data/mnist/t10k-labels.idx1-ubyte"
#define TEST_DATA_PATH      "./data/mnist/train-images.idx3-ubyte"
#define TEST_LABELS_PATH    "./data/mnist/train-labels.idx1-ubyte"

#define BUFFER_LENGTH 64

// Output formatting.
#define ERROR   B_RED "ERROR: " RESET     // Error message color.


// Global function definitions.



int main( int argc, char ** argv )
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


    // Test CELoss.
    int numClasses = 5, numPredictions = 1;

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


    // Test Softmax implementation.
    numClasses = 4;
    double * input = (double*) malloc( numClasses * sizeof( double ) );
    double * output = (double*) malloc( numClasses * sizeof( double ) );
    input[ 0 ] = -1;
    input[ 1 ] = 0;
    input[ 2 ] = 3;
    input[ 3 ] = 5;
    ACTIVATION_softmax( input, numClasses, output );
    printf( "Softmax Activation:\n" );
    for( int i = 0; i < numClasses; i++ )
        printf( "    %lf\n", output[ i ] );
    free( input );
    free( output );

    return 0;
}

