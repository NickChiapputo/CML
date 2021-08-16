#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "mnist.h"


int MNIST_parseHeader( char * filename, uint32_t * magicNumber, uint32_t * numImages, uint32_t * rows, uint32_t * cols, int data )
{
    FILE * fp = fopen( filename, "rb" );
    if( fp == NULL )
        return 1;


    int i, c;
    for( i = 0; i < 4 && ( c = getc( fp ) ) != EOF; i++ )
        (*magicNumber) |= c << ( 24 - ( 8 * i ) );

    for( i = 0; i < 4 && ( c = getc( fp ) ) != EOF; i++ )
        (*numImages) |= c << ( 24 - ( 8 * i ) );

    if( !data )
    {
        fclose( fp );
        return 0;
    }

    for( i = 0; i < 4 && ( c = getc( fp ) ) != EOF; i++ )
        (*rows) |= c << ( 24 - ( 8 * i ) );

    for( i = 0; i < 4 && ( c = getc( fp ) ) != EOF; i++ )
        (*cols) |= c << ( 24 - ( 8 * i ) );


    fclose( fp );
    return 0;
}

int MNIST_readDataFile( char * dataFilename, char * labelFilename, uint8_t ** data, uint8_t * labels, int numRead )
{
    FILE * datafp = fopen( dataFilename, "rb" );
    FILE * labelfp = fopen( labelFilename, "rb" );
    if( datafp == NULL || labelfp == NULL )
        return 1;


    // The data starts after the data header. 
    // Skip the data header (length 16 bytes).
    // Skip the label header (length 8 bytes)
    if( fseek( datafp, 16L, SEEK_SET ) || fseek( labelfp, 8L, SEEK_SET ) )
        return 1;


    int imageNum, pixel, val;
    uint8_t label;
    for( imageNum = 0; imageNum < numRead && ( label = getc( labelfp ) ) != EOF; imageNum++ )
    {
        for( pixel = 0; pixel < 784 && ( val = getc( datafp ) ) != EOF; pixel++ )
            data[ imageNum ][ pixel ] = val;
        labels[ imageNum ] = label;
    }

    fclose( datafp );
    fclose( labelfp );
    return 0;
}
