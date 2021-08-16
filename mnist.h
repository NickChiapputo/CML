#ifndef MNIST_H_
#define MNIST_H_

/****
    *   @description        Parse the header information from the bytefile for an MNIST image set file or from
    *                       an MNIST label file (train or test). The <data> flag is set to 1 for image set file
    *                       or to 0 for a label file. Read from the bytefile in blocks of 4 bytes (32 bits) to 
    *                       capture the magic number, number of images in the dataset, and number of rows and 
    *                       columns in the images (if a data file).
    * 
    *   @param  filename    Location of dataset file.
    *   @param magicNumber  Pointer to the unsigned 32-bit magic number at the start of the file.
    *                       Value should be 2,051.
    *   @param numImages    Pointer to the unsigned 32-bit count of images in the file.
    *                       Value should be 60,000.
    *   @param rows         Pointer to the unsigned 32-bit count of rows in the images.
    *                       Value should be 28.
    *   @param cols         Pointer to the unsigned 32-bit count of columns in the images.
    *                       Value should be 28.
    * 
    *   @return             0 for success, 1 for error.
****/
int MNIST_parseHeader( char * filename, uint32_t * magicNumber, uint32_t * numImages, uint32_t * rows, uint32_t * cols, int data );


/****
    *   @description        Read first <numRead> number of images from the dataset image file given by <dataFilename>
    *                       and corresponding labels from the label file given by <labelFilename>.
    *                       Files can be either train or test datasets. Data is stored in provided pre-allocated
    *                       array of arrays (<data>) and labels are stored in pre-allocated arrat (<labels>). Size of 
    *                       <data> and <labels> should be <numRead> arrays of size 28*28=784 (rows times columns of 
    *                       the images). Each array is a flattened image from the dataset in row-major order (exactly 
    *                       as stored in the dataset files).
    *                       
    *   @param dataFilename Location of the dataset image file.
    *   @param dataFilename Location of the label file.
    *   @param data         Pre-allocated array of arrays to hold data.
    *   @param labels       Pre-allocated array of labels corresponding to data.
    *   @param numRead      Number of images to read from the dataset image file.
    * 
    *   @return             0 for success, 1 for error.
****/
int MNIST_readDataFile( char * dataFilename, char * labelFilename, uint8_t ** data, uint8_t * labels, int numRead );

#endif