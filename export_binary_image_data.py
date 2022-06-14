import numpy as np
import os

def main():
    # Params
    test_set = 'set14'
    test_set_scale = 2
    test_set_resolution = 'LR'

    num_imgs_export = 1000

    data_location = f'./data/{test_set}/'
    data_filename = f'{data_location}{test_set_scale}x_{test_set_resolution}.npy'
    save_location = f'{data_location}'


    # Load in the dataset and select the desired number of images.
    dataset = np.load( data_filename, allow_pickle=True )

    if num_imgs_export > len( dataset ):
        num_imgs_export = len( dataset )

    # Open the file and write binary values.
    for i in range( len( dataset ) ):
        filename = f'{save_location}{i}_{test_set_scale}x_{test_set_resolution}'
        os.makedirs( os.path.dirname( filename ), exist_ok=True )

        height, width, num_channels = dataset[ i ].shape
        img_shape = np.array( [ height, width, num_channels ], 
            dtype=np.uint16 )
        print( f"Saving image {i+1} to '{filename}'.\n  {height}, {width}, {num_channels}" )

        with open( filename, 'wb' ) as f:
            f.write( img_shape[ 0 ] )   # Height
            f.write( img_shape[ 1 ] )   # Width
            f.write( img_shape[ 2 ] )   # Channels

            for z in range( num_channels ):
                for y in range( height ):
                    for x in range( width ):
                        f.write( dataset[ i ][ y, x, z ] )


if __name__ == "__main__":
    main()