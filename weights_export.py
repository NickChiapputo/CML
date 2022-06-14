import tensorflow as tf
import os
import struct
import numpy as np

def main():
    model_location = f'./models/1356-rgb/'
    save_location = f'{model_location}weights/'


    # Load in the model.
    print( f"Loading model..." )
    model = tf.keras.models.load_model( model_location, compile=False )
    model.model.summary()

    # Get the first layer and its weights.
    first_conv_layer = model.model.layers[ 1 ]
    first_conv_layer_weights = first_conv_layer.get_weights()
    print( first_conv_layer_weights[ 0 ].shape )

    # Calculate parameters.
    outC         = first_conv_layer_weights[ 0 ].shape[ 3 ]
    inC          = first_conv_layer_weights[ 0 ].shape[ 2 ]
    kernelWidth  = first_conv_layer_weights[ 0 ].shape[ 1 ]
    kernelHeight = first_conv_layer_weights[ 0 ].shape[ 0 ]


    os.makedirs( os.path.dirname( save_location ), exist_ok=True )

    # Save conv2d layer.
    with open( f"{save_location}{first_conv_layer.name}", "wb" ) as f:
        print( f"Saving weights for layer '{first_conv_layer.name}' to '{save_location}{first_conv_layer.name}'." )

        # Original shape: kernelHeight x kernelWidth x inputChannels x outputChannels
        # New shape:      kernelHeight x outputChannels x inputChannels x kernelWidth
        # This is done so that when the structs are written to the file, it does
        # so across each row (e.g., all columns in first row, all columns in 
        # second row, and so on). Prints floats in little-endian 
        # IEEE.754 32-bit format.
        weights = np.swapaxes( first_conv_layer_weights[ 0 ], 1, 3 )

        for z in range( outC ):
            for k in range( inC ):
                for i in range( kernelHeight ):
                    f.write( struct.pack( 'f' * len( weights[ i ][ z ][ k ] ), 
                        *weights[ i ][ z ][ k ] ) )

            # Write biases to file.
            f.write( struct.pack( 'f' * len( np.array( [first_conv_layer_weights[ 1 ][ z ]] ) ), 
                *np.array( [first_conv_layer_weights[ 1 ][ z ]] ) ) )
        print( f"  Done.\n" )

    # Save p_re_lu layer.
    layer = model.model.layers[ 2 ]
    layer_weights = layer.get_weights()
    with open( f"{save_location}{layer.name}", "wb" ) as f:
        print( f"Saving weights for layer '{layer.name}' to '{save_location}{layer.name}'." )

        f.write( struct.pack( 'f' * len( layer_weights[ 0 ][ 0, 0 ] ), 
            *layer_weights[ 0 ][ 0, 0 ] ) )

    # Save conv2d_1 layer.
    layer = model.model.layers[ 3 ]
    layer_weights = layer.get_weights()

    outC         = layer_weights[ 0 ].shape[ 3 ]
    inC          = layer_weights[ 0 ].shape[ 2 ]
    kernelWidth  = layer_weights[ 0 ].shape[ 1 ]
    kernelHeight = layer_weights[ 0 ].shape[ 0 ]
    with open( f"{save_location}{layer.name}", "wb" ) as f:
        print( f"Saving weights for layer '{layer.name}' to '{save_location}{layer.name}'." )
        weights = np.swapaxes( layer_weights[ 0 ], 1, 3 )

        for z in range( outC ):
            for k in range( inC ):
                for i in range( kernelHeight ):
                    f.write( struct.pack( 'f' * len( weights[ i ][ z ][ k ] ), 
                        *weights[ i ][ z ][ k ] ) )

            # Write biases to file.
            f.write( struct.pack( 'f' * len( np.array( [layer_weights[ 1 ][ z ]] ) ), 
                *np.array( [layer_weights[ 1 ][ z ]] ) ) )

    # Save p_re_lu_1 layer.
    layer = model.model.layers[ 4 ]
    layer_weights = layer.get_weights()
    with open( f"{save_location}{layer.name}", "wb" ) as f:
        print( f"Saving weights for layer '{layer.name}' to '{save_location}{layer.name}'." )

        f.write( struct.pack( 'f' * len( layer_weights[ 0 ][ 0, 0 ] ), 
            *layer_weights[ 0 ][ 0, 0 ] ) )

    # Save conv2d_2 layer.
    layer = model.model.layers[ 5 ]
    layer_weights = layer.get_weights()

    outC         = layer_weights[ 0 ].shape[ 3 ]
    inC          = layer_weights[ 0 ].shape[ 2 ]
    kernelWidth  = layer_weights[ 0 ].shape[ 1 ]
    kernelHeight = layer_weights[ 0 ].shape[ 0 ]
    with open( f"{save_location}{layer.name}", "wb" ) as f:
        print( f"Saving weights for layer '{layer.name}' to '{save_location}{layer.name}'." )
        weights = np.swapaxes( layer_weights[ 0 ], 1, 3 )

        for z in range( outC ):
            for k in range( inC ):
                for i in range( kernelHeight ):
                    f.write( struct.pack( 'f' * len( weights[ i ][ z ][ k ] ), 
                        *weights[ i ][ z ][ k ] ) )

            # Write biases to file.
            f.write( struct.pack( 'f' * len( np.array( [layer_weights[ 1 ][ z ]] ) ), 
                *np.array( [layer_weights[ 1 ][ z ]] ) ) )

    # Save p_re_lu_2 layer.
    layer = model.model.layers[ 6 ]
    layer_weights = layer.get_weights()
    with open( f"{save_location}{layer.name}", "wb" ) as f:
        print( f"Saving weights for layer '{layer.name}' to '{save_location}{layer.name}'." )

        f.write( struct.pack( 'f' * len( layer_weights[ 0 ][ 0, 0 ] ), 
            *layer_weights[ 0 ][ 0, 0 ] ) )

    # Save conv2d_3 layer.
    layer = model.model.layers[ 7 ]
    layer_weights = layer.get_weights()

    outC         = layer_weights[ 0 ].shape[ 3 ]
    inC          = layer_weights[ 0 ].shape[ 2 ]
    kernelWidth  = layer_weights[ 0 ].shape[ 1 ]
    kernelHeight = layer_weights[ 0 ].shape[ 0 ]
    with open( f"{save_location}{layer.name}", "wb" ) as f:
        print( f"Saving weights for layer '{layer.name}' to '{save_location}{layer.name}'." )
        weights = np.swapaxes( layer_weights[ 0 ], 1, 3 )

        for z in range( outC ):
            for k in range( inC ):
                for i in range( kernelHeight ):
                    f.write( struct.pack( 'f' * len( weights[ i ][ z ][ k ] ), 
                        *weights[ i ][ z ][ k ] ) )

            # Write biases to file.
            f.write( struct.pack( 'f' * len( np.array( [layer_weights[ 1 ][ z ]] ) ), 
                *np.array( [layer_weights[ 1 ][ z ]] ) ) )

if __name__ == "__main__":
    main()