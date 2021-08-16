#ifndef ACTIVATION_H_
#define ACTIVATION_H_

/****
	*	@description		Softmax activation function takes an array of inputs
	* 						and calculates a set of probabilities for each class.
	* 						The input values are all real values. It is expected
	* 						that the output array is pre-allocated. Some math tricks
	* 						are used in order to ensure number safety in the event
	* 						any of the input values are too large (> ~710). An explanation
	* 						of the implementation can be seen here: https://stackoverflow.com/a/34969389/2898057.
	* 						We also use the LogSumExp approach to prevent under/overflow problems
	* 						as defined here: https://en.wikipedia.org/wiki/LogSumExp.
	* 
	* 	@param input		Set of real value inputs.
	* 	@param inputLength	Number of input values (same as the number of outputs)
	* 	@param output 		Pre-allocated array for output probabilities. Sums to 1.
	* 
	* 	@return 			0 for success, 1 for error.
****/
int ACTIVATION_softmax( double * input, int inputLength, double * output );


/****
	*	@description
	* 
	* 
	* 
****/
double ACTIVATION_relu(  );

#endif