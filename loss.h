#ifndef LOSS_H_
#define LOSS_H_

/****
    *   @description            Compute the cross entropy loss using the equation 
    *                           -(1/N) * sum_{i=1}^{N}{sum_{j=1}^{k}{t(i,j)*log(p(i,j)}} given N predictions,
    *                           k classes, t is the 2D array of targets, p is the 2D array of predicted values,
    *                           and log is the natural logarithm.
    * 
    *   @param predictions      2D array of predicted values. Sum of each row should add up to 1.
    *   @param target           2D array of one-hot rows for target predictions.
    *   @param numPredictions   Number of predictions (or rows in predictions and target).
    *   @param numClasses       Number of classes (or columns in predictions and target).
    * 
    *   @return                 The computed cross-entropy loss value. A double is used for greater accuracy than float.
****/
double LOSS_crossEntropy( double ** predictions, uint8_t ** targets, int numPredictions, int numClasses );

#endif