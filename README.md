# Linear-Classifier-for-Data
The purpose of this project is to propose a linear classifier for the data accessible in the file data.py  

The data is accessible from the arrays numpy data['input'] and data['target']. 
The input vector is a vector of R3, the target value is an integer in the set {0,1,2,3} (4 classes).

It is a matter of predicting the value of target from input.

The task is therefore to find a function of R3 → {0, 1, 2, 3} that matches the data (as far as possible).

Exercice 1 :

1. Program a function that calculates the number of errors made by a linear classifier.

2. Program a function that calculates a loss for this classifier (based on the log-likelihood and softmax).

3. Apply gradient descent to this loss.

4. Indicate the solution found, and the number of errors made by this solution.
