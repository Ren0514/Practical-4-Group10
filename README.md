# Practical-4-Group10

Practical 4 (group-work): Stochastic gradient descent

Group 10:
Xinyi Ren(s2506973)
Xinhao Liu(s2495872)
Yingjie Yang(s2520758)

Overview
The code implements a neural network model to classify irises to species based
on the 4 characteristics given in the iris dataset. The neural network is
designed for a three-class classification task of irises species: setosa
encoded as 1, versicolor encoded as 2 and virginica encoded as 3. The main
idea of model training is Stochastic gradient descent.

The code includes 4 main steps:

1. Data preprocessing:
   Load the Iris dataset and transform labels of species into numeric values.
   Divide the dataset into training and test sets.

2. Initialization:
   Initialize a 4-8-7-3 neural network using the netup function.

3. Model training:
   Train the neural network using stochastic gradient descent (SGD) with the
   train function and the training data. The train function iterates over a
   specified number of training steps, randomly sampling some data at each step.
   For each sample set, implement functions of forward and backward propagation
   to update gradient of loss. After processing every data in each sample set,
   the weights and offsets is updated by the average gradients.

4. Model prediction:
   Make predictions on the test set using the predict_nn function with the trained
   network from train function. And calculate and print the misclassification
   rate of the trained model. The misclassification rate is the proportion of misclassified test examples in total test examples.
