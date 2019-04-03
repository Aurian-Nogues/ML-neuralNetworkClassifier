# ML-neuralNetworkClassifier
Matlab / Octave program to implement a neural network classifier algorithm

This programs trains a neural network to read hand written digits. It also allows to visualise the hidden layers of the neural network and assess classification accuracy.

Content:

training_data.mat - training set of hand written digits
displayData.m - Function to help visualize the dataset
fmincg.m - Function minimization routine
sigmoid.m - Sigmoid function
computeNumericalGradient.m - Numerically compute gradients ( to check backpropagation )
nnCostFunction.m - Neural network cost function
predict.m - Neural network prediction function
sigmoid.m - Sigmoid function
sigmoidGradient.m - compute gradient of sigmoid function
randInitializeWeights.m - randomly initialu=izw weights to break symmetry
training.m - script to load data, vizualize it, train the algo, vizualize hidden layers, assess prediction accuracy and predict random exemples