# Lab Cuda Vision

## Exercise 1 
1. Train and Evaluate an MLP Classifier on the FashionMNIST dataset using PyTorch
2. Draw learning curves (train/eval loss, train/eval classification accuracy) and confusion matrix
3. Use Optuna to optimize (at least) two hyper-parameters (e.g, number of layers or hidden dimension) and show me some nice optimization plots :)
4. Visualize the norm of the gradients for some parameters during training, i.e., how the error decreases for some parameters as the model learns

## Exercise 2
1. Train and compare the MLP from Assignment 1 and a simple CNN on the CIFAR-10 dataset (available in PyTorch) with somewhat optimized hyper-parameters
2. Visualize several convolutional kernels and activations from the first two convolutional layers
3. Train CNNs without regularization, with L2-Regularization, and with L1-Regularization. Compare the results: training and validation loss, accuracy, ...
4. Train model with and without data augmentation. Compare the results: training and validation loss, accuracy, ...

## Exercise 3

## Exercise 4
Task 1
1. Implement a Convolutional LSTM (ConvLSTM() and/or ConvLSTMCell()) from scratch
Task 2
1. Perform "Action Recognition" on the KTH-Actions dataset:
- https://www.csc.kth.se/cvap/actions/
- https://github.com/tejaskhot/KTH-Dataset
- Use spatial dimensionality of frames of 64x64
- Split videos into subsequences of e.g. 20 frames. Treat each of these subsequences as independent.
2. Implement a model with the following structure:
- Convolutional encoder
- Recurrent module
- Classifier (probably Conv + AvgPooling/Flattening + Linear)
3. Train, evaluate, and compare the model with the following recurrent modules:
- PyTorch LSTM model (using nn.LSTMCell)
- PyTorch GRU model (using nn.GRUCell)
- Your own ConvLSTM
Note: Different recurrent modules (e.g. LSTM vs ConvLSTM) might require slight changes in the encoder and classifier

## Tools and Libraries Used
- PyTorch
- Numpy
- Pandas
- Seaborn
- Matplotlib
- Optuna
