Neural Network Implementation from Scratch
This repository contains a simple implementation of a feedforward neural network from scratch in Python. The neural network is designed to solve the classic AND problem using gradient descent and backpropagation without relying on any deep learning libraries.

Project Overview
In this project, a simple neural network with one hidden layer is implemented. The goal is to train the network to predict the output of the AND operation, given two binary inputs. The network is trained using gradient descent, backpropagation, and the sigmoid activation function.

Key Features
Implements a neural network with one hidden layer.
Uses the sigmoid activation function for both the hidden and output layers.
Trains the model using gradient descent and backpropagation.
Solves the AND problem, which is a binary classification task.
Problem Definition
Dataset
The dataset used for this task is the binary input-output pairs for the AND operation:

Input: X = [[0, 0], [0, 1], [1, 0], [1, 1]]
Output: y = [[0], [0], [0], [1]]
Task
Train the neural network to predict the output of the AND operation based on the provided input combinations.

Methodology
Neural Network Architecture
Input Layer: 2 neurons (representing the two binary inputs).
Hidden Layer: 3 neurons with sigmoid activation.
Output Layer: 1 neuron with sigmoid activation for binary classification.
Forward Pass
Input is passed to the hidden layer, where weighted sums are calculated.
The hidden layer output is obtained by applying the sigmoid activation function.
The hidden layer output is passed to the output layer for further computation.
The final output is obtained by applying the sigmoid activation function in the output layer.
Backpropagation
The error between the predicted and actual output is calculated.
The error is propagated backward through the network, adjusting the weights and biases based on the calculated gradients using the chain rule.
The weights and biases are updated using gradient descent.
Loss Function
The Mean Squared Error (MSE) loss function is used to calculate the difference between the predicted and actual output.
Optimization
The model is trained using gradient descent to minimize the loss function, adjusting weights and biases during each iteration.
How to Run the Code
Clone the repository to your local machine.

bash
Copy code
git clone https://github.com/yourusername/NeuralNetwork-Implementation.git
Navigate to the project directory.

bash
Copy code
cd NeuralNetwork-Implementation
Run the Python script to train the neural network.

bash
Copy code
python neural_network.py
The network will train on the AND problem dataset for 10,000 epochs and output the loss at regular intervals (every 1000 epochs).

Results
After training, the model predicts the output for the AND problem dataset, showing the results of the predictions for each input combination.

Technologies Used
Python 3.x
NumPy (for matrix operations)
License
This project is licensed under the MIT License - see the LICENSE file for details.

Author
[Your Name]
Acknowledgments
This implementation is inspired by fundamental concepts of neural networks and machine learning.
