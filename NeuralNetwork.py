from ActivationFunctions import Sigmoid, N, ReLU
from Layer import Layer
import numpy as np

class Error:
  def calculate_error(expected_value, result):
    return (expected_value - result)**2

  def derivative(expected_value, result):
    return (expected_value - result)

class NN:

  def __init__(self, X, y, activation):

    self.layers = []

    self.X = X
    self.y = y

    self.activation = activation
    self.output_activation = N()

  def initialize_layers(self, input_size, output_size, hidden_layers=0, neurons_per_layer=4, learning_rate=0.01):

    self.number_of_layers = 2 + hidden_layers

    for i in range(self.number_of_layers):

      if i == 0:
        self.layers.append(Layer(input_size,neurons_per_layer,learning_rate))
      elif i == self.number_of_layers-1:
        self.layers.append(Layer(neurons_per_layer,output_size,learning_rate))
      else:
        self.layers.append(Layer(neurons_per_layer,neurons_per_layer,learning_rate))

  def train(self):

    self.pred(self.X)
    self.backpropagation()

  def backpropagation(self):
    i = self.number_of_layers-1

    while i >= 0:
      if i == self.number_of_layers-1:
        self.error_derivative = Error.derivative(self.y,self.layers_output[i])
        self.previous_grad = self.error_derivative * self.output_activation.derivative(self.layers_output[i])

      elif i == 0:
        self.grad = (self.previous_grad @ self.layers[i+1].weights.T) * self.activation.derivative(self.layers_output[i])

        self.layers[i+1].update_weights(self.previous_grad,self.layers_output[i])
        self.layers[i+1].update_bias(self.previous_grad)

        self.layers[i].update_weights(self.grad,self.X)
        self.layers[i].update_bias(self.grad)

        self.previous_grad = self.grad

      else:
        self.grad = (self.previous_grad @ self.layers[i+1].weights.T) * self.activation.derivative(self.layers_output[i])

        self.layers[i+1].update_weights(self.previous_grad,self.layers_output[i])
        self.layers[i+1].update_bias(self.previous_grad)

        self.previous_grad = self.grad

      i -= 1

  def get_result(self):
    return self.layers[self.number_of_layers-1].output

  def loss(self):
    return np.mean(Error.calculate_error(self.y, self.layers[self.number_of_layers-1].output))

  def pred(self,X_sample):
    self.layers_output = []

    for i in range(self.number_of_layers):

      if i == 0:
        self.layers_output.append(self.activation.calculate_func(self.layers[i].get_output(X_sample)))

      elif i == self.number_of_layers-1:
        self.layers_output.append(
            self.output_activation.calculate_func(
                self.layers[i].get_output(self.layers_output[i-1])
        ))

      else:
        self.layers_output.append(
            self.activation.calculate_func(
                self.layers[i].get_output(self.layers_output[i-1])
        )) 