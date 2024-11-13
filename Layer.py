import numpy as np

def XavierInitialization(input_size, output_size):
    return np.sqrt(6 / (input_size + output_size))

class Layer:
  def __init__(self,input_size,output_size,learning_rate):
    self.input_size = input_size
    self.output_size = output_size
    self.learning_rate = learning_rate

    self.weights = np.random.uniform(-(XavierInitialization(input_size,output_size)), XavierInitialization(input_size,output_size), size=(self.input_size, self.output_size))
    self.bias = np.zeros(shape = (1, self.output_size))
    self.output = None

  def get_output (self, X):
    self.output = X @ self.weights + self.bias
    return self.output

  def update_weights(self, grad, input):
    self.weights += self.learning_rate * input.T @ grad 

  def update_bias(self, grad):
    self.bias += self.learning_rate * np.sum(grad, axis=0, keepdims=True) 