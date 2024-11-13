import numpy as np

class Sigmoid:
  def calculate_func(self,X):
    return 1/(1+np.exp(-X))

  def derivative(self,X):
    return self.calculate_func(X)*(1-self.calculate_func(X))

class N:
  def calculate_func(self,X):
    return X

  def derivative(self,X):
    return 1

class ReLU:
  def calculate_func(self,X):
    return (X + abs(X))/2

  def derivative(self,X):
    return np.where(X > 0, 1, 0)