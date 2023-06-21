"""Definitions for neural network and running it."""

import math
import numpy as np
import random
from typing import List

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class TrainingDataset:
    inputs: List[List[float]]
    outputs: List[List[float]]
    l: int
    insize: int
    outsize: int

class NeuralNetwork:
    layers: int # Number of layers
    inputs: int # Size of input vector
    outputs: int # Size of output vector
    
    # All parameters of matrices.
    # There will be (inputs * inputs * (layers - 1)) + (inputs * outputs)
    # entries in this list -- (layers - 1) square matrices of the input size,
    # and a single layer mapping input size to output size.
    # Next is the list of bias vectors, which is (layers - 1) vectors with
    # (inputs) elements each.
    params: List[int]
    
    def __init__(self, layers, i, o):
        self.layers  = layers
        self.inputs  = i
        self.outputs = o
        lsize = (i * i * (layers - 1)) + (i * o) + (layers - 1) * i
        self.params = [random.random() * 2 - 1 for i in range(lsize)]
    
    def _middle_matrix(self, start_index: int, last_layer: bool) -> np.array:
        if not last_layer:
            return np.array([
                self.params[start_index + i*self.inputs : start_index + (i + 1)*self.inputs]
                for i in range(self.inputs)
            ])
        else:
            return np.array([
                self.params[start_index + i*self.inputs : start_index + (i + 1)*self.inputs]
                for i in range(self.outputs)
            ])
    
    # Get the output for a single thing.
    def apply(self, input: List[float]) -> List[float]:
        t = np.array(input)
        bindex = (self.inputs ** 2 * (self.layers - 1)) + (self.inputs * self.outputs)
        for i in range(self.layers - 1):
            t = np.matmul(self._middle_matrix(self.inputs ** 2 * i, False), t)
            for idx in range(len(t)):
                t[idx] = sigmoid(t[idx])
            for j in range(len(t)):
                t[j] += bindex + self.inputs * i + j
        t = np.matmul(self._middle_matrix(self.inputs ** 2 * (self.layers - 1), True), t)
        return t.tolist()
    
    # Returns the amount of inaccuracy on a single input.
    def obtain_loss(self, input: List[float], expected: List[float]) -> float:
        actual = self.apply(input)
        diff = sum([(actual[i] - expected[i]) ** 2 for i in range(len(actual))])
        return math.sqrt(diff) / len(actual)
    
    def obtain_data_loss(self, data: TrainingDataset) -> float:
        return sum([self.obtain_loss(data.inputs[i], data.outputs[i]) for i in range(data.l)]) / data.l
    
    # The loss that would be mitigated by changing parameter 'k'.
    def partial_diff(self, data: TrainingDataset, k: int):
        STEP_SIZE = 0.001
        current_loss = self.obtain_data_loss(data)
        self.params[k] += STEP_SIZE
        new_loss = self.obtain_data_loss(data)
        self.params[k] -= STEP_SIZE
        return (current_loss - new_loss) / STEP_SIZE
    
    # Adjust the matrix by one single increment.
    def learn(self, data: TrainingDataset, learning_speed: float) -> None:
        err = self.obtain_data_loss(data)
        adjustments = []
        for i in range(len(self.params)):
            adjustments.append(self.partial_diff(data, i) * learning_speed * math.sqrt(err))
        for i in range(len(adjustments)):
            self.params[i] += adjustments[i]
        new_err = self.obtain_data_loss(data)
        if new_err > err:
            for i in range(len(adjustments)):
                self.params[i] -= adjustments[i] / 2
