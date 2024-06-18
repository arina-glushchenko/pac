import numpy as np
import random

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

def loss(y, label):
    return label - y

class Layer:
    def __init__(self, l=0, w = [], b = []):  # Исправление: добавляем значение по умолчанию для l
        self.weights = w
        self.bias = b
        self.l = l
        self.s = 0

    def forward(self, x):
        self.s = np.dot(x, self.weights) + self.bias
        self.predicted_output = sigmoid(self.s)
        return self.predicted_output

    def backward(self, x, err, p, d_hidden_layer, d_predicted_output, hidden_layer_output):
        lr = 0.1
        if (p == 0):
            self.weights += inputs.T.dot(d_hidden_layer) * lr
            self.bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr
        else:
            self.weights += hidden_layer_output.T.dot(d_predicted_output) * lr
            self.bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr



class Model:
    def __init__(self, weights=[], hidden_bias=[], output_weight=[], output_bias=[], inputLayerNeurons=2,hiddenLayerNeurons=2, outputLayerNeurons=1):
        self.hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
        self.hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
        self.output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
        self.output_bias = np.random.uniform(size=(1, outputLayerNeurons))
        self.l1 = Layer(0, self.hidden_weights,self.hidden_bias)
        self.l2 = Layer(1, self.output_weights, self.output_bias)

    def forward(self, x):
        self.hidden_layer_output = self.l1.forward(x)
        self.predicted_output = self.l2.forward(self.hidden_layer_output)
        return self.predicted_output

    def backward(self, x, err):
        lr = 0.1
        d_predicted_output = err * sigmoid_derivative(self.predicted_output)

        error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
        d_hidden_layer = error_hidden_layer * sigmoid_derivative(self.hidden_layer_output)

        # Updating Weights and Biases

        self.l2.backward(x, err,1, d_hidden_layer, d_predicted_output, self.hidden_layer_output)
        self.l1.backward(x, err,0, d_hidden_layer, d_predicted_output, self.hidden_layer_output)
        # self.output_weights += self.hidden_layer_output.T.dot(d_predicted_output) * lr
        # self.output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
        # self.hidden_weights += inputs.T.dot(d_hidden_layer) * lr
        # self.hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    def train(self, inputs, expected_outputs, epochs):
                for _ in range(epochs):
                    y = self.forward(inputs)
                    err = loss(y, expected_outputs)
                    self.backward(inputs, err)
                return(y)



inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outputs = np.array([[0], [1], [1], [0]])

model = Model()
print(model.train(inputs, expected_outputs, epochs=10000))