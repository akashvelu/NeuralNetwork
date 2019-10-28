import NeuralNetwork as nn
import numpy as np
from mnist import MNIST
import math


def format(images, labels):
	data = []
	for index in range(0, len(labels)):
		input = np.array(images[index]) / 255
		output = np.zeros(10)
		output[labels[index]] = 1.0
		data.append((input, output))
	return data

print("Loading and formatting MNIST set")
mnist_set = MNIST('MNIST_data')
training_inputs, training_outputs = mnist_set.load_training()
test_inputs, test_outputs = mnist_set.load_testing()

training_data = format(training_inputs, training_outputs)
test_data = format(test_inputs, test_outputs)


mnist_nn = nn.NeuralNetwork(4, [784, 100, 50, 10])
print('Training neural network')
# train for 5 epochs with a learning rate of 0.5
mnist_nn.train(training_data, 5, 0.5)


print("Testing neural network")
numCorrect = 0
for i in range(len(test_data)):
    data = test_data[i]
    input = data[0]
    expected = test_outputs[i]
    prediction, vector = mnist_nn.predict(input)
    if (prediction == expected):
        numCorrect += 1
print("Test accuracy: ", numCorrect / len(test_data))
