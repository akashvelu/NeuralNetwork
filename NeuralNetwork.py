import numpy as np

def sigmoid(x):
	denom = 1 + np.exp(-1 * x)
	return (1 / denom)

def relu(x):
	return None

class NeuralNetwork:

    # numLayers is number of layers in Neural Network, including the input layer and output layer
    # there are (numLayers - 2) hidden layers
    # we need (numLayers - 1) weights and biases
    def __init__(self, numLayers, layerDims):
        self.numLayers = numLayers
        self.layerDims = layerDims

        self.weights = []
        self.biases = []
        for i in range(1, numLayers):
            # initialize weights to uniform random values between -1, 1 (np.random.rand returns values in [0, 1])
            weightMatrix = 2 * np.random.rand(self.layerDims[i], self.layerDims[i - 1]) - 1
            biasVector = 2 * np.random.rand(self.layerDims[i]) - 1
            self.weights.append(weightMatrix)
            self.biases.append(biasVector)

    # forward pass algorithm
    def forwardPass(self, input):
        layerValues = []
        layerValues.append(input)
        for i in range(1, self.numLayers):
            z = np.dot(self.weights[i - 1], layerValues[i - 1]) + self.biases[i - 1]
            newLayer = sigmoid(z)
            layerValues.append(newLayer)
        return layerValues

    def backPropagate(self, layers, expected):
        deltas = []
        biasGradients = []
        weightJacobians = []

        currLayer = layers[self.numLayers - 1]
        deltas.insert(0, (currLayer - expected) * currLayer * (1 - currLayer))

        for l in range(self.numLayers - 1, 0, -1):
            biasGradients.insert(0, deltas[0])
            jacobian = np.outer(deltas[0], layers[l - 1])
            weightJacobians.insert(0, jacobian)

            nextDelta = np.dot(self.weights[l - 1].T, deltas[0]) * layers[l - 1] * (1 - layers[l-1])
            deltas.insert(0, nextDelta)

        return weightJacobians, biasGradients



    def gradientDescent(self, learningRate, weightJacobians, biasGradients):

        for i in range(len(weightJacobians)):
            self.weights[i] -= (learningRate * weightJacobians[i])
            self.biases[i] -= (learningRate * biasGradients[i])


    def train(self, trainingData, epochs, learningRate):
        epoch = 0
        trainingSetSize = len(trainingData)

        while (epoch < epochs):
            print("Epoch: ", epoch)
            for i in range(len(trainingData)):
                currInput = trainingData[i][0]
                currExpected = trainingData[i][1]
                currLayers = self.forwardPass(currInput)
                weightJacobians, biasGradients = self.backPropagate(currLayers, currExpected)
                # print(biasGradients)
                self.gradientDescent(learningRate, weightJacobians, biasGradients)
            epoch += 1
        return


    def predict(self, input):
        output = self.forwardPass(input)[-1]
        return np.argmax(output), output

    def mse_error(self, output, expectedOutput):
        diff = output - expectedOutput
        return(np.dot(diff, diff))
