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
		for i in range(1, len(numLayers)):
			# initialize weights to uniform random values between -1, 1 (np.random.rand returns values in [0, 1])
			weightMatrix = 2 * np.random.rand(self.layerDims[i], self.layerDims[i - 1]) - 1
			biasVector = 2 * np.random.rand(self.layerDims[i])
			self.weights.append(weightMatrix)
			self.biases.append(biasVector)

	# forward pass algorithm
	def forwardPass(self, input):
		layerValues = []
		layerValues.append(input)
		for i in range(1, self.numLayers):
			newLayer = np.dot(self.weights[i - 1], layerValues[i - 1]) + self.biases[i - 1]
			this.layerValues.append(newLayer)
		return layerValues


	def backPropagation(self, layerValues, expectedOutput):
		currLayerValues = layerValues[self.numLayers - 1]
		biasGradients = []
		layerGradients = []
		weightJacobians = []
		layerGradients.append(currLayerValues - expectedOutput)
		biasGradients.append(currLayerValues * (1 - currLayerValues) * layerGradients[0])
		weightJacobians.append(np.dot(biasGradients[0].T, currLayerValues))

		for l in range(self.numLayers - 2, 0, -1):
			currLayerValues = layerValues[l]
			inter = layerGradients[0] * layerValues[l + 1] * (1 - layerValues[l + 1])
			layerGradients.insert(0, np.dot(self.weights[l + 1].T, inter))
			biasGradients.insert(0, layerGradients[0] * (1 - currLayerValues) * currLayerValues)
			# outer product
			jacobian = np.dot(biasGradients[0].T, currLayerValues)
			weightJacobians.insert(0, jacobian)

		return weightJacobians, biasGradients


	def gradientDescent(self, weightJacobians, biasGradients, learningRate):
		for i in range(len(self.weights)):
			self.weights[i] -= (learningRate * weightJacobians)
			self.biases[i] -= (learningRate* biasGradients)


	def train():
		return None

	def predict(self, input):
		output = this.forwardPass(input)
		return np.argmax(output), output

	def mse_error(self, output, expectedOutput):
		diff = output - expectedOutput
		return(np.dot(diff, diff))
	
