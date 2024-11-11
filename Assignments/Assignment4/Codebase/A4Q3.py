import numpy as np

#implementaiton 1
inputVector = np.array([4, 6, 8])

class neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

neuronOne = neuron([0.8, -0.2, 0.1], 0.3)
neuronTwo = neuron([0.5, 0.7, -0.2], -0.4)
neuronThree = neuron([1.1, 0.6, -0.3], 0.2)
neuronFour = neuron([0.2, -0.4, 0.2], 0.1)
neuronFive = neuron([0.3, 0.1, 0.9, 0.1], -0.9)
neuronSix = neuron([0.3, -0.1, 0.2, -0.5], 0.7)

def calcFCL(vector, neuron):
    vectorSum = 0
    for feature, weight in zip(vector, neuron.weights):
        vectorSum += feature * weight
    return vectorSum + neuron.bias

intermediateVector = np.array([])
intermediateVector = np.append(intermediateVector, [calcFCL(inputVector, neuronOne)])
intermediateVector = np.append(intermediateVector, [calcFCL(inputVector, neuronTwo)])
intermediateVector = np.append(intermediateVector, [calcFCL(inputVector, neuronThree)])
intermediateVector = np.append(intermediateVector, [calcFCL(inputVector, neuronFour)])
print(intermediateVector)

outputVector = np.array([])
outputVector = np.append(outputVector, [calcFCL(intermediateVector, neuronFive)])
outputVector = np.append(outputVector, [calcFCL(intermediateVector, neuronSix)])
print(outputVector)

#implementation 2
inVec = [4, 6, 8]

neuronsOne = []
neuronsOne += [[[0.8, -0.2, 0.1], 0.3]]
neuronsOne += [[[0.5, 0.7, -0.2], -0.4]]
neuronsOne += [[[1.1, 0.6, -0.3], 0.2]]
neuronsOne += [[[0.2, -0.4, 0.2], 0.1]]
neuronsTwo = []
neuronsTwo += [[[0.3, 0.1, 0.9, 0.1], -0.9]]
neuronsTwo += [[[0.3, -0.1, 0.2, -0.5], 0.7]]

def calcFCL(vector, neuron, num):
    vectorSum = 0
    for i in range(len(vector)):
        vectorSum += vector[i] * neuron[num][0][i]
    return vectorSum + neuron[num][1]

interVec = []
for i in range(4):
    interVec += [round(calcFCL(inVec, neuronsOne, i), 1)]
print(f"Intermediate Vector:\n{interVec}\n")

outVec = []
for i in range(2):
    outVec += [calcFCL(interVec, neuronsTwo, i)]
print(f"OutputVector:\n{outVec}")
