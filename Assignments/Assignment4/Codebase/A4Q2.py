import numpy as np

inputMap = np.array([[0.63, 0.79, 0.13, 0.46, 0.56, 0.20, 0.05, 0.83, 0.83],
                     [0.76, 0.86, 0.31, 0.57, 0.22, 0.70, 0.95, 0.42, 0.82],
                     [0.34, 0.62, 0.13, 0.07, 0.99, 0.83, 0.22, 0.19, 0.99],
                     [0.44, 0.91, 0.23, 0.68, 0.37, 0.05, 0.14, 0.19, 0.62],
                     [0.78, 0.97, 0.06, 0.07, 0.88, 0.60, 0.74, 0.60, 0.57],
                     [0.02, 0.28, 0.06, 0.02, 0.20, 0.77, 0.98, 0.86, 0.95],
                     [0.37, 0.89, 0.70, 0.35, 0.06, 0.22, 0.20, 0.35, 0.89],
                     [0.42, 0.76, 0.56, 0.20, 0.75, 0.78, 1.00, 0.62, 0.95],
                     [0.34, 0.34, 0.70, 0.69, 0.44, 0.22, 0.66, 0.64, 0.80]])

#max pooling: for each P*P subregion output max value
filterSize = 3
stride = 3
#W2 = ((W1 - P) / S) + 1, H2 = ((H1 - P) / S) + 1
outputMap = np.zeros((int((len(inputMap) - filterSize) / stride) + 1, int((len(inputMap[0]) - filterSize) / stride) + 1))
#search for width/stride rows
#search through the row until you hit filterSize
#search for height/stride columns
#combine it with the next filterSize columns
poolValue = 0
for y in range(filterSize):
    for x in range(filterSize):
        poolValue += inputMap[y][x]
outputMap[0][0] = poolValue


outputMapMax = np.zeros((int((len(inputMap) - filterSize) / stride) + 1, int((len(inputMap[0]) - filterSize) / stride) + 1))
outputMapAvg = np.zeros((int((len(inputMap) - filterSize) / stride) + 1, int((len(inputMap[0]) - filterSize) / stride) + 1))
for row in range(int(len(outputMap))):
    for column in range(int(len(outputMap[0]))):
        poolAvg = 0
        poolMax = 0
        for y in range(filterSize):
            for x in range(filterSize):
                poolAvg += inputMap[y+(row*3)][x+(column*3)]
                if poolMax < inputMap[y+(row*3)][x+(column*3)]:
                    poolMax = inputMap[y+(row*3)][x+(column*3)]
        outputMapMax[row][column] = poolMax
        outputMapAvg[row][column] = round(poolAvg / (filterSize * filterSize), 3)
print(f"Max Pooling:\n{outputMapMax}\n")
print(f"Average Pooling:\n{outputMapAvg}")
