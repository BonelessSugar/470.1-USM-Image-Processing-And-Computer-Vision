import numpy as np

inputMap = np.array([[[0.40, 0.13, 0.53, 0.35, 0.04],
                      [0.50, 0.87, 0.80, 0.24, 0.88],
                      [0.69, 0.45, 0.02, 0.75, 0.52],
                      [0.24, 0.76, 0.16, 0.70, 0.08],
                      [0.09, 0.90, 0.41, 0.27, 0.08]],

                     [[0.86, 0.98, 0.46, 0.80, 0.48],
                      [0.43, 0.54, 0.66, 0.75, 0.90],
                      [0.43, 0.47, 0.56, 0.96, 0.06],
                      [0.95, 0.03, 0.99, 0.64, 0.82],
                      [0.95, 0.61, 0.14, 0.03, 0.75]],
                
                     [[0.74, 0.32, 0.89, 0.33, 0.40],
                      [0.63, 0.61, 0.10, 0.47, 0.10],
                      [0.27, 0.55, 0.99, 0.51, 0.23],
                      [0.07, 0.51, 0.91, 0.32, 0.30],
                      [0.52, 0.36, 0.25, 0.91, 0.94]],
                
                     [[0.71, 0.71, 0.70, 0.62, 0.32],
                      [0.88, 0.06, 0.17, 0.56, 0.04],
                      [0.42, 0.18, 0.78, 0.43, 0.77],
                      [0.94, 0.14, 0.25, 0.13, 0.61],
                      [0.92, 0.99, 0.23, 0.25, 0.92]]])



kernelOne = np.array([[[0.32, 0.05, 0.43],
                       [0.91, 0.37, 0.60],
                       [0.23, 0.52, 0.40]],
              
                      [[0.31, 0.91, 0.47],
                       [0.49, 0.42, 0.92],
                       [0.64, 0.49, 0.62]],
              
                      [[0.58, 0.41, 0.17],
                       [0.39, 0.45, 0.77],
                       [0.11, 0.58, 0.79]],
              
                      [[0.92, 0.42, 0.93],
                       [0.38, 0.15, 0.52],
                       [0.74, 0.45, 0.53]]])



kernelTwo = np.array([[[0.07, 0.01, 0.59],
                       [0.02, 0.94, 0.68],
                       [0.78, 0.64, 0.73]],
              
                      [[0.14, 0.55, 0.42],
                       [0.90, 0.46, 0.80],
                       [0.65, 0.96, 0.81]],
              
                      [[0.32, 0.27, 0.45],
                       [0.76, 0.52, 0.13],
                       [0.52, 0.95, 0.88]],
              
                      [[0.34, 0.69, 0.81],
                       [0.10, 0.44, 0.03],
                       [0.77, 0.06, 0.67]]])

#zero pad the input map: z,y,x
paddedMap = np.zeros(((4,7,7)))
for i in range(0,4):
    paddedMap[i] = np.pad(inputMap[i], ((1,1)))

#calculate top left corner matrix multiplication and sum
kernelMapOne = np.zeros((4,3,3))
kernelOutOne = 0
kernelMapTwo = np.zeros((4,3,3))
kernelOutTwo = 0
xCount = 0
yCount = 0

for z in range(len(paddedMap)):
    xCount = 0
    for x in range(len(paddedMap[0])):
        yCount = 0
        if x == xCount + len(kernelOne[0]):
            xCount += 1
            break
        for y in range(len(paddedMap[0][0])):
            if y == yCount + len(kernelOne[0][0]):
                yCount += 1
                break
            kernelMapOne[z][x][y] = paddedMap[z][x + xCount][y + yCount] * kernelOne[z][x][y]
            kernelOutOne += kernelMapOne[z][x][y]
            kernelMapTwo[z][x][y] = paddedMap[z][x + xCount][y + yCount] * kernelTwo[z][x][y]
            kernelOutTwo += kernelMapTwo[z][x][y]

#now do it for all of the spaces
outputMap = np.zeros((2,5,5))
for row in range(len(outputMap[0])):
    for column in range(len(outputMap[0][0])):
        kernelOutOne = 0
        kernelOutTwo = 0
        for z in range(len(paddedMap)):
            xCount = row
            kernelX = 0
            for x in range(xCount,len(paddedMap[0])):
                yCount = column
                kernelY = 0
                if x == xCount + len(kernelOne[0]):
                    xCount += 1
                    break
                for y in range(yCount,len(paddedMap[0][0])):
                    if y == yCount + len(kernelOne[0][0]):
                        yCount += 1
                        break
                    kernelOutOne += paddedMap[z][x][y] * kernelOne[z][kernelX][kernelY]
                    kernelOutTwo += paddedMap[z][x][y] * kernelTwo[z][kernelX][kernelY]
                    kernelY += 1
                kernelX += 1
        outputMap[0][row][column] = kernelOutOne
        outputMap[1][row][column] = kernelOutTwo
print(outputMap)
