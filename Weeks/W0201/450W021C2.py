import numpy as np

def readVectors(file):
    with open(file, 'r') as inMatrix:
        #seperate the vectors
        content = inMatrix.read()
        content = content.strip()
        content = content.split('\n')

        line1 = content[0]
        floatVector = []
        #array of strings
        arrayNums = line1.split(' ')
        for num in arrayNums:
            #array of floats
            floatVector.append(float(num))
        #numpy array
        vector1 = np.array(floatVector)

        line2 = content[1]
        floatVector = []
        arrayNums = line2.split(' ')
        for num in arrayNums:
            floatVector.append(float(num))
        vector2 = np.array(floatVector)
        

        return vector1, vector2

def operations(v1, v2):
    print("V1: ", v1)
    print("V2: ", v2)
    print("V1 + V2: ", np.add(v1,v2))
    print("V1 - V2: ", np.subtract(v1,v2))
    print("2x Scalar 1: ", 2*v1)
    print("2x Scalar 2: ", 2*v2)
    print("Magnitude 1: ", np.linalg.norm(v1))
    print("Magnitude 2: ", np.linalg.norm(v2))

v1, v2 = readVectors(r'C:\Users\obeli\Downloads\matrices2.txt')
operations(v1, v2)