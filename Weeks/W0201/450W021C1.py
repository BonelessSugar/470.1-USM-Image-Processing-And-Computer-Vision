import numpy as np

def readMatricies(file):
    with open(file, 'r') as inMatrix:
        #read whole file, remove leading and trailing white space, 
        # split by \n\n, should split the 2 matricies
        content = inMatrix.read()
        content = content.strip()
        content = content.split('\n\n')

        #first \n split row
        matrix1 = []
        matrix2 = []

        lines1 = content[0].split('\n')
        #first space split column
        for line in lines1:
            row = []
            for num in line.split(' '):
                #string to float
                row.append(float(num))
            matrix1.append(row)

        lines2 = content[1].split('\n')
        #first space split column
        for line in lines2:
            row = []
            for num in line.split(' '):
                row.append(float(num))
            matrix2.append(row)

        matrixNP1 = np.array(matrix1)
        matrixNP2 = np.array(matrix2)
    return matrixNP1, matrixNP2

def calculation(m1, m2):
    #can use a for loop to do all of these, but numpy can do it for us
    #sum all ele
    print("Sum 1: ", np.sum(m1))
    print("Sum 2: ", np.sum(m2))

    #mean all ele
    print("Mean 1: ", np.mean(m1))
    print("Mean 2: ", np.mean(m2))

    #transpose
    print("Transpose 1: \n", np.transpose(m1))
    print("Transpose 2: \n", np.transpose(m2))

    #multiply ele in pos of both matrix
    print("Multiply: \n", np.multiply(m1, m2))

m1, m2 = readMatricies(r'C:\Users\obeli\Downloads\matrices.txt')
calculation(m1, m2)