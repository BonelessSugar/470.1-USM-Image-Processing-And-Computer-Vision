import numpy as np

def createIntArray():
    #ints btw 0 and 100
    return np.random.randint(1, 99, size=(5, 5))

def normalizeValues(intMatrix):
    #set to range 0 to 1
    #(x-min)/(max-min)
    max = np.max(intMatrix)
    min = np.min(intMatrix)
    return (intMatrix - min)/(max - min)

def calculations(normArr):
    #mean, median, std dev
    mean = np.mean(normArr)
    median = np.median(normArr)
    stddev = np.std(normArr)
    return [mean, median, stddev]

def write_results_to_file(filename, array, stats):
    # Write the normalized array and its statistics to a file. 
    with open(filename, 'w') as file:
        np.savetxt(file, array, fmt='%0.4f')
        
        file.write(f"\nStatistics:\nMean: {stats[0]:.4f}\nMedian: {stats[1]:.4f}\nStandard Deviation: {stats[2]:.4f}\n")

myMatrix = createIntArray()
print("Initial Array: \n", myMatrix)
normMatrix = normalizeValues(myMatrix)
print("Normalized Array: \n", normMatrix)
calcs = calculations(normMatrix)
print(f"Mean: {calcs[0]:.4f}")
print(f"Median: {calcs[1]:.4f}")
print(f"Standard Deviation: {calcs[2]:.4f}")
write_results_to_file('array_data.txt', normMatrix, calcs)