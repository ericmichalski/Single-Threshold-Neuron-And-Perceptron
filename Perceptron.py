from cgi import test
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cmap
import copy

beforeTrainingWeights = []

# Training function
def simulateThreshold(inputFile = "dataFiles/random_01.txt", training = True, weights = [], initialWeights = []):
    combinedF = open(inputFile, "r")

    # Learning rate
    N = 0.01

    # If there are no weights (initial epoch)
    if (weights == []):
        # Set random weights
        weights.append(1) # Add bias weight "1" to start of weights wi0
        for x in range(0,784):
            weights.append(random.uniform(0, 0.5))
        if (training):
            global beforeTrainingWeights
            beforeTrainingWeights = weights.copy()

    yList = []
    # Parse input text file by splitting each line and whitespace and adding each element into a list
    for line in combinedF:
        netInput = 0
        data = line[2:].split()

        # Calculate net input
        for x in range(0,784):
            newVal = weights[x+1] * float(data[x])
            netInput = netInput + newVal
        netInput = netInput + float(weights[0]) # Add bias weight to net input

        # Determine output y(t)
        if (netInput > 1):
            y = 1
        else:
            y = 0
        yList.append(y)

        if (training):
            # Adjust weights based on output and error (desired - actual)
            desiredY = line[0]
            for k in range(0, 784):
                weights[k+1] = weights[k+1] + ((N * (int(desiredY) - y)) * float(data[k]))
            weights[0] = weights[0] + (N * (int(desiredY) - y)) # Adjust bias based on errors
    return weights, yList

afterTrainingWeights = []
precisionPre = 0
recallPre = 0
f1Pre = 0

# Train the model
def Train():
    epochs = 40
    weights = []
    finalYs = []
    errorFraction = []
    global afterTrainingWeights
    afterTrainingWeights, finalYs = simulateThreshold("dataFiles/random_01.txt", True, copy.deepcopy(weights)) # First epoch
    errorFraction.append(GetErrorFraction("dataFiles/random_01.txt", finalYs))

    # For each initial output value, compare it with the actual value to determine pre-classification (true/false pos/neg)
    actualValues = []
    initialFile = open("dataFiles/random_01.txt", "r")
    # Parse input file and get each image's actual value into an array
    for line in initialFile:
        actualValues.append(line[0])
    j = 0
    classificationInOrder = []
    trueOnes = 0
    trueZeros = 0
    falseOnes = 0
    falseZeros = 0

    for each in finalYs:
        if (each == int(actualValues[j])):
            if (each == 1):
                classificationInOrder.append("T1")
                trueOnes += 1
            else:
                classificationInOrder.append("T0")
                trueZeros += 1
        else:
            if (each == 1):
                classificationInOrder.append("F1")
                falseOnes += 1
            else:
                classificationInOrder.append("F0")
                falseZeros += 1
        j += 1
    
    global precisionPre, recallPre, f1Pre
    precisionPre = trueOnes / (trueOnes + falseOnes)
    recallPre = trueOnes / (trueOnes + falseZeros)
    f1Pre = 2 * ((precisionPre * recallPre) / (precisionPre + recallPre))

    # Finish training
    for x in range(0,epochs-1):
        afterTrainingWeights, finalYs = simulateThreshold("dataFiles/random_01.txt", True, copy.deepcopy(afterTrainingWeights)) # Rest of epochs
        errorFraction.append(GetErrorFraction("dataFiles/random_01.txt", copy.deepcopy(finalYs)))

    return errorFraction

def GetErrorFraction(inputFile, finalYs):
    combinedF = open(inputFile, "r")
    actualValues = []
    # Parse input file and get each image's actual value into an array
    for line in combinedF:
        actualValues.append(line[0])
    
    trueOnes = 0
    trueZeros = 0
    falseOnes = 0
    falseZeros = 0
    j = 0
    for each in finalYs:
        if (each == int(actualValues[j])):
            if (each == 1):
                trueOnes += 1
            else:
                trueZeros += 1
        else:
            if (each == 1):
                falseOnes += 1
            else:
                falseZeros += 1
        j += 1

    return ((falseZeros + falseOnes) / 800)

# Test the model
def Test():
    # Get actual values once since the input is the same for all the thresholds
    combinedF = open("dataFiles/testSet_01.txt", "r")
    actualValues = []
    # Parse input file and get each image's actual value into an array
    for line in combinedF:
        actualValues.append(line[0])

    # Run through the threshold values from 0 to 40 in increments of 1
    outWeights = []
    finalYs = []
    precisionList = []
    recallList = []
    f1List = []
    trueOnesL = []
    falseOnesL = []   
    
    errorFraction = []
    epochs = 40
    outWeights, finalYs = simulateThreshold("dataFiles/testSet_01.txt", False, copy.deepcopy(afterTrainingWeights)) # First epoch
    errorFraction.append(GetErrorFraction("dataFiles/testSet_01.txt", copy.deepcopy(finalYs)))

    for x in range(0,epochs-1):
        outWeights, finalYs = simulateThreshold("dataFiles/testSet_01.txt", False, copy.deepcopy(afterTrainingWeights)) # Rest of epochs
        errorFraction.append(GetErrorFraction("dataFiles/testSet_01.txt", copy.deepcopy(finalYs)))


    # For each final output value, compare it with the actual value to determine classification (true/false pos/neg)
    j = 0
    classificationInOrder = []
    trueOnes = 0
    trueZeros = 0
    falseOnes = 0
    falseZeros = 0

    for each in finalYs:
        if (each == int(actualValues[j])):
            if (each == 1):
                classificationInOrder.append("T1")
                trueOnes += 1
            else:
                classificationInOrder.append("T0")
                trueZeros += 1
        else:
            if (each == 1):
                classificationInOrder.append("F1")
                falseOnes += 1
            else:
                classificationInOrder.append("F0")
                falseZeros += 1
        j += 1
    
    precision = trueOnes / (trueOnes + falseOnes)
    recall = trueOnes / (trueOnes + falseZeros)
    f1 = 2 * ((precision * recall) / (precision + recall))
    
    precisionList.append(precision)
    recallList.append(recall)
    f1List.append(f1)
    trueOnesL.append(trueOnes)
    falseOnesL.append(falseOnes)

    plt.subplot(2, 2, 1)
    X = ['Precision','Recall','F1']
    X_axis = np.arange(len(X))
    preStats = [precisionPre, recallPre, f1Pre]
    postStats = [precision, recall, f1]
    plt.bar(X_axis - 0.2, preStats, 0.4, label = 'Pre')
    plt.bar(X_axis + 0.2, postStats, 0.4, label = 'Post')
    plt.xticks(X_axis, X)
    plt.xlabel('Group')
    plt.ylabel('Values')
    plt.title("Precision, Recall, and F1 v Threshold")
    plt.legend(loc='best')

    plt.subplot(2, 2, 3)
    beforeTrainingWeights.pop(0)
    arrA = np.array(beforeTrainingWeights)
    # Convert 1D array to a 2D numpy array of 28 rows and 28 columns
    arrA2D = np.reshape(arrA, (28, 28))
    plt.title("Heatmap of Initial Weights")
    plt.xlabel('X Axis Pixels')
    plt.ylabel('Y Axis Pixels')
    plt.imshow(arrA2D, cmap=cmap.hot)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    copyAfter = copy.deepcopy(afterTrainingWeights)
    copyAfter.pop(0)
    arrB = np.array(copyAfter)
    # Convert 1D array to a 2D numpy array of 28 rows and 28 columns
    arrB2D = np.reshape(arrB, (28, 28)).transpose()
    plt.title("Heatmap of Post-Training Weights")
    plt.xlabel('X Axis Pixels')
    plt.ylabel('Y Axis Pixels')
    plt.imshow(arrB2D, cmap=cmap.hot)
    plt.colorbar()
    plt.show()

    return errorFraction

# Challenge the model
def Challenge():
    # Get actual values of challenge set
    combinedF = open("dataFiles/challengeSet.txt", "r")
    actualValues = []
    # Parse input file and get each image's actual value into an array
    for line in combinedF:
        actualValues.append(line[0])
    # Run through a singular instance of the chosen "best" threshold
    outWeights = []
    finalys = []
    threshold = 20
    outWeights, finalys = simulateThreshold("dataFiles/challengeSet.txt", False, copy.deepcopy(afterTrainingWeights))

    # Count the number of 2-9's as 1's or 0's and output as a 2 x 8 matrix
    j = 0
    NumOnes = [0, 0, 0, 0, 0, 0, 0, 0]
    NumZeros = [0, 0, 0, 0, 0, 0, 0, 0]
    for each in finalys:
        if (each == 1):
            if (int(actualValues[j]) == 2):
                NumOnes[0] = NumOnes[0] + 1
            if (int(actualValues[j]) == 3):
                NumOnes[1] = NumOnes[1] + 1
            if (int(actualValues[j]) == 4):
                NumOnes[2] = NumOnes[2] + 1
            if (int(actualValues[j]) == 5):
                NumOnes[3] = NumOnes[3] + 1
            if (int(actualValues[j]) == 6):
                NumOnes[4] = NumOnes[4] + 1
            if (int(actualValues[j]) == 7):
                NumOnes[5] = NumOnes[5] + 1
            if (int(actualValues[j]) == 8):
                NumOnes[6] = NumOnes[6] + 1
            if (int(actualValues[j]) == 9):
                NumOnes[7] = NumOnes[7] + 1
        else:
            if (int(actualValues[j]) == 2):
                NumZeros[0] = NumZeros[0] + 1
            if (int(actualValues[j]) == 3):
                NumZeros[1] = NumZeros[1] + 1
            if (int(actualValues[j]) == 4):
                NumZeros[2] = NumZeros[2] + 1
            if (int(actualValues[j]) == 5):
                NumZeros[3] = NumZeros[3] + 1
            if (int(actualValues[j]) == 6):
                NumZeros[4] = NumZeros[4] + 1
            if (int(actualValues[j]) == 7):
                NumZeros[5] = NumZeros[5] + 1
            if (int(actualValues[j]) == 8):
                NumZeros[6] = NumZeros[6] + 1
            if (int(actualValues[j]) == 9):
                NumZeros[7] = NumZeros[7] + 1
        j += 1

    matrix = np.column_stack((NumOnes, NumZeros))
    print("Count of the numbers 2 through 9 (top to bottom) identified as ones and zeros (left to right)")
    print(matrix)

trainError = Train()
testError = Test()

xList = list(range(0,40))
plt.plot(xList, trainError, label = "Train Error")
plt.plot(xList, testError, label = "Test Error")
plt.xlabel('Epochs')
plt.ylabel('Error Amount')
plt.title("Error Fraction for Training and Testing")
plt.legend(loc='best')
plt.show()

Challenge()