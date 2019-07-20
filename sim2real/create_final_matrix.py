import numpy as np  
ARS = np.load("ARS_matrix.npy")
A = np.load("A_matrix.npy")
B = np.load("B_matrix.npy")
finalA = ARS@A
finalB = ARS@B
np.savetxt("finalA.csv", finalA, delimiter=',')
np.save("finalA.npy", finalA)
np.savetxt("finalB.csv", finalB, delimiter = ',')
np.save("finalB.npy", finalB)

printstr = 'int ARS_matrixA[10][10] = \n'
printstr = printstr + '{\n'
for i in range(10):
    printstr = printstr + '{'
    for j in range(10):
        printstr = printstr + str(int(finalA[i][j]*(10**6)))
        if j != 9:
            printstr = printstr +','
    printstr = printstr +'}'
    if i != 9:
        printstr = printstr+',\n'
printstr =printstr +'\n};\n'

printstr = printstr + 'int ARS_matrixB[10] = {'
for i in range(10):
    printstr = printstr + str(int(finalB[i]*(10**6)))
    if i != 9:
        printstr = printstr + ','
printstr = printstr + '};'

fobj = open("final.txt", 'w')
fobj.write(printstr)
fobj.close()
