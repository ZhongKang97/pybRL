import numpy as np  
ARS = np.load("AUG12/policy_4.npy")
A = np.load("AUG12/A_matrix.npy")
B = np.load("AUG12/B_matrix.npy")
finalA = ARS@A
finalB = ARS@B
print(finalA.shape)
print(finalB.shape)
np.savetxt("AUG12/finalA.csv", finalA, delimiter=',')
np.save("AUG12/finalA.npy", finalA)
np.savetxt("AUG12/finalB.csv", finalB, delimiter = ',')
np.save("AUG12/finalB.npy", finalB)

printstr = 'int ARS_matrixA[18][8] = \n'
printstr = printstr + '{\n'
for i in range(18):
    printstr = printstr + '{'
    for j in range(8):
        printstr = printstr + str(int(finalA[i][j]*(10**6)))
        if j != 7:
            printstr = printstr +','
    printstr = printstr +'}'
    if i != 17:
        printstr = printstr+',\n'
printstr =printstr +'\n};\n'

printstr = printstr + 'int ARS_matrixB[18] = {'
for i in range(18):
    printstr = printstr + str(int(finalB[i]*(10**6)))
    if i != 17:
        printstr = printstr + ','
printstr = printstr + '};'

fobj = open("AUG12/final.txt", 'w')
fobj.write(printstr)
fobj.close()
