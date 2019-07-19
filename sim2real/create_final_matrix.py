import numpy as np  
ARS = np.load("ARS_matrix.npy")
A = np.load("A_matrix.npy")
B = np.load("B_matrix.npy")
finalA = ARS@A
finalB = ARS@B
np.savetxt("finalA.csv", finalA)
np.save("finalA.npy", finalA)
np.savetxt("finalB.csv", finalB)
np.save("finalB.npy", finalB)