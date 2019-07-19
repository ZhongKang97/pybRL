import numpy as np 
import pandas as pd 

#check Final Results only
finalA = np.load("finalA.npy")
finalB = np.load("finalB.npy")
inp_df = pd.read_csv("action1.csv")
val_df = inp_df.loc[inp_df['index'].isin([0.0, 199.0, 100.0])]
val_df = val_df.drop(columns = ['index'])
for i, row in val_df.iterrows():
    print( finalA @ row.values + finalB)

#check A and B
# A = np.load("A_matrix.npy")
# B = np.load("B_matrix.npy")
# inp_df = pd.read_csv("action1.csv")
# val_df = inp_df.loc[inp_df['index'].isin([0.0, 199.0, 100.0])]
# val_df = val_df.drop(columns = ['index'])
# for i, row in val_df.iterrows():
#     print( A @ row.values + B)

#check ARS with A and B , Bezier Points

# ARS = np.load("ARS_matrix.npy")
# A = np.load("A_matrix.npy")
# B = np.load("B_matrix.npy")
# inp_df = pd.read_csv("action1.csv")
# val_df = inp_df.loc[inp_df['index'].isin([0.0, 199.0, 100.0])]
# val_df = val_df.drop(columns = ['index'])
# for i, row in val_df.iterrows():
#     print('Beziers: ', ARS @ (A @ row.values + B))
