from sklearn import linear_model
import pandas as pd
import numpy as np
reg = linear_model.LinearRegression()

inp_df = pd.read_csv("AUG12/trot_real_data.csv")
print(len(inp_df))
out_df = pd.read_csv("AUG12/regr_out.csv")
print(len(out_df))
sim_df = pd.read_csv("AUG12/trot_sim_data.csv")
val_df = inp_df.loc[inp_df['index'].isin([0, 49])]
weight = 1
inp_df = pd.concat([inp_df]+[val_df]*weight)
val_out_df = pd.DataFrame(columns = ['index','flh','flk','frh','frk','blh','blk','brh','brk'])
for i, row in val_df.iterrows():
    index = row['index']
    corres_row = sim_df.loc[sim_df['index'] == index]
    val_out_df = val_out_df.append(corres_row)
val_df = val_df.drop(columns = ['index'])
val_out_df = val_out_df.drop(columns = ['index'])
inp_df = inp_df.drop(columns = ['index'])
inputs = inp_df.values
out_df = out_df.drop(columns = ['index'])
out_df = pd.concat([out_df] + [val_out_df]*weight)
outputs = out_df.values
# print(inp_df, out_df)
reg.fit(inputs, outputs)
print("Done regression, starting validation")

val_df = val_df.values
val_out_df = val_out_df.values
mean_error = np.mean(np.abs(reg.predict(val_df) - val_out_df))
print("mean error: ", mean_error)
# print("mean score: ", reg.score(val_df, val_out_df))


# print("Now testing with alternate model")

# inp_df = pd.read_csv("action2.csv")
# inp_df = inp_df.loc[inp_df['index'].isin([0.0, 199.0, 100.0])]
# out_df = pd.DataFrame(columns = ['index','s1','flh','flk','frh','frk','s2','blh','blk','brh','brk'])

# for i, row in inp_df.iterrows():
#     index = row['index']
#     corres_row = sim_df.loc[sim_df['index'] == index]
#     out_df = out_df.append(corres_row)

# inp_df = inp_df.drop(columns = ['index'])
# out_df = out_df.drop(columns = ['index'])
# inp_df = inp_df.values
# out_df = out_df.values

# mean_error = np.mean(np.abs(reg.predict(inp_df) - out_df))
# # print(reg.predict(inp_df))
# # print(reg.coef_ @ inp_df[0] + reg.intercept_)
# print("mean error: ", mean_error)
# # print("mean score: ", reg.score(inp_df, out_df))
np.save('AUG12/A_matrix.npy', reg.coef_)
np.save('AUG12/B_matrix.npy', reg.intercept_)

