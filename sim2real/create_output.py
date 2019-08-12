import pandas as pd
sim_df = pd.read_csv('AUG12/trot_sim_data.csv')
act_df = pd.read_csv('AUG12/trot_real_data.csv')
out_df = pd.DataFrame(columns = ['index', 'flh', 'flk', 'frh', 'frk', 'blh', 'blk', 'brh', 'brk'])
for i, row in act_df.iterrows():
    index = row['index']
    corres_row = sim_df.loc[sim_df['index'] == index]
    out_df = out_df.append(corres_row)
out_df.to_csv('AUG12/regr_out.csv', index = None)