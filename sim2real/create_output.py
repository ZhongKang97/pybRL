import pandas as pd
sim_df = pd.read_csv('sim_action1.txt', delimiter = ' ')
act_df = pd.read_csv('action1.csv', delimiter = ',')

out_df = pd.DataFrame(columns = ['index','s1', 'flh', 'flk', 'frh', 'frk', 's2', 'blh', 'blk', 'brh', 'brk'])
for i, row in act_df.iterrows():
    index = row['index']
    corres_row = sim_df.loc[sim_df['index'] == index]
    out_df = out_df.append(corres_row)
out_df.to_csv('action1_out.csv', index = None)