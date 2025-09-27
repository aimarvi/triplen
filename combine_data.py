import glob
import pandas as pd

dfs = []
for fname in sorted(glob.glob('../datasets/NNN/only_raster_data_batch*.pkl')):
    dfs.append(pd.read_pickle(fname))
final_df = pd.concat(dfs, ignore_index=True)

final_df.to_pickle('../datasets/NNN/only_raster_data_full.pkl')
