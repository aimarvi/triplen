import os
import pandas as pd

DATA_DIR = '../../datasets/NNN/'

# load in all data and consolidate into one big df
df = pd.DataFrame()
for CAT in ['face', 'body', 'object']:
    cat_df = pd.read_pickle(os.path.join(DATA_DIR, f'{CAT}_ed.pkl'))
    no_nans = cat_df.groupby('ROI').filter(lambda x: not x.isna().any().any())
    df = pd.concat([df, no_nans])

# extract per-ROI global ED baseline
g = (
    df[df['Method'] == 'global']
    .loc[:, ['ROI', 'ED']]
    .rename(columns={'ED': 'ED_global'})
)
df = df.merge(g, on='ROI', how='left')

# calculate % change (relative to global) and group by ROI (for bootstrap)
df['percent_change'] = (df['ED'] - df['ED_global']) / df['ED_global'] * 100

df_rel = df[df['Method'] != 'global'].copy()
roi_agg = (df_rel.groupby(['ROI', 'Method'])['percent_change'].median().reset_index())

idx = roi_agg.groupby('Method')['percent_change']

summary = pd.DataFrame({
    'min_val': idx.min(),
    'max_val': idx.max(),
    'mean_val': idx.mean(),
    'median_val': idx.median(),
    'std_val': idx.std(),
})
print(summary)
