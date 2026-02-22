import os
import pandas as pd
import manifold_dynamics.PATHS as PTH

'''
make a unique ID for each fMRI-defined ROI
ROIs have specific values (y1, y2) along shank

save to s3://{USER}/datasets/triple-n/others/roi-uid.csv
'''

# load spreadsheet (change to read_excel if .xlsx)
df = pd.read_excel(os.path.join(PTH.OTHERS, 'exclude_area.xls'))
print(df.columns)

savepath = os.path.join(PTH.OTHERS, 'roi-uid.csv')

# choose the minimal identifying columns
id_cols = ['SesIdx', 'RoiIndex', 'AREALABEL', 'Categoty']
df['SesIdx'] = df['SesIdx'].map(lambda x: f'{int(x):02d}')
df['RoiIndex'] = df['RoiIndex'].map(lambda x: f'{int(x):02d}')

# create unique identifier
df['uid'] = (
    df[id_cols]
    .astype(str)
    .agg('.'.join, axis=1)
)

# keep only required columns
out_df = df[['uid', 'y1', 'y2']]

# save to new file
out_df.to_csv(savepath, index=False)
print('saved to:', savepath)
