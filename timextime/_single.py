import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import imageio.v2 as iio

from _grid import *

# DATA_DIR = ../../datasets/NNN/object_roi_data.pkl
DATA_DIR = './../../datasets/NNN/face_roi_data.pkl'
dat = pd.read_pickle(DATA_DIR)
# ROI_LIST = ['AB3_18_B', 'MB3_12_B', 'AB3_12_B', 'AB3_17_B']
# ROI_LIST = ['Unknown_19_F']# , 'MF1_7_F', 'MF1_8_F', 'MF1_9_F']
# ROI_LIST = ['PITP4_10_O', 'Unknown_6_O', 'MO1s2_5_O', 'Unknown_16_O',
#  'Unknown_26_O', 'AO5_25_O']
ROI_LIST = ['Unknown_19_F']
SAVE_DIR = './../../../buckets/manifold-dynamics/time-time/increasing-scale/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

for roi in ROI_LIST:
    metric = 'correlation'
    out_path = os.path.join(SAVE_DIR, f'{roi}_{metric}.gif')
    out = build_grid_gif(dat, [roi], step=1, k_max=300, metric=metric, out_path=out_path)
    print('Saved:', roi)
