import os

ROOT = os.path.join('s3://visionlab-members/', os.getenv('USER'))

# Top level directory
DATADIR = os.path.join(ROOT, 'datasets', 'triple-n', 'V1')

PROCESSED = os.path.join(DATADIR, 'Processed')
RAW = os.path.join(DATADIR, 'Raw', 'GoodUnitV2')
OTHERS = os.path.join(DATADIR, 'others')

IMAGEDIR = os.path.join(DATADIR, 'NSD1000_LOC')

# Where to save figures
SAVEDIR = os.path.join(ROOT, 'manifold-dynamics')
