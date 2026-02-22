import os

# Root directory
# $USER should match aws bucket. change manually if necessary
ROOT = os.path.join('s3://visionlab-members/', os.getenv('USER'))

# Top level directory
DATADIR = os.path.join(ROOT, 'datasets', 'triple-n', 'V1')

# Basic data subdirs
PROCESSED = os.path.join(DATADIR, 'Processed')
RAW = os.path.join(DATADIR, 'Raw', 'GoodUnitV2')
OTHERS = os.path.join(DATADIR, 'others')

# Stimuli
IMAGEDIR = os.path.join(DATADIR, 'NSD1000_LOC')

# Where to save figures
SAVEDIR = os.path.join(ROOT, 'manifold-dynamics')
