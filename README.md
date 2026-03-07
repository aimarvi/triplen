# Manifold Dynamics (Triple-N V1)

This repo analyzes macaque IT neural responses from the Triple-N dataset, with a focus on **time-time representational geometry** (how neural geometry changes over time).

The practical pipeline is:

`GoodUnit*.mat + Processed_ses*.mat + exclude_area.xls`  
`-> roi-uid.csv`  
`-> session ROI raster (units x time x images x repeats)`  
`-> trial-averaged PSTH / tuning RDM`  
`-> ED2 and cross-validated geometry analyses`

## Data Format (What You Actually Use)

## 1) `GoodUnit_*.mat` (raw spike responses)
Key fields used in this repo:
- `GoodUnitStrc.Raster`: per-unit spike raster over time and valid trials
- `meta_data.trial_valid_idx`: image ID for each trial
  - `0` = invalid trial (excluded)
  - `1..1000` = NSD images
  - `1001..1072` = localizer images
- Time axis is typically `-50..400 ms` at `1 ms` resolution (`450` bins)

## 2) `Processed_ses*.mat` (per-session summary metadata)
Key field used in ROI extraction:
- `pos`: unit position along probe/shank

This file is used with ROI depth limits (`y1`, `y2`) to keep only units in the selected ROI.

## 3) `exclude_area.xls` (manual ROI definitions)
Used to build `roi-uid.csv`.

## ROI and Naming Conventions

`roi-uid.csv` is the core mapping table used throughout the analysis.

Columns in `roi-uid.csv`:
- `uid`
- `y1`
- `y2`

`uid` format:
- `SesIdx.RoiIndex.AREALABEL.Categoty`
- Example: `18.19.Unknown.F`

Meaning:
- `SesIdx`: session ID
- `RoiIndex`: ROI index within session mapping
- `AREALABEL`: area label
- `Categoty`: category label (kept as spelled in source table)

Useful forms:
- **ROI UID (single session)**: `11.04.MO1s1.O`
- **ROI key (all sessions for one ROI)**: `04.MO1s1.O`

## Session Organization

Session pairing is resolved automatically by matching filenames:
- `GoodUnit_<date>..._g<idx>.mat`
- `Processed_ses<session>_<date>_..._<idx>.mat`

In `io_matlab_s3.py`, files are matched by `(date, idx)` and then sorted by `ses` number.

## Main Data Shapes

After ROI extraction (`session_raster_extraction.py`):
- `raster`: `(units, time, images, repeats)`
- Typical dimensions:
  - `time = 450`
  - `images = 1072`
  - `repeats = variable, NaN-padded to max repeats per image`

Common downstream views:
- Trial-averaged PSTH: `(units, time, images)`
- Time-time tuning RDM: `(time_window, time_window)`

## Most Relevant `src/manifold_dynamics` Files

- `paths.py`
  - Central S3 path definitions (raw, processed, others, stimuli)

- `io_matlab_s3.py`
  - Lists/matches raw and processed files by session
  - Loads MATLAB files from S3 (`v7.3` via `h5py`, `v5` via `scipy.io.loadmat`)

- `unique_label.py`
  - Builds `roi-uid.csv` from `exclude_area.xls`

- `session_raster_extraction.py`
  - Core conversion from session files to ROI raster tensor
  - Applies ROI depth filter using `y1/y2`

- `neural_utils.py`
  - Raster loading, trial binning (e.g., 20 ms), responsive mask caching
  - General neural utility functions

- `spike_response_stats.py`
  - Unit response significance test (`is_responsive`)

- `tuning_utils.py`
  - Core representational geometry utilities
  - `tuning_rdm()` is the main time-time RDM function
  - ED metrics (`ED1`, `ED2`)

- `crossval.py`
  - CLI for cross-validated time-time analysis from ROI UID or ROI key

## Minimal Workflow

1. Build/refresh ROI mapping:
- Run `unique_label.py` to produce `roi-uid.csv`

2. Extract ROI-session raster:
- Use `session_raster_extraction.extract_session_raster(roi_uid)`

3. Compute time-time geometry:
- Use `tuning_utils.tuning_rdm(...)`
- Compute ED with `tuning_utils.ED2(...)`

4. Run CV analysis:
- Use `crossval.py` with either a 4-part UID or 3-part ROI key

