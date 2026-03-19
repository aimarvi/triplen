# MEMORIES.md - Neural Geometry Dynamics

## Abstract Context
This project analyzes how neural population geometry changes over time for preferred image sets versus matched controls. Recent work focused on explaining time-time ED compression using eigenspectrum analyses, temporal PCA of RDM trajectories, and trajectory-constraint metrics.

## Key Insights
- The fixed ROI set used for current hypothesis testing is:
  - `19.Unknown.F`
  - `07.MF1.F`
  - `08.MF1.F`
  - `09.MF1.F`
- Instantaneous eigenspectrum analysis did **not** support the idea that top-k ED compression comes from stronger variance concentration into the first few PCs. For the 4-ROI set, top-k was often more distributed than random-k at a single timepoint.
- Temporal PCA of the RDM trajectory **did** support a lower-dimensional temporal-subspace account:
  - higher temporal `PC1` and `PC3` variance fractions for top-k
  - fewer temporal PCs needed to explain `80%` and `90%` variance
- Constrained-trajectory analysis also supported a stronger interpretation:
  - lower `L_norm`
  - higher autocorrelation AUC
  - longer autocorrelation half-decay
  - slightly lower turning angle
- Clean current interpretation:
  - Preferred image sets produce lower-dimensional and more temporally coherent trajectories of representational geometry.

## Data Shapes / API Patterns
- Canonical ROI loader:
  - `manifold_dynamics.neural_utils.significant_trial_raster(roi_uid, alpha, bin_size_ms)`
  - returns `(units, time, images, trials)`
  - accepts either:
    - 4-part UID: `SesIdx.RoiIndex.AREALABEL.Categoty`
    - 3-part ROI key: `RoiIndex.AREALABEL.Categoty`
- Trial-averaged array shape used by most analyses:
  - `X = np.nanmean(raster_4d, axis=3)` -> `(units, time, images)`
- Standard onset/windows from `spike_response_stats.py`:
  - `ONSET_TIME = 50`
  - `BASE_WIN_MS = (-50, 0)`
  - `RESP_WIN_MS = (50, 220)`
- Standard cropped time window for temporal geometry analyses:
  - `tstart=100`, `tend=350`
- `top-k` defaults should come from:
  - `f"{pth.OTHERS}/topk_vals.pkl"`
- `treves_rolls_sparsity(X, axis)` now lives in:
  - `src/manifold_dynamics/tuning_utils.py`
  - `axis=1` on `(units, images)` gives one value per unit
  - `axis=0` gives one value per image

## Session Log (2026-03-17)
- Added `treves_rolls_sparsity()` to `src/manifold_dynamics/tuning_utils.py`.
- Built `dynamic_modes/sparsity.py` as a single-ROI ad hoc script:
  - odd/even split cross-validation
  - top-k versus bootstrapped random-k
  - normalized image and population sparsity
  - 2x2 figure with PSTH overlays
- Built `alexnet/locality_test.py`:
  - compares `top-k`, `preferred local`, `general local`, `random`
  - `general local` seeds are sampled from non-top images
- Verified `locality_test.py` versus `alexnet/neighbor_scales.py` for `19.Unknown.F`:
  - `ED_topk` matched exactly
  - preferred-local / neighbor ED matched exactly
  - random ED differed because `locality_test.py` consumes RNG earlier to choose general-local seeds
- Built `alexnet/locality_layers.py`:
  - sweeps AlexNet layers from `alexnet/layers.txt`
  - compares preferred-local and general-local ED by layer
  - horizontal references for `top-k` and `random`
  - uses SEM shading, not 95% CI
  - flattens conv-layer activations to `(images, features)`
  - skips unavailable layer keys gracefully
- Verified `alexnet/locality_layers.py` runs correctly for `19.Unknown.F` and saves:
  - `s3://visionlab-members/amarvi/manifold-dynamics/neighbors/locality_layers/19.Unknown.F.png`
  - `s3://visionlab-members/amarvi/manifold-dynamics/neighbors/locality_layers/19.Unknown.F.pkl`
- Resolved the AMC3 label mismatch by updating `topk_vals.pkl`:
  - old inconsistent key: `28.AMC3.F`
  - canonical key now: `28.AMC3.O`
  - this matches `roi-uid.csv` and the `single-session-raster/` metadata.
  - verified on 2026-03-19 that the actual S3 object and refreshed local cache now contain `28.AMC3.O`
    with value `{'k': 40, 'metric': 274.21972956592697}`

## Session Log (2026-03-16)
- Built `interpret_temporal_pca.py`, `interpret_temporal_pca_01.py`, `interpret_temporal_pca_02.py`.
- Temporal PCA result for the 4-ROI set:
  - `pc1_variance`: top `0.2605`, random `0.1508`, delta `+0.1097`
  - `pc3_variance`: top `0.4722`, random `0.3317`, delta `+0.1405`
  - `n_components_80`: top `14.25`, random `19.06`, delta `-4.81`
  - `n_components_90`: top `27.25`, random `36.49`, delta `-9.24`
- Built `interpret_trajectory_constraints.py`, `_01.py`, `_02.py`.
- Constrained-trajectory result for the 4-ROI set:
  - `L_total`: top `158.2460`, random `162.0486`, delta `-3.8026`
  - `L_norm`: top `66.3926`, random `71.5276`, delta `-5.1350`
  - `AUC_autocorr`: top `37.9713`, random `23.7554`, delta `+14.2159`
  - `half_decay_lag`: top `30.75`, random `23.06`, delta `+7.69`
  - `mean_turn_angle`: top `1.5804`, random `1.5954`, delta `-0.0150`
- Saved 4-ROI temporal PCA figures to `~/Downloads/interpret_temporal_pca_*four_roi_sd_norm.png`.
- Saved 4-ROI trajectory-constraint figures to `~/Downloads/interpret_constraints_*four_roi_sd_norm.png`.

## Session Log (2026-03-14)
- Built `interpret_eigenspectra.py`, `_01.py`, `_02.py`.
- Lighter payload rule used there:
  - save top-k result
  - save aggregated random summary
  - do not save every bootstrap/random draw
- Four-ROI eigenspectrum result went against the simple instantaneous-concentration hypothesis:
  - `pc1_fraction`: top `0.1354`, random `0.1687`, delta `-0.0333`
  - `top3_fraction`: top `0.2483`, random `0.2883`, delta `-0.0400`
  - `tail_fraction`: top `0.7517`, random `0.7117`, delta `+0.0400`
  - `spectral_entropy`: top `3.3959`, random `3.2946`, delta `+0.1013`
  - `ed`: top `23.3771`, random `19.8959`, delta `+3.4813`

## Useful Paths
- Core analysis utils:
  - `src/manifold_dynamics/neural_utils.py`
  - `src/manifold_dynamics/tuning_utils.py`
  - `src/manifold_dynamics/spike_response_stats.py`
- Recent scripts:
  - `interpret_eigenspectra_01.py`
  - `interpret_eigenspectra_02.py`
  - `interpret_temporal_pca_01.py`
  - `interpret_temporal_pca_02.py`
  - `interpret_trajectory_constraints_01.py`
  - `interpret_trajectory_constraints_02.py`
  - `dynamic_modes/sparsity.py`
  - `alexnet/locality_test.py`
  - `alexnet/locality_layers.py`
- AlexNet layer map:
  - `alexnet/layers.txt`
- Saved AlexNet activations:
  - `s3://visionlab-members/amarvi/manifold-dynamics/alexnet/alexnet_acts.pkl`

## Tips for Future Agents
- Repo style patterns matter more than generic cleanup preferences.
  - There are two accepted script styles in this repo and they should stay distinct:
    - ad hoc scripts:
      - top-of-file configuration constants
      - `vprint()` at module scope
      - linear execution, no `main()` / `argparse`
    - CLI scripts:
      - `main() + argparse`
      - `vprint()` defined inside `main()`
  - `vprint()` is standard in both styles and should not be removed under a generic “avoid helper functions” cleanup pass.
  - Prefer compact, linear scripts with explicit steps over introducing extra helper functions unless a helper materially clarifies shared logic.
  - Standard S3 reads use `vst.fetch(...)`; standard S3 writes use `fsspec.open(...)`.
  - When `--save` is present, the common pattern is:
    - save canonical analysis output to `pth.SAVEDIR/...`
    - optionally also save a local copy to `~/Downloads/` for inspection
  - For ad hoc scripts with `SAVE = True`, the common pattern is:
    - save a local repo artifact if relevant
    - save canonical output to `pth.SAVEDIR/...`
    - save a convenience copy to `~/Downloads/`
  - Follow the concrete style of nearby repo scripts such as:
    - `timextime/crossval.py`
    - `timextime/ed_main.py`
    - `alexnet/locality_test.py`
    - `alexnet/plot_centroids.py`
  - Consistency across the repo is a priority. When changing a script, match the surrounding file family rather than imposing a new generic style.
- The saved AlexNet activation file does **not** contain every layer listed in `alexnet/layers.txt`.
  - Available keys:
    - `features.2`
    - `features.5`
    - `features.7`
    - `features.9`
    - `features.11`
    - `features.12`
    - `classifier.2`
    - `classifier.5`
  - Missing keys such as `classifier.1`, `classifier.4`, and `classifier.6` should be skipped cleanly.
- Convolutional layer activations arrive as spatial tensors like `(images, channels, height, width)` and must be flattened to `(images, features)` before nearest-neighbor calculations.
- `visionlab_utils.storage.fetch()` is the standard read path for S3-backed assets in this repo.
- Many long S3-backed analysis runs are quiet until completion because they only print at the end. Before assuming a stall, check whether the expected output file already landed.
- `np.trapz` was unavailable in this environment; `np.trapezoid` worked.
- When comparing new raw-raster time-time ED scripts against `../datasets/NNN/face_ed.pkl`, note the time-window mismatch:
  - old reference uses `tstart=100`, `tend=300`
  - current canonical default stays `tstart=100`, `tend=350`
  - this alone can explain a substantial ED mismatch even when the rest of the pipeline is aligned.

## Session Log (2026-03-19)
- Started replacing `aux_controls/sampling_strength.py` with a canonical S3-backed version.
- Canonical source decision for this analysis:
  - `local_ed` / `global_ed` must come from `f"{pth.SAVEDIR}/timextime/ed_main/{roi}.pkl"`
  - old `../datasets/NNN/*` ED tables should be ignored unless explicitly requested by the user
- User emphasized that repo coding style should follow local script patterns, not generic simplification:
  - `vprint()` is standard and should be treated as an exception to the “avoid helper functions” rule
  - avoid bulky `.txt` side outputs unless clearly needed
  - `aux_controls/sampling_strength.py` was rewritten in ad hoc style to match other exploratory analysis scripts:
    - top-of-file config
    - module-level `vprint()`
    - linear execution
