import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import utils

# =========================
# CONFIG
# =========================
ONSET_MS = 0          # stimulus onset (ms)
WINDOW_MS = 350        # analyze 0-300 ms post onset
BIN_MS = 1             # temporal bin size (ms). If not 1, change indices below
K_RANGE = range(2, 11) # candidates for number of clusters
RANDOM_STATE = 0
N_INIT = "auto"        # or an int (e.g., 20) if on older sklearn

# ===================================
# LOAD IN DATA 
# ===================================
dat = pd.read_pickle('../../datasets/NNN/unit_data_full.pkl')
# start and end times calculated from CONFIG
start = int(ONSET_MS / BIN_MS)
end   = start + int(WINDOW_MS / BIN_MS)

timecourses = []
valid_idx = []  # track which rows succeed
for i, row in dat.iterrows():
    try:
        tc = utils.get_unit_timecourse(row, start, end)
        timecourses.append(tc)
        valid_idx.append(i)
    except Exception as e:
        # skip units that don't have valid PSTH
        print(f"Skipping unit {i}: {e}")
        pass
X = np.vstack(timecourses)  # (n_valid_units, T)

# Z-score each unit across time (mean=0, std=1)
X = zscore(X, axis=1)
# Replace any NaNs from zero-variance rows with zeros
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# =========================
# INERTIA/SILHOUETTE: choose K
# =========================
sil_scores = []
labels_by_k = {}
inertias = {}
for k in tqdm(K_RANGE):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    lab = km.fit_predict(X)
    # inertia to measure fit of k clusters
    km.fit(X)
    inertias[k] = km.inertia_
    labels_by_k[k] = lab
    # alternatively, use silhouette
    # requires >1 cluster and less than n_samples
    sil = silhouette_score(X, lab)
    sil_scores.append((k, sil))

################### by INERTIA ###################
df = pd.DataFrame({
    "k": list(inertias.keys()),
    "inertia": list(inertias.values())
})
# seaborn lineplot
plt.figure(figsize=(7, 4))
sns.lineplot(data=df, x="k", y="inertia", marker="o", linewidth=2.5)
plt.show()

############### by SILHOUETTE score ########################
# Pick best K
best_k, best_sil = max(sil_scores, key=lambda t: t[1])
print("Silhouette scores:")
for k, s in sil_scores:
    print(f"  k={k}: silhouette={s:.4f}")
print(f"=> Selected k={best_k} (silhouette={best_sil:.4f})")

# Attach labels back to the original `dat`
dat['cluster'] = np.nan
labels = labels_by_k[best_k]
dat.loc[valid_idx, 'cluster'] = labels
