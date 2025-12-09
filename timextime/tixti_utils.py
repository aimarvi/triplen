import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import pearsonr, spearmanr, entropy, rankdata
from tqdm import tqdm

# some CONFIG parameters
RAND = 0
RESP = (50,220)
BASE = (-50,0)
ONSET = 50
RESP = slice(ONSET + RESP[0], ONSET + RESP[1])
BASE = slice(ONSET + BASE[0], ONSET + BASE[1])

def geo_rdm(dat, roi, mode='top', step=5, k_max=200, metric='correlation', random_state=RAND):
    '''
    calculates a series of Time x Time RDMs:
      (1) for a single time point
        (a) calculate a K x K image RDM across all units
        (b) take upper diagonal (kRDV) and store in a list
      (2) calculate uber RDM where each entry is =distance(kRDV, kRDV')
      (3) final RDM should be Time x Time
      (4) repeat for different manifold scales (K)
    
    args:
        dat (DataFrame): data
        roi (str): ROI name (eg. 'MF1_7_F')
        mode (str): ['top', 'shuff']
            top: orders images by response magnitude
            shuff: randomly selects images of the same scale
        step (int): manifold scale step size between two Time x Time RDMs
        k_max (int): maximum manifold scale in final Time x Time RDM
        metric (str): distance metric for Time x Time RDM
        random_state (int): random seed

    returns:
        sizes: list(manifold scales)
        rdvs: list(all Time x Time RDMs)
    '''
    rng = np.random.default_rng(random_state)

    sig = dat[dat['p_value'] < 0.05]
    df = sig[sig['roi'] == roi]
    if len(df) == 0:
        raise ValueError(f"No data for ROI {ROI}")
    X = np.stack(df['img_psth'].to_numpy())          # (units, time, images)

    # sort by response magnitude (baseline subtracted)
    scores = np.nanmean(X[:, RESP, :], axis=(0,1)) - np.nanmean(X[:, BASE, :], axis=(0,1))
    order = np.argsort(scores)[::-1] if mode == 'top' else rng.permutation(scores.size)

    # ================= choose the image-set bins to calculate RDMs ========
    sizes = [k for k in range(step, min(k_max, X.shape[2]) + 1, step)]
    # =================== ramping step size ================================ 
    # sizes = [k for k in range(1, 2*step)] + [k for k in range(2*step, min(k_max, X.shape[2])+1, step)]
    
    rdvs = []
    for k in tqdm(sizes):
        idx = order[:k]
        # 50 - 350 msec time window, relative to image onset
        # suggested by TK
        Ximg = X[:, 100:400, idx] # (units, time, images)
        Xrdv = np.array([pdist(Ximg[:, t, :].T, metric='correlation') for t in range(Ximg.shape[1])])

        # rank so distance metric is spearman instead of pearson
        Xrdv = np.apply_along_axis(rankdata, 1, Xrdv)
        R = squareform(pdist(Xrdv, metric=metric))   # (time, time)
        rdvs.append(R)
    return sizes, rdvs

def static_rdm(dat, roi, mode='top', scale=30, tstart=100, tend=400, metric='correlation', random_state=RAND):
    rng = np.random.default_rng(random_state)

    sig = dat[dat['p_value'] < 0.05]
    df = sig[sig['roi'] == roi]
    if len(df) == 0:
        raise ValueError(f"No data for ROI {roi}")
    X = np.stack(df['img_psth'].to_numpy())          # (units, time, images)

    # score images (using global RESP/BASE you already defined)
    scores = np.nanmean(X[:, RESP, :], axis=(0, 1)) - np.nanmean(X[:, BASE, :], axis=(0, 1))
    order = np.argsort(scores)[::-1] if mode == 'top' else rng.permutation(scores.size)

    # pick image subset
    idx = order[:scale]

    # restrict to desired time window
    Ximg = X[:, tstart:tend, idx]                    # (units, time, images)

    # time-by-RDV (one RDV per timepoint)
    Xrdv = np.array([
        pdist(Ximg[:, t, :].T, metric='correlation')
        for t in range(Ximg.shape[1])
    ])  # (time, n_pairs)

    # # Spearman: rank-transform rows, then use correlation distance across time
    Xrank = np.apply_along_axis(rankdata, 1, Xrdv)
    R = squareform(pdist(Xrank, metric=metric))      # (time, time)

    return R, Xrdv

def rdv(X):
    ind = np.triu_indices_from(X, k=1)
    return(X[ind])

def l2(X):
    return np.sqrt(np.sum((X)**2))

def ED1(R):
    S = -0.5 * R**2
    lam = np.linalg.eigvalsh(S)
    lam = np.clip(lam, 0, None)
    return (lam.sum()**2) / (lam**2).sum()

def ED2(R):
    # R = distance matrix
    n = R.shape[0]
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ (R**2) @ J
    lam = np.linalg.eigvalsh(B)
    lam = np.clip(lam, 0, None)
    return (lam.sum()**2) / (lam**2).sum()


def entropy(V):
    v = np.abs(V)
    return v / v.sum()
