import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import pearsonr, spearmanr, entropy, rankdata
from scipy.ndimage import gaussian_filter1d

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
    '''
    calculates a single Time x Time RDM, given:
      (1) a scale (top k images)
      (2) a time window (0-400 msec; 50 = image onset)

    args:
        dat (DataFrame): data
        roi (str): ROI name (eg. 'MF1_7_F')
        mode (str): ['top', 'shuff']
            top: orders images by response magnitude
            shuff: randomly selects images of the same scale
        scale (int): manifold scale used to calculate Time x Time RDMs
        tstart (int): start of time window
        tend (int): end of time window
        metric (str): distance metric for Time x Time RDM
        random_state (int): random seed

    returns:
        R: array(Time x Time RDM) (square)
        Xrdv: vectorized form of R (pre rank transformation) 
    '''
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

def time_avg_rdm(dat, roi, window=RESP, images='all', metric='correlation', random_state=RAND):
    """
    Calculates a traditional, image RDM within a given response window

    args:
        dat (DataFrame): data
        roi (str): ROI name (eg. 'MF1_7_F')
        window (tuple): window for averaging neural responses
        images (str): which image set to choose ('all', 'nsd', 'localizer')
        metric (str): distance metric for RDM
        random_state (int): random seed

    returns:
        R: array(Image RDM) (square)
        Xrdv: vectorized form of R  

    """

    rng = np.random.default_rng(random_state)

    sig = dat[dat['p_value'] < 0.05]
    df = sig[sig['roi'] == roi]
    if len(df) == 0:
        raise ValueError(f"No data for ROI {ROI}")
    X = np.stack(df['img_psth'].to_numpy())          # (units, time, images)

    istart = 0; iend = X.shape[2];
    match images:
        case 'all':
            pass
        case 'nsd':
            iend = 1000
        case 'localizer':
            istart = 1000

    # average unit responses over time window
    Xw = np.nanmean(X[:, window, istart:iend], axis=1)
    Xrdv = pdist(Xw.T, metric=metric)
    R = squareform(Xrdv)

    return R, Xrdv

def landscape(dat, roi, rsp=RESP, random_state=RAND):
    '''
    normalized (baseline subtracted) response to each image within a given time window
    uses global RESP variable by default

    args:
        dat (DataFrame): data
        roi (str): ROI name (eg. 'MF1_7_F')
        rsp (tuple): response window (default: RESP)
        random_state (int): random seed

    returns:
        scores: scores for each image
    '''
    rng = np.random.default_rng(random_state)

    sig = dat[dat['p_value'] < 0.05]
    df = sig[sig['roi'] == roi]
    if len(df) == 0:
        raise ValueError(f"No data for ROI {roi}")
    X = np.stack(df['img_psth'].to_numpy())          # (units, time, images)

    # score images (using global RESP/BASE you already defined)
    scores = np.nanmean(X[:, rsp, :], axis=(0, 1)) - np.nanmean(X[:, BASE, :], axis=(0, 1))

    return scores

def rdv(X):
    ind = np.triu_indices_from(X, k=1)
    return(X[ind])

def l2(X):
    return np.sqrt(np.sum((X)**2))

def ED1(R):
    """
    Standard calculation of effective dimensionality
    """
    S = -0.5 * R**2
    lam = np.linalg.eigvalsh(S)
    lam = np.clip(lam, 0, None)
    return (lam.sum()**2) / (lam**2).sum()

def ED2(R):
    """
    Alternate form of effective dimensionality 
    Use when R is a matrix of distance values (eg. cosine distance, pearson r, spearman rho, etc)
    """
    n = R.shape[0]
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ (R**2) @ J
    lam = np.linalg.eigvalsh(B)
    lam = np.clip(lam, 0, None)
    return (lam.sum()**2) / (lam**2).sum()

def entropy(V):
    v = np.abs(V)
    return v / v.sum()
