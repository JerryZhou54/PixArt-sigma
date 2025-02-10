import itertools
import torch
from multiprocessing import Pool

def return_idx(X, T, n):
    I = []
    I.append((X <= T[0]).nonzero().reshape(-1))
    if len(T) >= 2:
        for i in range(len(T) - 1):
            I.append(((X > T[i]) & (X <= T[i + 1])).nonzero().reshape(-1))
    I.append((X > T[-1]).nonzero().reshape(-1))
    return I

def intraclassvar(X, T, n, N):
    I = return_idx(X, T, n)
    V = []
    W = []
    for i in range(n + 1):
        idx = I[i]
        if len(idx) != 0:
            var_ = torch.var(X[idx].float())
            W.append(len(idx) / N)
            V.append(var_.item())
    intraclassvar = sum([v * w for v, w in zip(V, W)])
    return intraclassvar

def nThresh(X, n_classes=2, bins=10, n_jobs=None):
    '''
    X : torch.Tensor
        1-dimensional PyTorch tensor on CUDA
    n_classes : int
        Number of expected classes. n_classes - 1 threshold values will be returned in a list
    bins : int
        Number of bins to use when binning the space of X
    n_jobs : int
        Number of cores to use. If None, all possible cores will be used
    '''
    n = n_classes - 1
    min_val = X.min().item()
    max_val = X.max().item()
    
    if max_val == min_val:
        # If all values in X are the same, return a single threshold at that value
        return [min_val]

    V = torch.arange(min_val, max_val, (max_val - min_val) / bins)[:-1]
    Ts = list(itertools.combinations(V.tolist(), n))
    N = len(X)

    # Convert X to CPU for multiprocessing
    X_cpu = X.cpu()
    
    with Pool(n_jobs) as p:
        q = p.starmap(intraclassvar, [(X_cpu, T, n, N) for T in Ts])
    
    return Ts[torch.argmin(torch.tensor(q)).item()]