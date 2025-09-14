"""
jointNMF_clean.py
------------------
Cleaned (style/organization) version of the original jointNMF_6.py.
Functionality, defaults, math, and outputs are intentionally unchanged.
"""
import os
import warnings
import numpy as np

def clNMF(
    a,
    X1,
    X2,
    k,
    delta,
    num_iter,
    model_path,
    early_stopping_epoch=50,
    init_W=None,
    init_H1=None,
    init_H2=None,
    print_enabled=True,
):
    """
    Perform joint NMF with a shared dictionary W such that X1 ≈ W H1 and X2 ≈ W H2.
    """
    if print_enabled:
        print('---------------------------------------------------------------------')
        print('Objective: a*||X1 - WH1||^2 + ||X2 - WH2||^2 + constraint for W growth control + '
              'sparsity constraint for H1 + sparsity constraint for H2')
        print('')

    # ---- Initialization ------------------------------------------------
    if init_W is None:
        W = np.random.rand(np.size(X1, 0), k) / 2.0
        D = np.diag(np.linalg.norm(W, axis=0, ord=2))
        W = W @ np.linalg.inv(D)
    else:
        W = init_W

    if init_H1 is None:
        H1 = np.random.rand(k, np.size(X1, 1)) / 2.0
        H1 = D @ H1
    else:
        H1 = init_H1

    if init_H2 is None:
        H2 = np.random.rand(k, np.size(X2, 1)) / 2.0
        H2 = D @ H2
    else:
        H2 = init_H2

    best_loss = np.inf
    best_epoch = 0
    loss_list = []
    sparsity_list = []

    # ---- Multiplicative updates loop -----------------------------------
    for epoch in range(num_iter):
        # Update W
        X1H1_T = X1 @ H1.T
        WH1H1_T = W @ H1 @ H1.T
        X2H2_T = X2 @ H2.T
        WH2H2_T = W @ H2 @ H2.T

        for i in range(np.size(W, 0)):
            for j in range(np.size(W, 1)):
                numerator = a * X1H1_T[i, j] + X2H2_T[i, j]
                denominator = a * WH1H1_T[i, j] + WH2H2_T[i, j] + delta
                W[i, j] = W[i, j] * (numerator / denominator)

        # Update H2
        W_TX2 = W.T @ X2
        W_TWH2 = W.T @ W @ H2 + delta
        for i in range(np.size(H2, 0)):
            for j in range(np.size(H2, 1)):
                H2[i, j] = H2[i, j] * (W_TX2[i, j] / W_TWH2[i, j])

        # Update H1
        W_TX1 = W.T @ X1
        W_TWH1 = W.T @ W @ H1 + delta
        for i in range(np.size(H1, 0)):
            for j in range(np.size(H1, 1)):
                H1[i, j] = H1[i, j] * (W_TX1[i, j] / W_TWH1[i, j])

        if (W < 0).sum().sum() or (H1 < 0).sum().sum() or (H2 < 0).sum().sum():
            print('WARNING: Negative matrix!!')

        # Column re-scaling
        S = np.diag(np.linalg.norm(W, axis=0, ord=2))
        try:
            W = W @ np.linalg.inv(S)
            H1 = S @ H1
            H2 = S @ H2

            loss = a * np.power(np.linalg.norm(X1 - W @ H1, 'fro'), 2) \
                   + np.power(np.linalg.norm(X2 - W @ H2, 'fro'), 2)
            loss_list.append(loss)

            W_size = W.size
            sparsity = (np.sqrt(W_size) - (np.sum(np.absolute(W)) / np.sqrt(np.sum(np.square(W))))) \
                       / (np.sqrt(W_size) - 1)
            sparsity_list.append(sparsity)
        except Exception:
            pass

        # Early stopping
        if epoch == 0 or (loss < best_loss):
            best_loss = loss if epoch == 0 else loss
            best_epoch = epoch

            H2_len = np.sqrt(np.sum(np.square(H2), 1))
            H1_len = np.sqrt(np.sum(np.square(H1), 1))
            H2_len[H2_len == 0] = 1
            H1_len[H1_len == 0] = 1

            W_H1_len_H2_len = (np.multiply(W, H1_len.transpose())
                               + np.multiply(W, H2_len.transpose()))
            index_descend = (-np.sum(np.square(W_H1_len_H2_len), 0)).argsort()
            W = W[:, index_descend]
            H1 = H1[index_descend, :]
            H2 = H2[index_descend, :]

            np.save(os.path.join(model_path, 'W.npy'), W)
            np.save(os.path.join(model_path, 'H1.npy'), H1)
            np.save(os.path.join(model_path, 'H2.npy'), H2)
            np.save(os.path.join(model_path, 'loss.npy'), loss)

        if (epoch - best_epoch) >= early_stopping_epoch:
            break

    np.save(os.path.join(model_path, 'loss_list.npy'), loss_list)
    np.save(os.path.join(model_path, 'sparsity_list.npy'), sparsity_list)

    return W, H1, H2, loss_list


class EarlyStopping(object):
    def __init__(self, mode='loss', min_delta=0, patience_epoch=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience_epoch
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience_epoch == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'loss'}:
            raise ValueError('mode ' + mode + ' is unknown!')

        if mode == 'loss':
            self.is_better = lambda a, best: a < best - best * min_delta

