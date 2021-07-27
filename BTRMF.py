import numpy as np
import pandas as pd
import tqdm
from metrics import rmse,mape
from numpy.linalg import inv as inv
from numpy.random import normal as normrnd
from scipy.linalg import khatri_rao as kr_prod
from scipy.stats import wishart
from numpy.linalg import solve as solve
from scipy.linalg import cholesky as cholesky_upper
from scipy.linalg import solve_triangular as solve_ut


def mvnrnd_pre(mu, Lambda):
    src = normrnd(size=(mu.shape[0],))
    return solve_ut(cholesky_upper(Lambda, overwrite_a=True, check_finite=False),
                    src, lower=False, check_finite=False, overwrite_b=True) + mu


def cov_mat(mat, mat_bar):
    mat = mat - mat_bar
    return mat.T @ mat


def sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0=1, vargin=0):
    """Sampling N-by-R factor matrix W and its hyperparameters (mu_w, Lambda_w)."""

    dim1, rank = W.shape
    W_bar = np.mean(W, axis=0)
    temp = dim1 / (dim1 + beta0)
    var_W_hyper = inv(np.eye(rank) + cov_mat(W, W_bar) + temp * beta0 * np.outer(W_bar, W_bar))
    var_Lambda_hyper = wishart.rvs(df=dim1 + rank, scale=var_W_hyper)
    var_mu_hyper = mvnrnd_pre(temp * W_bar, (dim1 + beta0) * var_Lambda_hyper)

    if dim1 * rank ** 2 > 1e+8:
        vargin = 1

    if vargin == 0:
        var1 = X.T
        var2 = kr_prod(var1, var1)
        var3 = (var2 @ tau_ind.T).reshape([rank, rank, dim1]) + var_Lambda_hyper[:, :, None]
        var4 = var1 @ tau_sparse_mat.T + (var_Lambda_hyper @ var_mu_hyper)[:, None]
        for i in range(dim1):
            W[i, :] = mvnrnd_pre(solve(var3[:, :, i], var4[:, i]), var3[:, :, i])
    elif vargin == 1:
        for i in range(dim1):
            pos0 = np.where(tau_sparse_mat[i, :] != 0)
            Xt = X[pos0[0], :]
            var_mu = tau[i] * Xt.T @ tau_sparse_mat[i, pos0[0]] + var_Lambda_hyper @ var_mu_hyper
            var_Lambda = tau[i] * Xt.T @ Xt + var_Lambda_hyper
            W[i, :] = mvnrnd_pre(solve(var_Lambda, var_mu), var_Lambda)

    return W


def sample_theta(X, theta, Lambda_x, time_lags, beta0=1):
    dim, rank = X.shape
    d = time_lags.shape[0]
    tmax = np.max(time_lags)
    theta_bar = np.mean(theta, axis=0)
    temp = d / (d + beta0)
    var_theta_hyper = inv(np.eye(rank) + cov_mat(theta, theta_bar)
                          + temp * beta0 * np.outer(theta_bar, theta_bar))
    var_Lambda_hyper = wishart.rvs(df=d + rank, scale=var_theta_hyper)
    var_mu_hyper = mvnrnd_pre(temp * theta_bar, (d + beta0) * var_Lambda_hyper)

    for k in range(d):
        theta0 = theta.copy()
        theta0[k, :] = 0
        mat0 = np.zeros((dim - tmax, rank))
        for L in range(d):
            mat0 += X[tmax - time_lags[L]: dim - time_lags[L], :] @ np.diag(theta0[L, :])
        varPi = X[tmax: dim, :] - mat0
        var0 = X[tmax - time_lags[k]: dim - time_lags[k], :]
        var = np.einsum('ij, jk, ik -> j', var0, Lambda_x, varPi)
        var_Lambda = np.einsum('ti, tj, ij -> ij', var0, var0, Lambda_x) + var_Lambda_hyper
        theta[k, :] = mvnrnd_pre(solve(var_Lambda, var + var_Lambda_hyper @ var_mu_hyper), var_Lambda)

    return theta


def sample_Lambda_x(X, theta, time_lags):
    dim, rank = X.shape
    d = time_lags.shape[0]
    tmax = np.max(time_lags)
    mat = X[: tmax, :].T @ X[: tmax, :]
    temp = np.zeros((dim - tmax, rank, d))
    for k in range(d):
        temp[:, :, k] = X[tmax - time_lags[k]: dim - time_lags[k], :]
    new_mat = X[tmax: dim, :] - np.einsum('kr, irk -> ir', theta, temp)
    Lambda_x = wishart.rvs(df=dim + rank, scale=inv(np.eye(rank) + mat + new_mat.T @ new_mat))

    return Lambda_x


def sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, theta, Lambda_x):
    """Sampling T-by-R factor matrix X."""

    dim2, rank = X.shape
    tmax = np.max(time_lags)
    tmin = np.min(time_lags)
    d = time_lags.shape[0]
    A = np.zeros((d * rank, rank))
    for k in range(d):
        A[k * rank: (k + 1) * rank, :] = np.diag(theta[k, :])
    A0 = np.dstack([A] * d)
    for k in range(d):
        A0[k * rank: (k + 1) * rank, :, k] = 0
    mat0 = Lambda_x @ A.T
    mat1 = np.einsum('kij, jt -> kit', A.reshape([d, rank, rank]), Lambda_x)
    mat2 = np.einsum('kit, kjt -> ij', mat1, A.reshape([d, rank, rank]))

    var1 = W.T
    var2 = kr_prod(var1, var1)
    var3 = (var2 @ tau_ind).reshape([rank, rank, dim2]) + Lambda_x[:, :, None]
    var4 = var1 @ tau_sparse_mat
    for t in range(dim2):
        Mt = np.zeros((rank, rank))
        Nt = np.zeros(rank)
        Qt = mat0 @ X[t - time_lags, :].reshape(rank * d)
        index = list(range(0, d))
        if t >= dim2 - tmax and t < dim2 - tmin:
            index = list(np.where(t + time_lags < dim2))[0]
        elif t < tmax:
            Qt = np.zeros(rank)
            index = list(np.where(t + time_lags >= tmax))[0]
        if t < dim2 - tmin:
            Mt = mat2.copy()
            temp = np.zeros((rank * d, len(index)))
            n = 0
            for k in index:
                temp[:, n] = X[t + time_lags[k] - time_lags, :].reshape(rank * d)
                n += 1
            temp0 = X[t + time_lags[index], :].T - np.einsum('ijk, ik -> jk', A0[:, :, index], temp)
            Nt = np.einsum('kij, jk -> i', mat1[index, :, :], temp0)

        var3[:, :, t] = var3[:, :, t] + Mt
        if t < tmax:
            var3[:, :, t] = var3[:, :, t] - Lambda_x + np.eye(rank)
        X[t, :] = mvnrnd_pre(solve(var3[:, :, t], var4[:, t] + Nt + Qt), var3[:, :, t])

    return X


def sample_precision_tau(sparse_mat, mat_hat, ind):
    var_alpha = 1e-6 + 0.5 * np.sum(ind, axis=1)
    var_beta = 1e-6 + 0.5 * np.sum(((sparse_mat - mat_hat) ** 2) * ind, axis=1)
    return np.random.gamma(var_alpha, 1 / var_beta)
def BTRMF(sparse_mat_ori, rank=50, time_lags=(1, 2), burn_iter=1000, gibbs_iter=200):
    """Bayesian Temporal Regularized Matrix Factorization, BTRMF."""


    sparse_mat = sparse_mat_ori.copy()
    dim1, dim2 = sparse_mat.shape
    time_lags = np.array(time_lags)
    d = time_lags.shape[0]
    W = 0.1 * np.random.randn(dim1, rank)
    X = 0.1 * np.random.randn(dim2, rank)
    theta = 0.01 * np.random.randn(d, rank)
    if np.isnan(sparse_mat).any() == True:
        sparse_mat[np.isnan(sparse_mat)] = 0
    ind = sparse_mat != 0
    tau = np.ones(dim1)
    mat_hat_plus = np.zeros((dim1, dim2))
    for it in tqdm.trange(burn_iter + gibbs_iter):
        tau_ind = tau[:, None] * ind
        tau_sparse_mat = tau[:, None] * sparse_mat
        W = sample_factor_w(tau_sparse_mat, tau_ind, W, X, tau, beta0=1, vargin=0)
        Lambda_x = sample_Lambda_x(X, theta, time_lags)
        theta = sample_theta(X, theta, Lambda_x, time_lags)
        X = sample_factor_x(tau_sparse_mat, tau_ind, time_lags, W, X, theta, Lambda_x)
        mat_hat = W @ X.T
        tau = sample_precision_tau(sparse_mat, mat_hat, ind)
        if it + 1 > burn_iter:
            mat_hat_plus += mat_hat
    mat_hat = mat_hat_plus / gibbs_iter
    mat_hat[mat_hat < 0] = 0

    return mat_hat


def test_BTRMF():
    dense_mat = pd.read_csv('./datasets/Seattle-data-set/mat.csv', index_col=0)
    rm = pd.read_csv('./datasets/Seattle-data-set/RM_mat.csv', index_col=0)
    dense_mat = dense_mat.values
    rm = rm.values

    binary_mat2 = np.round(rm + 0.5 - 0.2)
    nan_mat2 = binary_mat2.copy()

    nan_mat2[nan_mat2 == 0] = np.nan

    sparse_mat2 = np.multiply(nan_mat2, dense_mat)

    pos2 = np.where((dense_mat != 0) & (binary_mat2 == 0))

    # sparse_tensor2 = sparse_mat2.reshape([sparse_mat2.shape[0], 28, 288])

    BTRMF_res2 = BTRMF(sparse_mat2, rank=50, time_lags=(1, 2, 288), burn_iter=100, gibbs_iter=20)

    BTRMF_res2_mape2 = mape(dense_mat[pos2], BTRMF_res2[pos2])
    BTRMF_res2_rmse2 = rmse(dense_mat[pos2], BTRMF_res2[pos2])

    print("BTRMF_res2_mape2", BTRMF_res2_mape2)
    print("BTRMF_res2_rmse2", BTRMF_res2_rmse2)

if __name__=="__main__":
    test_BTRMF()

    # BTRMF_res2_mape2
    # 5.850776139250494
    # BTRMF_res2_rmse2
    # 3.714650635105775