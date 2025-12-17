import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
gates = ['EdU+BrdU-', 'EdU-BrdU+', 'EdU+BrdU+', 'EdU-BrdU-']


def build_base_matrices():
    n = 3
    S_base = np.array([[-1,  0, +2],
                       [+1, -1,  0],
                       [ 0, +1, -1]], dtype=float)
    sources = np.array([0,1,2], dtype=int)
    return S_base, sources

def build_A_from_rates(k):
    S_base, sources = build_base_matrices()
    A = np.zeros((3,3))
    Rn = S_base.shape[1]
    for r in range(Rn):
        col = S_base[:, r]
        src = sources[r]
        A[:, src] += col * k.iloc[r]
    return A

def expand_to_labels(S_base, sources_base, n_labels):
    n = S_base.shape[0]
    Rn = S_base.shape[1]
    S_big = np.zeros((n * n_labels, Rn * n_labels), dtype=float)
    sources_big = np.zeros(Rn * n_labels, dtype=int)
    for L in range(n_labels):
        row_offset = L * n
        col_offset = L * Rn
        S_big[row_offset:row_offset+n, col_offset:col_offset+Rn] = S_base.copy()
        for r in range(Rn):
            sources_big[col_offset + r] = row_offset + sources_base[r]
    return S_big, sources_big

def build_A_big(k, n_labels):
    A_base = build_A_from_rates(k)
    A_big = np.kron(np.eye(n_labels), A_base)
    return A_big

def build_label1_mapping():
    P1 = np.zeros((6,3), dtype=float)
    P1[0,0] = 1.0   # G1 -> G1_l0
    P1[4,1] = 1.0   # S  -> S_l1   (moved to labelled block)
    P1[2,2] = 1.0   # G2M -> G2M_l0
    return P1

def build_label2_mapping_after_label1():
    P2 = np.zeros((12,6), dtype=float)
    # pre indices: 0:G1_l0,1:S_l0,2:G2M_l0,3:G1_l1,4:S_l1,5:G2M_l1
    # post blocks: [00(0..2), 01(3..5), 10(6..8), 11(9..11)]
    P2[0,0] = 1.0   # G1_l0 -> G1_00
    P2[4,1] = 1.0   # S_l0  -> S_01 (label2 applied -> moves to label2=1 within label1=0)
    P2[2,2] = 1.0   # G2M_l0 -> G2M_00
    P2[6,3] = 1.0   # G1_l1 -> G1_10
    P2[10,4] = 1.0  # S_l1  -> S_11 (label2 applied -> moves to label2=1 within label1=1)
    P2[8,5] = 1.0   # G2M_l1 -> G2M_10
    return P2

def build_observation_mapping():
    M = np.zeros((4,12), dtype=float)
    # EdU-only = label1=1,label2=0 -> indices 6,7,8
    M[0,6:9] = 1.0
    # BrdU-only = label1=0,label2=1 -> indices 3,4,5
    M[1,3:6] = 1.0
    # Double = label1=1,label2=1 -> indices 9,10,11
    M[2,9:12] = 1.0
    # Negative = label1=0,label2=0 -> indices 0,1,2
    M[3,0:3] = 1.0
    return M

def integrate_piecewise_with_labels(k, mu0, Sigma0, t0, t_label1, t_label2, t_end):
    S_base, sources_base = build_base_matrices()
    Rn = S_base.shape[1]
    P1 = build_label1_mapping()
    P2 = build_label2_mapping_after_label1()
    M_obs = build_observation_mapping()

    def integrate_segment(mu_init, Sigma_init, A_big, S_big, sources_big, rates_big, t_a, t_b):
        n = mu_init.size
        def ode(t, y):
            mu = y[:n]
            Sigma_flat = y[n:]
            Sigma = Sigma_flat.reshape((n,n))
            dmu = A_big @ mu
            a = np.zeros(sources_big.shape[0])
            for r in range(len(a)):
                src = sources_big[r]
                a[r] = rates_big[r] * max(mu[src], 0.0)
            D = S_big @ np.diag(a) @ S_big.T
            dSigma = A_big @ Sigma + Sigma @ A_big.T + D
            return np.concatenate([dmu, dSigma.reshape(n*n)])
        y0 = np.concatenate([mu_init, Sigma_init.reshape(mu_init.size * mu_init.size)])
        sol = solve_ivp(ode, (t_a, t_b), y0, method='BDF', atol=1e-8, rtol=1e-6)
        yf = sol.y[:, -1]
        mu_f = yf[:n]
        Sigma_f = yf[n:].reshape((n,n))
        Sigma_f = 0.5 * (Sigma_f + Sigma_f.T)
        return mu_f, Sigma_f

    # segment 0
    A0 = build_A_from_rates(k)
    S0 = S_base.copy()
    sources0 = sources_base.copy()
    rates0 = np.array(k, dtype=float)
    mu_pre = mu0.copy()
    Sigma_pre = Sigma0.copy()
    if t_label1 > t0:
        mu_post1, Sigma_post1 = integrate_segment(mu_pre, Sigma_pre, A0, S0, sources0, rates0, t0, t_label1)
    else:
        mu_post1, Sigma_post1 = mu_pre.copy(), Sigma_pre.copy()

    # apply P1
    mu_after_P1 = P1 @ mu_post1
    Sigma_after_P1 = P1 @ Sigma_post1 @ P1.T

    # segment 1 with 2 label classes
    n_labels1 = 2
    S_big1, sources_big1 = expand_to_labels(S_base, sources_base, n_labels1)
    rates_big1 = np.tile(rates0, n_labels1)
    A_big1 = build_A_big(k, n_labels1)
    if t_label2 > t_label1:
        mu_post2, Sigma_post2 = integrate_segment(mu_after_P1, Sigma_after_P1, A_big1, S_big1, sources_big1, rates_big1, t_label1, t_label2)
    else:
        mu_post2, Sigma_post2 = mu_after_P1.copy(), Sigma_after_P1.copy()

    # apply P2
    mu_after_P2 = P2 @ mu_post2
    Sigma_after_P2 = P2 @ Sigma_post2 @ P2.T

    # final segment with 4 label classes
    n_labels2 = 4
    S_big2, sources_big2 = expand_to_labels(S_base, sources_base, n_labels2)
    rates_big2 = np.tile(rates0, n_labels2)
    A_big2 = build_A_big(k, n_labels2)
    if t_end > t_label2:
        mu_final, Sigma_final = integrate_segment(mu_after_P2, Sigma_after_P2, A_big2, S_big2, sources_big2, rates_big2, t_label2, t_end)
    else:
        mu_final, Sigma_final = mu_after_P2.copy(), Sigma_after_P2.copy()

    counts_mean = M_obs @ mu_final
    counts_cov = M_obs @ Sigma_final @ M_obs.T
    return counts_mean, counts_cov, {'mu_final': mu_final, 'Sigma_final': Sigma_final, 'M_obs': M_obs}

def compute_snr_from_modelfunc(model_func, theta, R=5, rel_step=1e-6, eps=1e-9, reg=1e-12, param_names=None):
    theta = np.asarray(theta, dtype=float)
    p = theta.size
    if param_names is None:
        param_names = [f"theta_{i}" for i in range(p)]
    m0, Sigma0 = model_func(theta)
    m0 = np.asarray(m0, dtype=float).ravel()
    Sigma0 = np.asarray(Sigma0, dtype=float)
    k_obs = m0.size
    S = np.zeros((k_obs, p), dtype=float)
    for j in range(p):
        delta = rel_step * max(1.0, abs(theta[j])) + eps
        thp = theta.copy(); thm = theta.copy()
        thp[j] += delta; thm[j] -= delta
        mp, _ = model_func(thp); mm, _ = model_func(thm)
        S[:, j] = (np.asarray(mp).ravel() - np.asarray(mm).ravel()) / (2.0 * delta)
    trace = np.trace(Sigma0)
    scaled_reg = reg * (trace / k_obs) if trace != 0 else reg
    Sigma_reg = Sigma0 + np.eye(k_obs) * scaled_reg
    try:
        Sigma_inv = np.linalg.inv(Sigma_reg)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma_reg)
    Fisher = R * (S.T @ Sigma_inv @ S)
    try:
        cov_theta = np.linalg.inv(Fisher)
    except np.linalg.LinAlgError:
        cov_theta = np.linalg.pinv(Fisher)
    se = np.sqrt(np.maximum(np.diag(cov_theta), 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        SNR = np.abs(theta) / se
        SNR[se == 0] = np.nan
    results = []
    for j in range(p):
        results.append({'param': param_names[j], 'theta': float(theta[j]), 'std_error': float(se[j]), 'SNR': float(SNR[j])})
    df = pd.DataFrame(results).set_index('param')
    diagnostics = {'S': S, 'm0': m0, 'Sigma0': Sigma0, 'Fisher': Fisher, 'cov_theta': cov_theta}
    return df, diagnostics

def get_noisy_initial(k, mu0, Sigma0, n_end):
    n=3
    k1, k2, k3 = k
    initial = mu0 + [Sigma0[0,0], Sigma0[0,1], Sigma0[1,1], Sigma0[1,2], Sigma0[2,2], Sigma0[2,0]]
    # G1, S, G2M, G1^2, G1*S, S^2, S*G2M, G2M^2, G2M*G1
    K = np.array([
        [-k1, 0,  2*k3,    0,   0,    0,   0,    0,    0], #G1
        [k1, -k2,   0,     0,   0,    0,   0,    0,    0], #S
        [0,   k2,  -k3,    0,   0,    0,   0,    0,    0], #G2M
        [k1,   0,  4*k3, -2*k1, 0,    0,   0,    0,   4*k3], #G1^2
        [-k1,  0,    0,   k1, -k1-k2, 0,  2*k3,  0,    0], #G1*S
        [k1,   k2,   0,    0,  2*k1,-2*k2, 0,    0,    0], #S^2
        [0,   -k2,   0,    0,   0,    k2,-k2-k3, 0,    k1], #S*G2M
        [0,   k2,   k3,    0,   0,    0, 2*k2, -2*k3,  0], #G2M^2
        [0,    0,  -2*k3,  0,   k2,   0,   0,   2*k3, -k1-k3],#G2M*G1
        #G1    S    G2M   G1^2  G1*S S^2 S*G2M  G2M^2 G2M*G1
    ], dtype=float)
    ode = lambda t, y: K @ y
    def reached_n_end(t, y): return sum(y[:n]) - n_end
    reached_n_end.terminal = True
    sol = solve_ivp(ode, (0, 1000), initial, method='BDF', atol=1e-8, rtol=1e-6, events=reached_n_end)
    yf = sol.y[:, -1]
    mu_f = yf[:n]
    Sigma_f = np.array(
        [[yf[n], yf[n+1], yf[-1]],
        [yf[n+1], yf[n+2], yf[n+3]],
        [yf[-1], yf[n+3], yf[-2]]
    ], dtype=float)
    return mu_f, Sigma_f



def demo():
    # --------------------------
    # Demo run
    # --------------------------
    k = np.array([0.087656, 0.103810, 0.337980])
    mu0 = np.array([165.0, 106.0, 30.0])
    Sigma0 = np.diag(mu0) * 0.0
    t0 = 0.0; t_label1 = 0.0; t_label2 = 4.0; t_end = 4.5

    counts_mean, counts_cov, meta = integrate_piecewise_with_labels(k, mu0, Sigma0, t0, t_label1, t_label2, t_end)
    print("Observed counts mean (EdU-only, BrdU-only, Double, Negative):", counts_mean)
    print("Observed counts covariance matrix:\n", counts_cov)

    def model_func_for_snr(theta_vec):
        m, cov, _ = integrate_piecewise_with_labels(theta_vec, mu0, Sigma0, t0, t_label1, t_label2, t_end)
        return m, cov

    df_snr, diag = compute_snr_from_modelfunc(model_func_for_snr, k, R=5)
    print("\nSNR results:\n", df_snr)

    counts_mean, counts_cov, df_snr, diag

def sweep():
    # --------------------------
    # Sweep run
    # --------------------------
    df_gill = pd.read_json('DL bootstrap 11:07:2024-big concatted.json')
    df_new = df_gill[['k1', 'k2', 'k3', 'BrdU time', 'Initial G1', 'Initial S', 'Initial G2M']].copy()
    df_new['Covariance matrix'] = None
    for i, df_row in df_new.iterrows():
        k = df_row[['k1', 'k2', 'k3']]
        mu0 = df_row[['Initial G1', 'Initial S', 'Initial G2M']]
        t0 = t_label1 = 0.0; t_label2 = df_row['BrdU time']; t_end = t_label2 + 0.5

        counts_mean, counts_cov, meta = integrate_piecewise_with_labels(k, mu0, np.zeros((3,3)), t0, t_label1, t_label2, t_end)
        for j, gate in enumerate(gates):
            df_new.at[i, f'Mean {gate}'] = counts_mean[j]
            df_new.at[i, f'Std {gate}'] = np.sqrt(counts_cov[j, j])
        df_new.at[i, 'Covariance matrix'] = counts_cov

    df_new.to_json('data/DL analytic cov.json')

def noisy_sweep():
    # --------------------------
    # Sweep run, but initialise at [1,1,1] to find covariances of initial condition too
    # --------------------------
    df_gill = pd.read_json('DL bootstrap 11:07:2024-big concatted.json')
    n = sum(df_gill[['Initial G1', 'Initial S', 'Initial G2M']].iloc[0])
    df_new = df_gill[['k1', 'k2', 'k3', 'BrdU time', 'Initial G1', 'Initial S', 'Initial G2M']].copy()
    df_new['Covariance matrix'] = None
    df_new['Initial covariance matrix'] = None
    for i, df_row in df_new.iterrows():
        k = df_row[['k1', 'k2', 'k3']]
        mu0 = df_row[['Initial G1', 'Initial S', 'Initial G2M']].to_numpy(dtype=int)
        t0 = t_label1 = 0.0; t_label2 = df_row['BrdU time']; t_end = t_label2 + 0.5
        _, Sigma0 = get_noisy_initial(k, [1,1,1], np.zeros((3,3)), n)
        df_new.at[i, 'Initial covariance matrix'] = Sigma0
        counts_mean, counts_cov, meta = integrate_piecewise_with_labels(k, mu0, Sigma0, t0, t_label1, t_label2, t_end)
        for j, gate in enumerate(gates):
            df_new.at[i, f'Mean {gate}'] = counts_mean[j]
            df_new.at[i, f'Std {gate}'] = np.sqrt(counts_cov[j, j])
        df_new.at[i, 'Covariance matrix'] = counts_cov

    df_new.to_json('data/DL analytic cov noisy.json')

if __name__ == '__main__':
    # sweep()
    noisy_sweep()
