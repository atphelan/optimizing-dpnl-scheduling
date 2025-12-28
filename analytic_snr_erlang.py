import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from gECC import generate_steady_state_erlang

gates = ['EdU+BrdU-', 'EdU-BrdU+', 'EdU+BrdU+', 'EdU-BrdU-']


def build_base_matrices_erlang(n_steps):
    """
    Build stoichiometry S and source indices for a 3-phase Erlang chain.
    States: G1_1..G1_n, S_1..S_n, G2M_1..G2M_n
    """
    n_states = 3 * n_steps
    S = []
    sources = []

    def idx(phase, step):
        return phase * n_steps + step

    # G1 chain
    for i in range(n_steps - 1):
        v = np.zeros(n_states)
        v[idx(0, i)] -= 1
        v[idx(0, i + 1)] += 1
        S.append(v)
        sources.append(idx(0, i))

    # G1 -> S
    v = np.zeros(n_states)
    v[idx(0, n_steps - 1)] -= 1
    v[idx(1, 0)] += 1
    S.append(v)
    sources.append(idx(0, n_steps - 1))

    # S chain
    for i in range(n_steps - 1):
        v = np.zeros(n_states)
        v[idx(1, i)] -= 1
        v[idx(1, i + 1)] += 1
        S.append(v)
        sources.append(idx(1, i))

    # S -> G2M
    v = np.zeros(n_states)
    v[idx(1, n_steps - 1)] -= 1
    v[idx(2, 0)] += 1
    S.append(v)
    sources.append(idx(1, n_steps - 1))

    # G2M chain
    for i in range(n_steps - 1):
        v = np.zeros(n_states)
        v[idx(2, i)] -= 1
        v[idx(2, i + 1)] += 1
        S.append(v)
        sources.append(idx(2, i))

    # division
    v = np.zeros(n_states)
    v[idx(2, n_steps - 1)] -= 1
    v[idx(0, 0)] += 2
    S.append(v)
    sources.append(idx(2, n_steps - 1))

    return np.column_stack(S), np.array(sources)


def build_A_from_rates_erlang(k, n_steps):
    """
    Build linear generator A for Erlang chain.
    k = (k1, k2, k3)
    """
    k1, k2, k3 = k
    rates = np.concatenate([
        np.full(n_steps, k1 * n_steps),
        np.full(n_steps, k2 * n_steps),
        np.full(n_steps, k3 * n_steps),
    ])

    n = 3 * n_steps
    A = np.zeros((n, n))

    for i in range(n - 1):
        A[i, i] -= rates[i]
        A[i + 1, i] += rates[i]

    # division
    A[0, -1] += 2 * rates[-1]
    A[-1, -1] -= rates[-1]

    return A


def expand_to_labels(S, sources, n_labels):
    S_big = np.kron(np.eye(n_labels), S)
    sources_big = np.concatenate([
        sources + i * S.shape[0] for i in range(n_labels)
    ])
    return S_big, sources_big


def build_label1_mapping_erlang(n_steps):
    """
    Map 1-label system -> 2-label system.
    Label-1 applied to all S substates.
    """
    block = 3 * n_steps
    P = np.zeros((2 * block, block))

    # label-0 block (unchanged G1, G2M)
    # G1
    P[0:n_steps, 0:n_steps] = np.eye(n_steps)
    # G2M
    P[2*n_steps:3*n_steps, 2*n_steps:3*n_steps] = np.eye(n_steps)

    # label-1 block (S states only)
    P[block + n_steps : block + 2*n_steps,
      n_steps : 2*n_steps] = np.eye(n_steps)

    return P


def build_label2_mapping_after_label1_erlang(n_steps):
    """
    Map 2-label system -> 4-label system.
    Label-2 applied to all S substates.
    """
    block = 3 * n_steps
    P = np.zeros((4 * block, 2 * block))

    # ---- from label 00 ----
    # G1
    P[0:n_steps, 0:n_steps] = np.eye(n_steps)
    # G2M
    P[2*n_steps:3*n_steps, 2*n_steps:3*n_steps] = np.eye(n_steps)
    # S -> 01
    P[block + n_steps : block + 2*n_steps,
      n_steps : 2*n_steps] = np.eye(n_steps)

    # ---- from label 10 ----
    offset = block
    out = 2 * block
    # G1
    P[out : out + n_steps,
      offset : offset + n_steps] = np.eye(n_steps)
    # G2M
    P[out + 2*n_steps : out + 3*n_steps,
      offset + 2*n_steps : offset + 3*n_steps] = np.eye(n_steps)
    # S -> 11
    P[out + block + n_steps : out + block + 2*n_steps,
      offset + n_steps : offset + 2*n_steps] = np.eye(n_steps)

    return P


def build_observation_mapping_erlang(n_steps):
    block = 3 * n_steps
    M = np.zeros((4, 4 * block))

    # EdU+ BrdU-
    M[0, 2*block : 3*block] = 1.0
    # EdU- BrdU+
    M[1, block : 2*block] = 1.0
    # EdU+ BrdU+
    M[2, 3*block : 4*block] = 1.0
    # EdU- BrdU-
    M[3, 0:block] = 1.0

    return M


def integrate_piecewise_with_labels_erlang(
    k, n_steps, mu0, Sigma0,
    t0, t_label1, t_label2, t_end, A_cache
):
    N = 3 * n_steps

    mu0 = np.asarray(mu0, float).ravel()
    Sigma0 = np.asarray(Sigma0, float)

    if mu0.size != N:
        raise ValueError(f"mu0 must have length {N}, got {mu0.size}")

    if Sigma0.shape != (N, N):
        raise ValueError(f"Sigma0 must be shape {(N, N)}, got {Sigma0.shape}")

    S_base, sources_base = build_base_matrices_erlang(n_steps)
    A_base = A_cache[1]

    def integrate_segment(mu0, Sigma0, A, S, sources, rates, t_a, t_b):
        n = mu0.size

        def ode(t, y):
            mu = y[:n]
            Sigma = y[n:].reshape(n, n)

            dmu = A @ mu

            a = rates * np.maximum(mu[sources], 0.0)
            # D = S @ np.diag(a) @ S.T
            D = np.zeros((n, n))
            for r, ar in enumerate(a):
                if ar != 0.0:
                    sr = S[:, r]
                    D += ar * np.outer(sr, sr)


            dSigma = A @ Sigma + Sigma @ A.T + D
            return np.concatenate([dmu, dSigma.ravel()])

        y0 = np.concatenate([mu0, Sigma0.ravel()])
        sol = solve_ivp(ode, (t_a, t_b), y0, method="BDF")
        yf = sol.y[:, -1]

        mu_f = yf[:n]
        Sigma_f = yf[n:].reshape(n, n)
        return mu_f, 0.5 * (Sigma_f + Sigma_f.T)

    # rates per reaction channel
    k1, k2, k3 = k
    rates_base = np.concatenate([
        np.full(n_steps - 1, k1 * n_steps),
        [k1 * n_steps],
        np.full(n_steps - 1, k2 * n_steps),
        [k2 * n_steps],
        np.full(n_steps - 1, k3 * n_steps),
        [k3 * n_steps],
    ])

    # segment 0
    mu, Sigma = integrate_segment(mu0, Sigma0,
                                  A_base, S_base,
                                  sources_base, rates_base,
                                  t0, t_label1)

    # apply label 1
    P1 = build_label1_mapping_erlang(n_steps)
    mu = P1 @ mu
    Sigma = P1 @ Sigma @ P1.T

    # segment 1
    S1, src1 = expand_to_labels(S_base, sources_base, 2)
    A1 = A_cache[2]
    rates1 = np.tile(rates_base, 2)

    mu, Sigma = integrate_segment(mu, Sigma, A1, S1, src1, rates1,
                                  t_label1, t_label2)

    # apply label 2
    P2 = build_label2_mapping_after_label1_erlang(n_steps)
    mu = P2 @ mu
    Sigma = P2 @ Sigma @ P2.T

    # final segment
    S2, src2 = expand_to_labels(S_base, sources_base, 4)
    A2 = A_cache[4]
    rates2 = np.tile(rates_base, 4)

    mu, Sigma = integrate_segment(mu, Sigma, A2, S2, src2, rates2,
                                  t_label2, t_end)

    M_obs = build_observation_mapping_erlang(n_steps)
    return (
        M_obs @ mu,
        M_obs @ Sigma @ M_obs.T,
        {"mu_final": mu, "Sigma_final": Sigma}
    )


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


def sweep_erlang(n_steps):
    df_gill = pd.read_csv('data/DL lookup 2025-12-19 erlang 4.csv').sample(frac=1).reset_index(drop=True)
    n = sum(df_gill[['Initial G1', 'Initial S', 'Initial G2M']].iloc[0])
    df_new = df_gill[['k1', 'k2', 'k3', 'BrdU time', 'Initial G1', 'Initial S', 'Initial G2M']].copy()
    df_new['Covariance matrix'] = None
    l = len(df_gill)

    n_steps_progress = 10  # report 10%,20%,...,100%
    progress_thresholds = {round(l * p / n_steps_progress): p*10 for p in range(1, n_steps_progress+1)}
    next_progress = min(progress_thresholds.keys()) if progress_thresholds else None

    print('Beginning iteration over parameter space...')
    for i, df_row in df_new.iterrows():

        if next_progress is not None and i >= next_progress:
            print(f'{progress_thresholds[next_progress]}% of parameter combinations complete.')
            del progress_thresholds[next_progress]
            next_progress = min(progress_thresholds.keys()) if progress_thresholds else None

        k = df_row[['k1', 'k2', 'k3']].to_numpy()
        A_base = build_A_from_rates_erlang(k, n_steps)
        A_cache = {
            1: csr_matrix(A_base),
            2: csr_matrix(np.kron(np.eye(2), A_base)),
            4: csr_matrix(np.kron(np.eye(4), A_base)),
        }
        mu0 = generate_steady_state_erlang(k[1]/k[0], k[2]/k[0], n, n_steps, padding=0)
        t0 = t_label1 = 0.0
        t_label2 = df_row['BrdU time']
        t_end = t_label2 + 0.5

        counts_mean, counts_cov, meta = integrate_piecewise_with_labels_erlang(
            k, n_steps, mu0, np.zeros((3*n_steps, 3*n_steps)), t0, t_label1, t_label2, t_end, A_cache
        )

        for j, gate in enumerate(gates):
            df_new.at[i, f'Mean {gate}'] = counts_mean[j]
            df_new.at[i, f'Std {gate}'] = np.sqrt(counts_cov[j, j])
        df_new.at[i, 'Covariance matrix'] = counts_cov

    print('Finished integrating, data ready.')
    df_new.to_json(f'data/DL analytic cov erlang {n_steps}.json')



if __name__ == '__main__':
    sweep_erlang(4)
