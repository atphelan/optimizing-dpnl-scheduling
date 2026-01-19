'''
Analytic solver utilities for the 3-step Poisson process cell cycle
'''

import numpy as np
import pandas as pd
import os

from sympy import Symbol, Eq, Interval 
from sympy.solvers import nonlinsolve
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
layout = go.Layout(plot_bgcolor='white')

from gECC import DualLabelingEnsemble

class DPNL_sampler(DualLabelingEnsemble):
    def save_data(self, run_type):
        return self.end_states


gates = ['EdU+BrdU+', 'EdU-BrdU+', 'EdU+BrdU-', 'EdU-BrdU-']

def generate_steady_state(K2, K3, n, padding=0, include_squares=False):
    '''
    K2 = k2/k1, K3 = k3/k1
    '''
    x = Symbol('x')
    y = Symbol('y')
    dxdt = K3 * x**2 + K3 * x * y - (3*K3 + 1) * x - 2 * K3 * y + 2 * K3
    dydt = K3 * y**2 + K3 * x * y - (K2 + K3) * y + x
    soln = nonlinsolve([Eq(dxdt, 0), Eq(dydt, 0)], [x,y])
    # Should always be only fully real solution
    i = Interval(0, 1)
    for s in soln.args:
        if s[0].is_real and s[1].is_real:
            if i.contains(s[0]) and i.contains(s[1]):
                output = np.array([int(round(s[0]*n)), int(round(s[1]*n)), int(round((1-s[0]-s[1])*n))] + ([0]*6 if include_squares else []) + [0]*padding)
                if include_squares:
                    for i in range(3):
                        output[3 + 2*i] = output[i]**2
                        output[4 + 2*i] = output[i]*output[(i+1)%3]
                return output
    ValueError("No real solutions")

def get_n(v):
    return sum(v[0:3])

def get_var_s(v):
    return v[5] - v[1]**2

def get_var_n(v):
    out = sum(v[3::2] + 2*v[4::2]) - sum(v[0:3])**2
    if out < 0.0:
        raise ValueError('Negative variance!')
    return out

def get_covar_S_nonS(v):
    return v[4] + v[6] - v[1]*(v[0]+v[2])

def make_specific_solution_plot():
    k1 = 1/11
    k2 = 1/8
    k3 = 1/5
    t_sample_delay = 0.5

    dnsqdt = lambda t, n: [
        2*k3*n[2] - k1*n[0],
        k1*n[0] - k2*n[1],
        k2*n[1] - k3*n[2],
        (2*k3*n[2]*(n[0] + n[1] + n[2]) + k3*n[2])
    ]
    n0 = generate_steady_state(k2/k1, k3/k1, 300, 1) # g1, g2m, n, n^2 initial condition
    n0[-1] = 300
    n0 = list(n0)
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
    ])
    dalldt = lambda t, v: K.dot(v.T)
    all0 = n0[0:3] + [n0[0]**2] + [n0[0]*n0[1]] + [n0[1]**2] + [n0[1]*n0[2]] + [n0[2]**2] + [n0[2]*n0[0]]
    all0 = np.array(all0).astype(np.float64)

    # fig, ax = plt.subplots()

    # Import data to compare
    gates = ['EdU+BrdU+', 'EdU-BrdU+', 'EdU+BrdU-', 'EdU-BrdU-']
    colours = ['mediumpurple', 'orangered', 'dodgerblue', 'dimgrey']
    if 'utils' != os.getcwd()[-5:]:
        os.chdir('utils')
    # print('Reading file')
    df = pd.read_json('DPNL inference results.json')
    # print('k1 values: ', df['k1'].unique(), ';', len(df['k1'].unique()), 'unique values of k1.')
    # print('k2 values: ', df['k2'].unique(), ';', len(df['k2'].unique()), 'unique values of k2.')
    df['tg1'] = 1/df['k1']
    df['ts'] = 1/df['k2']
    df['tg2m'] = 1/df['k3']
    df['x'] = df['Initial G1']/(df['Initial G1']+df['Initial S']+df['Initial G2M'])
    df['y'] = df['Initial S']/(df['Initial G1']+df['Initial S']+df['Initial G2M'])
    df['z'] = df['Initial G2M']/(df['Initial G1']+df['Initial S']+df['Initial G2M'])
    #%%
    for k in ['k1', 'k2']:
        df[f'Relative error {k}'] = (df[k] - df[f'Mean inferred {k}'])/df[f'Std inferred {k}']
    df['Total cells at end'] = df['Mean EdU+BrdU+']+df['Mean EdU-BrdU+']+df['Mean EdU+BrdU-']+df['Mean EdU-BrdU-']
    df['Labelled fraction'] = 1 - df['Mean EdU-BrdU-']/df['Total cells at end']

    df = df.loc[np.logical_and(np.isclose(df['k1'], k1, 0.8E-2), np.isclose(df['k2'], k2, 0.8E-2))]

    analytic_df = pd.DataFrame([], columns=[
        'Waiting time', 'Mean EdU+BrdU+', 'Std EdU+BrdU+', 'Mean EdU-BrdU+', 'Std EdU-BrdU+',
        'Mean EdU+BrdU-', 'Std EdU+BrdU-', 'Mean EdU-BrdU-', 'Std EdU-BrdU-'
    ])
    for t_wait in np.linspace(0, 12.5, 101):
        # break
        # Set up experiment
        edu_sp = all0.copy()
        edu_sp[0] = 0.0; edu_sp[2:5] = 0.0; edu_sp[6:] = 0.0
        dn = (all0 - edu_sp).copy()
        dn[4:7] = 0.0
        brdu_sp = 0.0*all0.copy()
        dp = 0.0*all0.copy()

        # print("Initial EdU+ cells for {}".format(t_wait), " is ", edu_sp)
        # print("Initial EdU- cells for {}".format(t_wait), " is ", dn)

        # Find average trajectory of experiment with that setup
        edu_sol = solve_ivp(dalldt, [0.0, t_wait], edu_sp, max_step=0.05)
        edu_neg_sol = solve_ivp(dalldt, [0.0, t_wait], dn, max_step=0.05)

        # Sort BrdU staining step
        edu_sp = edu_sol.y.T[-1]
        dp[1] = edu_sp[1]; dp[5] = edu_sp[5]
        edu_sp[1] = 0.0; edu_sp[4:7] = 0.0
        dn = edu_neg_sol.y.T[-1]
        brdu_sp[1] = dn[1]; brdu_sp[5] = dn[5]
        dn[1] = 0.0; dn[4:7] = 0.0

        # print("Post-BrdU EdU-sp cells for {}".format(t_wait), " is ", edu_sp)
        # print("Post-BrdU dn cells for {}".format(t_wait), " is ", dn)
        # print("Post-BrdU BrdU-sp cells for {}".format(t_wait), " is ", brdu_sp)
        # print("Post-BrdU dp cells for {}".format(t_wait), " is ", dp)

        edu_sp_sol = solve_ivp(dalldt, [t_wait, t_wait+t_sample_delay], edu_sp, max_step=0.01)
        dn_sol = solve_ivp(dalldt, [t_wait, t_wait+t_sample_delay], dn, max_step=0.01)
        brdu_sp_sol = solve_ivp(dalldt, [t_wait, t_wait+t_sample_delay], brdu_sp, max_step=0.01)
        dp_sol = solve_ivp(dalldt, [t_wait, t_wait+t_sample_delay], dp, max_step=0.01)

        edu_sp = edu_sp_sol.y.T[-1]
        dn = dn_sol.y.T[-1]
        brdu_sp = brdu_sp_sol.y.T[-1]
        dp = dp_sol.y.T[-1]

        plot_fn = lambda cell_counts: get_var_n(cell_counts)**0.5
        # plot_fn = lambda cell_counts: get_n(cell_counts)

        analytic_df = pd.concat([analytic_df, pd.DataFrame([
            {
                'Waiting time': t_wait,
                'Mean EdU+BrdU+': get_n(dp),
                'Std EdU+BrdU+': get_var_n(dp)**0.5,
                'Mean EdU-BrdU+': get_n(brdu_sp),
                'Std EdU-BrdU+': get_var_n(brdu_sp)**0.5,
                'Mean EdU+BrdU-': get_n(edu_sp),
                'Std EdU+BrdU-': get_var_n(edu_sp)**0.5,
                'Mean EdU-BrdU-': get_n(dn),
                'Std EdU-BrdU-': get_var_n(dn)**0.5,
            }
        ])])

        # ax.scatter(t_wait, plot_fn(edu_sp), color='b', marker='.', label=None)
        # ax.scatter(t_wait, plot_fn(dn), color='k', marker='.', label=None)
        # ax.scatter(t_wait, plot_fn(brdu_sp), color='r', marker='.', label=None)
        # ax.scatter(t_wait, plot_fn(dp), color='m', marker='.', label=None)

        # ax.scatter(t_wait, get_n(dp), color='m', label=None)


        # ax.scatter(t_wait, np.sum([get_var_n(gate) for gate in [edu_sp, dn, brdu_sp, dp]]), color='cyan')

    #     # print(t_wait, get_var_n(edu_sp))
    #     # print(t_wait, get_var_n(dp))


    analytic_df.to_csv('analytic_dual_pulse_moments.csv')

    mean_fig = go.Figure(data=[go.Scatter(x=analytic_df['Waiting time'], y=analytic_df[f'Mean {gate}'], mode='lines',
        marker=dict(
            size=5,
            symbol='circle',
            color=c,
        ),
        line=dict(width=2.5),
        name = None,
        showlegend=False
    ) for gate, c in zip(gates, colours)], layout=layout)

    # ax.scatter(df['BrdU time'], df['Mean EdU+BrdU+'], marker='x', color='m')
    # ax.scatter(df['BrdU time'], df['Mean EdU+BrdU-'], marker='x', color='b')
    # ax.scatter(df['BrdU time'], df['Mean EdU-BrdU+'], marker='x', color='r')
    # ax.scatter(df['BrdU time'], df['Mean EdU-BrdU-'], marker='x', color='k')

    for gate, c in zip(gates, colours):
        mean_fig.add_trace(go.Scatter(x=df['BrdU time'], y=df[f'Mean {gate}'], mode='markers',
                marker=dict(
                    size=8,
                    line=dict(width=3, color=c),
                    symbol='x-thin',
                    color=c,
                ),
                name=gate,
                showlegend=True
        ))

    mean_fig.update_layout(
        xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
        yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
        xaxis_title=dict(text=None, font=dict(size=24)),
        yaxis_title=dict(text=None, font=dict(size=24)),
        autosize=False,
        height=400,
        width=500,
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest',
        font=dict(size=24),
        boxmode='group',
        # showlegend=True,
        legend=dict(yanchor="top",y=0.84,xanchor="right",x=0.97,bordercolor='dimgrey',borderwidth=2,font=dict(size=18)),
    )
    mean_fig.update_xaxes(
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        showgrid=True,
        gridwidth=1,
        gridcolor='gray',
        linewidth=1.5,
        linecolor='black',
        mirror=True,
        nticks=7,
        range=[0.0, 12.5],
        dtick=2,
    )
    mean_fig.update_yaxes(
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        showgrid=True,
        gridwidth=1,
        gridcolor='gray',
        nticks=10,
        linewidth=1.5,
        linecolor='black',
        mirror=True,
        # dtick=1,
        range=[0.0, 220.0],
    )
    mean_fig.show()

    std_fig = go.Figure(data=[go.Scatter(x=analytic_df['Waiting time'], y=analytic_df[f'Std {gate}'], mode='lines',
        marker=dict(
            size=5,
            symbol='circle',
            color=c,
        ),
        line=dict(width=2.5),
        name = None,
        showlegend=False
    ) for gate, c in zip(gates, colours)], layout=layout)

    for gate, c in zip(gates, colours):
        std_fig.add_trace(go.Scatter(x=df['BrdU time'], y=df[f'Std {gate}'], mode='markers',
                marker=dict(
                    size=8,
                    line=dict(width=3, color=c),
                    symbol='x-thin',
                    color=c,
                ),
                name=gate,
                showlegend=True
        ))

    std_fig.update_layout(
        xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
        yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=12),
        xaxis_title=dict(text=None, font=dict(size=24)),
        yaxis_title=dict(text=None, font=dict(size=24)),
        autosize=False,
        height=400,
        width=500,
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest',
        font=dict(size=24),
        boxmode='group',
        # showlegend=True,
        legend=dict(yanchor="bottom",y=0.03,xanchor="right",x=0.97,bordercolor='dimgrey',borderwidth=2,font=dict(size=18)),
    )
    std_fig.update_xaxes(
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        showgrid=True,
        gridwidth=1,
        gridcolor='gray',
        linewidth=1.5,
        linecolor='black',
        mirror=True,
        nticks=7,
        range=[0.0, 12.5],
        dtick=2,
    )
    std_fig.update_yaxes(
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='black',
        showgrid=True,
        gridwidth=1,
        gridcolor='gray',
        nticks=10,
        linewidth=1.5,
        linecolor='black',
        mirror=True,
        # dtick=1,
        range=[0.0, 11.0],
    )
    std_fig.show()
 

# Get mean and standard deviation of the number of each labelled cell type
def solve_from_k(k1, k2, k3, t_wait):
    # Set up model
    get_K = lambda k1, k2, k3: np.array([
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
    ])
    # Set up experiment, edge case checks
    check_noninstant = np.array([k1, k2, k3]) < 1E10
    if k1 < 1E10:
        K2 = k2/k1
        K3 = k3/k1
    else:
        K2 = np.nan
        K3 = np.nan
    if (check_noninstant).all():
        initial_state = generate_steady_state(K2, K3, n=300, padding=0, include_squares=True)
        K = get_K(k1, k2, k3)
    elif sum(check_noninstant) == 1: # two infinitely fast cell cycle phases
        initial_state = np.array([0.0]*9)
        nonzero_loc = np.where(check_noninstant == 1)[0][0]
        initial_state[nonzero_loc] = 300.0
        initial_state[3 + 2*nonzero_loc] = 90000.0
        K = np.zeros((9,9))
        K[nonzero_loc,nonzero_loc] = [k1, k2, k3][nonzero_loc]
        K[nonzero_loc*2 + 3,nonzero_loc*2 + 3] = [2*k1, 2*k2, 2*k3][nonzero_loc]
        K[nonzero_loc*2 + 3,nonzero_loc] = [k1, k2, k3][nonzero_loc]
    elif sum(check_noninstant) == 2: # One instant phase
        initial_state = np.array([0.0]*9)
        nonzero_locs = np.where(check_noninstant == 1)[0]
        pos_quad = lambda a, b, c: (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        initial_state[nonzero_locs[1]] = 300.0*pos_quad(*[[K2, 1+K2, -1],[K3, 1+K3, -1],[k3/k2, 1+k3/k2, -1]][sum(nonzero_locs)-1])
        initial_state[nonzero_locs[0]] = 300.0 - initial_state[nonzero_locs[1]]
        initial_state[3:] = [initial_state[0]**2, initial_state[0]*initial_state[1], initial_state[1]**2,
            initial_state[1]*initial_state[2], initial_state[2]**2, initial_state[2]*initial_state[0]
        ]
        K = get_K(*[[0, k2, k3], [k1, 0, k3], [k1, k2, 0]][np.where(check_noninstant == 0)[0][0]])
        zl = set([0,1,2]).difference(nonzero_locs).pop()
        K[zl] = 0; K[3+2*zl] = 0; K[3+(2*zl+1)%6] = 0; K[3+(2*zl-1)%6] = 0
        K = K.T
        K[zl] = 0; K[3+2*zl] = 0; K[3+(2*zl+1)%6] = 0; K[3+(2*zl-1)%6] = 0
        K = K.T
        if zl == 0:
            K[1,2] = 2*k3
            K[5,6] = 4*k3; K[5,2] = 4*k3 # N_S^2
            K[6,7] = 2*k3; K[6,2] = -2*k3 # N_S*N_G2M
            # print('Sum K: ', np.sum(K))
        elif zl == 1: # Same as above but 1 -> 0.
            K[2,0] = k1
            K[7,0] = k1; K[7,8] = 2*k1 # N_G2M^2
            K[8,3] = k1; K[8,0] = -k1 # N_G2M*N_G1
            # print('Sum K: ', np.sum(K))
        elif zl == 2:
            K[0,1] = 2*k2
            K[3,2] = 4*k2; K[3,4] = 4*k2
            K[5,5] = 2*k2 # I hope I guessed that right.
            K[4,1] = -2*k2; K[4,4] = -k1-k2; K[4,5] = 2*k2
            # print('Sum K: ', np.sum(K))
        else:
            raise ValueError("Expected instantaneous phase not properly defined")
    else:
        raise ValueError('Entire cycle is instantaneous')
    if np.sum(K > 1E308) > 0:
        raise ValueError('Failed to reduce dimension of problem, please debug.')
    # print(np.round(K, 3))
    
    
    dalldt = lambda t, v: K.dot(v.T)
    edu_sp = initial_state.copy()
    edu_sp[0] = 0.0; edu_sp[2:5] = 0.0; edu_sp[6:] = 0.0 # Clear everything except S phase and S phase squared
    dn = (initial_state - edu_sp).copy() # Everything non-edu-single-positive is double negative for now
    dn[4:7] = 0.0 # S-phase inclusive squared quantities
    brdu_sp = 0.0*initial_state.copy()
    dp = 0.0*initial_state.copy()

    # print("Initial number of EdU+ cells for {}".format((k1, k2, k3)), " is ", sum(edu_sp))
    # print("Initial number of EdU- cells for {}".format((k1, k2, k3)), " is ", sum(dn))

    # Find average trajectory of experiment with that setup
    edu_sol = solve_ivp(dalldt, [0.0, t_wait], edu_sp, method='Radau')#, max_step=0.05)
    edu_neg_sol = solve_ivp(dalldt, [0.0, t_wait], dn, method='Radau')#, max_step=0.05)

    # Sort BrdU staining step
    edu_sp = edu_sol.y.T[-1] # Final state in solution series for edu+ cells
    dp[1] = edu_sp[1]; dp[5] = edu_sp[5] # take S phase cells over to be double positive
    edu_sp[1] = 0.0; edu_sp[4:7] = 0.0 # Remove them from single positive
    dn = edu_neg_sol.y.T[-1]
    brdu_sp[1] = dn[1]; brdu_sp[5] = dn[5] # Label S phase unlabelled cells with brdu
    dn[1] = 0.0; dn[4:7] = 0.0 # Remove from source
    t_sample_delay = 0.5

    edu_sp_sol = solve_ivp(dalldt, [t_wait, t_wait+t_sample_delay], edu_sp, method='Radau')#, max_step=0.01)
    dn_sol = solve_ivp(dalldt, [t_wait, t_wait+t_sample_delay], dn, method='Radau')#, max_step=0.01)
    brdu_sp_sol = solve_ivp(dalldt, [t_wait, t_wait+t_sample_delay], brdu_sp, method='Radau')#, max_step=0.01)
    dp_sol = solve_ivp(dalldt, [t_wait, t_wait+t_sample_delay], dp, method='Radau')#, max_step=0.01)

    edu_sp = edu_sp_sol.y.T[-1]
    dn = dn_sol.y.T[-1]
    brdu_sp = brdu_sp_sol.y.T[-1]
    dp = dp_sol.y.T[-1]

    # print(k1, k2, k3, 'dn = ', np.round(dn, 1))

    # plot_fn = lambda cell_counts: get_var_n(cell_counts)**0.5
    analytic_df = pd.DataFrame([{'k1': k1, 'k2': k2, 'k3': k3, 'Waiting time': t_wait}])
    analytic_df = pd.concat([analytic_df, pd.DataFrame([
        {
            'Starting G1': initial_state[0],
            'Starting S': initial_state[1],
            'Starting G2M': initial_state[2],
            'Mean EdU+BrdU+': get_n(dp),
            'Std EdU+BrdU+': get_var_n(dp)**0.5,
            'Mean EdU-BrdU+': get_n(brdu_sp),
            'Std EdU-BrdU+': get_var_n(brdu_sp)**0.5,
            'Mean EdU+BrdU-': get_n(edu_sp),
            'Std EdU+BrdU-': get_var_n(edu_sp)**0.5,
            'Mean EdU-BrdU-': get_n(dn),
            'Std EdU-BrdU-': get_var_n(dn)**0.5,
        }
    ])], axis='columns')
    # M_obs = np.array([analytic_df[f'Mean {gate}'] for gate in gates])
    # counts_cov = M_obs @ Sigma_final @ M_obs.T

    return analytic_df


if __name__ == '__main__':
    print(solve_from_k(0.087656, 0.103810, 0.337980, 4.0))

