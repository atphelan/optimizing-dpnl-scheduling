import pandas as pd
import numpy as np

gates = ['EdU+BrdU-', 'EdU+BrdU+', 'EdU-BrdU+']

def find_s_optimum_from_counts(dt: int, edu_sp: int, dp: int, brdu_sp: int, print_only=False):
    '''
    Function takes the following:
    dt - the waiting time between labels used in a pilot dual nucleoside labeling assay
    edu_sp - the number of EdU single-positive cells counted as a raw number or percentage of total cells counted
    dp - the number of double-positive cells counted, processed as above
    brdu_sp - the number of BrdU single-positive cells counted, processed as above

    The function then returns a recommendation for the time interval to use, based on maximal Signal-to-Noise Ratio in our simulation data.
    Recommendations are based on the closest match to the ratios of EdU+BrdU-, EdU-BrdU+, and EdU+BrdU+ cells 
    observed in simulations of continuously proliferating cells.
    Caution should be taken when using the results on cell populations that do not match this profile.
    The tool does not perform parameter inference itself, nor any interpolation.
    '''
    assert (0 < dt and 13 > dt), 'Please enter a time to the nearest hour between 1 and 12.'
    assert edu_sp > 0
    assert dp > 0
    assert brdu_sp > 0
    relative_counts = np.array([edu_sp, dp, brdu_sp])
    total_cells = sum(relative_counts)
    relative_counts = relative_counts/total_cells

    # Reference simulations generated using the stochastic 3-stage cell-cycle model described in the manuscript,
    # spanning plausible parameter combinations and inter-label waiting times for a 24-hour cycle.
    reference_df = pd.read_json('DPNL inference results.json')

    lookup_slice = reference_df[reference_df['BrdU time']==dt].copy() # Waiting time is provided by user
    lookup_slice['Scaled Euclidean distance from obs'] = lookup_slice.apply(
        lambda lookup_row: np.sqrt(sum(
            [
                ((lookup_row[f'Proportion of labeled {gate}'] - count)/lookup_row[f'Proportion of labeled {gate}'])**2 for gate, count in zip(gates, relative_counts)
            ]
        )),
        axis=1
    )
    # Scaled Euclidean distance emphasizes relative proportions rather than absolute counts, helping to generalize the tool
    nearest_row = lookup_slice.iloc[lookup_slice['Scaled Euclidean distance from obs'].argmin(skipna=True)]
    comparable_series = reference_df[np.logical_and(np.isclose(reference_df['k1'], nearest_row['k1']), np.isclose(reference_df['k2'], nearest_row['k2']
                                                                                                                ))]
    max_snr = comparable_series['SNR k2'].max()
    dt_opt = comparable_series[comparable_series['SNR k2'] == max_snr]['BrdU time'].iloc[0]

    if print_only:
        readout(dt_opt, max_snr, dt)
    else:
        return dt_opt, max_snr

def find_s_optimum_from_phases(tg1: float, ts: float, tp: float, print_only=False):
    '''
    Function takes the following:
    tg1 - estimated duration of G1 phase
    ts - estimated duration of S phase
    tp - estimated duration of whole cell cycle
    Units are not important as long as they are consistent, but hours are expected.

    The function then returns a recommendation for the time interval to use, based on maximal Signal-to-Noise Ratio in our simulation data.
    Recommendations are based on the closest match to the ratios of tg1, ts, and tg2m (inferred from total period),
    observed in simulations of continuously proliferating cells.
    Caution should be taken when using the results on cell populations that do not match this profile.
    The tool does not perform parameter inference itself, nor any interpolation.
    '''
    assert 0 < tg1
    assert 0 < ts
    assert 0 < tp

    relative_times = np.array([tg1, ts, tp-tg1-ts])/tp

    # Reference simulations generated using the stochastic 3-stage cell-cycle model described in the manuscript,
    # spanning plausible parameter combinations and inter-label waiting times for a 24-hour cycle.
    reference_df = pd.read_json('DPNL inference results.json')

    lookup_slice = reference_df[reference_df['BrdU time']==2].copy() # Picking a specific waiting time to narrow down rows to fit
    lookup_slice['Scaled Euclidean distance from obs'] = lookup_slice.apply(
        lambda lookup_row: np.sqrt(sum(
            [
                ((lookup_row[phase] - t*24.0)/lookup_row[phase])**2 for phase, t in zip(['tg1', 'ts', 'tg2m'], relative_times)
            ]
        )),
        axis=1
    )
    # Scaled Euclidean distance emphasizes relative proportions rather than absolute counts, helping to generalize the tool
    nearest_row = lookup_slice.iloc[lookup_slice['Scaled Euclidean distance from obs'].argmin(skipna=True)]
    comparable_series = reference_df[np.logical_and(np.isclose(reference_df['k1'], nearest_row['k1']), np.isclose(reference_df['k2'], nearest_row['k2']
                                                                                                                ))]
    max_snr = comparable_series['SNR k2'].max()
    dt_opt = comparable_series[comparable_series['SNR k2'] == max_snr]['BrdU time'].iloc[0]
    dt_opt = int(round(dt_opt * tp/24))

    if print_only:
        readout(dt_opt, max_snr)
    else:
        return dt_opt, max_snr

def find_custom_optimum_from_counts(dt: int, edu_sp: int, dp: int, brdu_sp: int, g1_weight: float, s_weight: float, print_only=False):
    '''
    Function takes the following:
    dt - the waiting time between labels used in a pilot dual nucleoside labeling assay
    edu_sp - the number of EdU single-positive cells counted as a raw number or percentage of total cells counted
    dp - the number of double-positive cells counted, processed as above
    brdu_sp - the number of BrdU single-positive cells counted, processed as above
    g1_weight - the relative contribution of SNR k_g1 to the objective SNR
    s_weight - contribution of SNR k_S to the objective SNR.
    Note g1 inference should be prioritised if g2m phase time is needed - s phase is always well resolved in comparison so g2m can be inferred from g1 + s.
    
    The function then returns a recommendation for the time interval to use, based on maximal Signal-to-Noise Ratio in our simulation data.
    Recommendations are based on the closest match to the ratios of EdU+BrdU-, EdU-BrdU+, and EdU+BrdU+ cells 
    observed in simulations of continuously proliferating cells.
    Caution should be taken when using the results on cell populations that do not match this profile.
    The tool does not perform parameter inference itself, nor any interpolation.
    '''
    assert (0 < dt and 13 > dt), 'Please enter a time to the nearest hour between 1 and 12.'
    assert edu_sp > 0
    assert dp > 0
    assert brdu_sp > 0
    assert 0 <= g1_weight
    assert 0 <= s_weight
    assert g1_weight + s_weight > 0
    relative_counts = np.array([edu_sp, dp, brdu_sp])
    total_cells = sum(relative_counts)
    relative_counts = relative_counts/total_cells

    # Reference simulations generated using the stochastic 3-stage cell-cycle model described in the manuscript,
    # spanning plausible parameter combinations and inter-label waiting times for a 24-hour cycle.
    reference_df = pd.read_json('DPNL inference results.json')
    reference_df['Objective SNR'] = g1_weight*reference_df['SNR k1'] + s_weight*reference_df['SNR k2']


    lookup_slice = reference_df[reference_df['BrdU time']==dt].copy() # Waiting time is provided by user
    lookup_slice['Scaled Euclidean distance from obs'] = lookup_slice.apply(
        lambda lookup_row: np.sqrt(sum(
            [
                ((lookup_row[f'Proportion of labeled {gate}'] - count)/lookup_row[f'Proportion of labeled {gate}'])**2 for gate, count in zip(gates, relative_counts)
            ]
        )),
        axis=1
    )
    # Scaled Euclidean distance emphasizes relative proportions rather than absolute counts, helping to generalize the tool
    nearest_row = lookup_slice.iloc[lookup_slice['Scaled Euclidean distance from obs'].argmin(skipna=True)]
    comparable_series = reference_df[np.logical_and(np.isclose(reference_df['k1'], nearest_row['k1']), np.isclose(reference_df['k2'], nearest_row['k2']
                                                                                                                ))]
    max_snr = comparable_series['Objective SNR'].max()
    dt_opt = comparable_series[comparable_series['Objective SNR'] == max_snr]['BrdU time'].iloc[0]
    
    if print_only:
        readout(dt_opt, max_snr, dt)
    else:
        return dt_opt, max_snr


def find_custom_optimum_from_phases(tg1: float, ts: float, tp: float, g1_weight: float, s_weight: float, print_only=False):
    '''
    Function takes the following:
    tg1 - estimated duration of G1 phase
    ts - estimated duration of S phase
    tp - estimated duration of whole cell cycle
    g1_weight - the relative contribution of SNR k_g1 to the objective SNR
    s_weight - contribution of SNR k_S to the objective SNR.
    Note g1 inference should be prioritised if g2m phase time is needed - s phase is always well resolved in comparison so g2m can be inferred from g1 + s.
    
    Units for times are not important as long as they are consistent, but hours are expected.

    The function then returns a recommendation for the time interval to use, based on maximal Signal-to-Noise Ratio in our simulation data.
    Recommendations are based on the closest match to the ratios of tg1, ts, and tg2m (inferred from total period),
    observed in simulations of continuously proliferating cells.
    Caution should be taken when using the results on cell populations that do not match this profile.
    The tool does not perform parameter inference itself, nor any interpolation.
    '''
    assert 0 < tg1
    assert 0 < ts
    assert 0 < tp
    assert 0 <= g1_weight
    assert 0 <= s_weight
    assert g1_weight + s_weight > 0

    g1_weight, s_weight = np.array([g1_weight, s_weight])/sum([g1_weight, s_weight]) # Normalization

    relative_times = np.array([tg1, ts, tp-tg1-ts])/tp

    # Reference simulations generated using the stochastic 3-stage cell-cycle model described in the manuscript,
    # spanning plausible parameter combinations and inter-label waiting times for a 24-hour cycle.
    reference_df = pd.read_json('DPNL inference results.json')
    reference_df['Objective SNR'] = g1_weight*reference_df['SNR k1'] + s_weight*reference_df['SNR k2']

    lookup_slice = reference_df[reference_df['BrdU time']==2].copy() # Picking a specific waiting time to narrow down rows to fit
    lookup_slice['Scaled Euclidean distance from obs'] = lookup_slice.apply(
        lambda lookup_row: np.sqrt(sum(
            [
                ((lookup_row[phase] - t*24.0)/lookup_row[phase])**2 for phase, t in zip(['tg1', 'ts', 'tg2m'], relative_times)
            ]
        )),
        axis=1
    )
    # Scaled Euclidean distance emphasizes relative proportions rather than absolute counts, helping to generalize the tool
    nearest_row = lookup_slice.iloc[lookup_slice['Scaled Euclidean distance from obs'].argmin(skipna=True)]
    comparable_series = reference_df[np.logical_and(np.isclose(reference_df['k1'], nearest_row['k1']), np.isclose(reference_df['k2'], nearest_row['k2']
                                                                                                                ))]
    max_snr = comparable_series['Objective SNR'].max()
    dt_opt = comparable_series[comparable_series['Objective SNR'] == max_snr]['BrdU time'].iloc[0]
    dt_opt = int(round(dt_opt * tp/24))

    if print_only:
        readout(dt_opt, max_snr)
    else:
        return dt_opt, max_snr


def readout(dt_opt, max_snr, dt=0):

    print(f'A time of {dt_opt} hours is recommended, for which we found a SNR of {round(max_snr, 1)} for similar labeled cell proportions in our simulation study.')
    if dt == dt_opt:
        print('It looks like you already used the ideal timing!')


if __name__ == '__main__':
    ### DEMO ###
    print('1.')
    find_s_optimum_from_counts(dt=2, edu_sp=6.3, dp=23.0, brdu_sp=6.6, print_only=True) # Numbers as used in the publication Figure 1c, which are Leukemic Stem-Like cells in Bone marrow.
    # The tool recommends 4 hours, which would balance the DP and SP population sizes more.
    print('2.')
    find_s_optimum_from_counts(dt=4, edu_sp=1, dp=1, brdu_sp=1, print_only=True) # Trial numbers
    # The tool recommends 2 hours, which would give fewer SP cells but more DP cells.
    # This suggests a nontrivial Signal-to-Noise Ratio optimization trade-off not previously considered by researchers.
    print('3.')
    find_custom_optimum_from_counts(dt=2, edu_sp=6.3, dp=23.0, brdu_sp=6.6, g1_weight=1.0, s_weight=0.0, print_only=True) # Optimize for k_g1 instead
    print('4.')
    find_custom_optimum_from_counts(dt=2, edu_sp=6.3, dp=23.0, brdu_sp=6.6, g1_weight=0.7, s_weight=0.3, print_only=True) # Optimize for 70% k_g1, 30% k_s instead

    print('5.')
    find_s_optimum_from_counts(dt=2, edu_sp=3.22, dp=13.2, brdu_sp=4.75, print_only=True) # MPP cells from Akinduro et al., 2018.
    print('6.')
    find_s_optimum_from_counts(dt=2, edu_sp=1.18, dp=1.40, brdu_sp=1.33, print_only=True) # ST-HSC cells from Akinduro et al., 2018.
    print('7.')
    find_s_optimum_from_counts(dt=2, edu_sp=1.09, dp=6.10, brdu_sp=2.18, print_only=True) # HSC cells from Akinduro et al., 2018.
    # Note the large differences between EdU+BrdU- and EdU-BrdU+ cells - this implies many newly divided cells are differentiating or exiting the BM before harvest.
    # These large differences constitute an example when the optimization tool is not recommended for use.

    print('8.')
    find_s_optimum_from_phases(tg1=4, ts=4.5, tp=8.5, print_only=True) # Mean numbers for all cells in bone marrow (Frindel, Generation cycle of mouse bone marrow, 1967)
    # The tool recommends 2 hours, which conveniently is the most commonly used timing for studies on mouse cells.
    print('9.')
    find_custom_optimum_from_phases(tg1=4, ts=4.5, tp=8.5, g1_weight=1.0, s_weight=0.0, print_only=True) # Same numbers, now optimize k_g1
    print('10.')
    find_custom_optimum_from_phases(tg1=4, ts=4.5, tp=8.5, g1_weight=0.7, s_weight=0.3, print_only=True) # Same numbers, now optimize 70% k_g1 & 30% k_s

