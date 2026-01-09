import pandas as pd
import numpy as np

gates = ['EdU+BrdU-', 'EdU+BrdU+', 'EdU-BrdU+', 'EdU-BrdU-']

def find_s_optimum_from_reference(dt: int, edu_sp: int, dp: int, brdu_sp: int):
    assert (0 < dt and 13 > dt), 'Please enter a time to the nearest hour between 1 and 12.'
    assert edu_sp > 0
    assert dp > 0
    assert brdu_sp > 0
    relative_counts = np.array([edu_sp, dp, brdu_sp])
    total_cells = int(sum(relative_counts))
    relative_counts = relative_counts/total_cells

    reference_df = pd.read_json('DL bootstrap 11:07:2024-big concatted.json')

    lookup_slice = reference_df[reference_df['BrdU time']==dt].copy() # Waiting time is provided by user
    lookup_slice['Scaled Euclidean distance from obs'] = lookup_slice.apply(
        lambda lookup_row: np.sqrt(sum(
            [
                ((lookup_row[f'Proportion of labeled {gate}'] - count)/lookup_row[f'Proportion of labeled {gate}'])**2 for gate, count in zip(gates[:-1], relative_counts)
            ]
        )),
        axis=1
    )
    nearest_row = lookup_slice.iloc[lookup_slice['Scaled Euclidean distance from obs'].argmin(skipna=True)]
    comparable_series = reference_df[np.logical_and(np.isclose(reference_df['k1'], nearest_row['k1']), np.isclose(reference_df['k2'], nearest_row['k2']
                                                                                                                ))]
    max_snr = comparable_series['SNR k2'].max()
    dt_opt = comparable_series[comparable_series['SNR k2'] == max_snr]['BrdU time'].iloc[0]

    print(f'A time of {dt_opt} is recommended, for which we found a SNR of {round(max_snr, 1)} for similar labeled cell proportions in our simulation study.')
    if dt == dt_opt:
        print('It looks like you already used this timing!')

### DEMO ###
find_s_optimum_from_reference(2, 6.3, 23.0, 6.6) # Numbers as used in the publication Figure 1c.