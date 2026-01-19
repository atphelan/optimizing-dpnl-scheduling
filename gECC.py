'''
Dual Pulse Nucleoside Labeling experiments - a gillespie-simulated Eukaryotic Cell Cycle
Experiment - Single simulated dual label process
Ensemble - Collection of repeat experiments with same parameters
Batch - Collection of ensembles over different parameters, 1 ensemble per parameter combination
Bootstrap - Collection of ensembles, n_experiments ensembles per parameter combination, user-defined repeats per ensemble.

Batch builds a lookup table which Bootstrap then uses to infer the distribution of k values that can be inferred from noisy dual labeling data.
'''
import itertools
from gillespie import ReactionEnsemble
import numpy as np
import pandas as pd
import os
from datetime import datetime
from scipy.optimize import fsolve

'''
Class files
'''

class DualLabelingEnsemble(ReactionEnsemble):

    def _reset(self, t_0, x_0, noisy_initial=False, cov=None):

        if noisy_initial:
            # self.run_until_n_for_initial(ps['n']) # Sets self.x inside
            # self.x[0:3] += np.multiply(np.random.randn(3), np.sqrt(self.x[0:3])).round().astype(int)
            fluctuation = np.random.multivariate_normal([0.0, 0.0, 0.0], cov)*self.alpha
            self.x = self.x_0 + np.pad(fluctuation, (0, 9), 'constant').round().astype(int)
            assert (self.x >= 0).all(), 'Negative terms in initial state! Reduce noise contribution.'
            self.t = t_0
        else:
            self.x = x_0
            self.t = t_0

    def _species(self):
        # 3 stages: G_1 (prep for S), S (Chromosome duplication), G_2 (prep for M), M (Mitosis, cytosis).
        # The latter two are short and grouped together. Cell type (HSPC, CPC, MC) not specified.
        return [r'$G_1 (EdU^-BrdU^-)$', r'$S (EdU^-BrdU^-)$', r'$G_2M (EdU^-BrdU^-)$', 
            r'$G_1 (EdU^+BrdU^-)$', r'$S (EdU^+BrdU^-)$', r'$G_2M (EdU^+BrdU^-)$',
            r'$G_1 (EdU^+BrdU^+)$', r'$S (EdU^+BrdU^+)$', r'$G_2M (EdU^+BrdU^+)$',
            r'$G_1 (EdU^-BrdU^+)$', r'$S (EdU^-BrdU^+)$', r'$G_2M (EdU^-BrdU^+)$'
        ]


    def _setup_extras(self, extras: dict):
        '''
        Gillespie doesn't directly track individuals' states.
        So stained populations will be tracked separately but with the same dynamics.
        '''
        # self.x is a 12-element numpy array of (clear, blue, purple, red) x (G1, S, G2M) subpopulations
        self.k1 = extras['k1']
        self.k2 = extras['k2']
        self.edu_time = extras['edu_t'] # Will set to np.inf after staining to avoid re-staining and to minimise variables
        self.brdu_time = extras['brdu_t'] # likewise
        self.measurement_time = extras['finish_t'] # end of simulation here
        self.cycle_period = extras['cycle_period']
        self.k3 = (self.cycle_period - 1/self.k1 - 1/self.k2)**-1


    def _stoichiometry(self):
        '''
        3 conversions:
        G_1 -> S begin chromosome duplication
        S -> G_2M prep for and perform mitosis (grouped since the latter is fast)
        G_2M -> G_1 + G_1 divide into two cells and begin another cycle

        This happens for all four possible stainings, and at the same rates.
        The actual instantiation of a stained population is handled directly, not by reaction.
        '''
        return [
            {'name': r'$G_1 \rightarrow S$ (unstained)', 'stoich': (np.array([1,0,0, 0,0,0,0,0,0,0,0,0]), np.array([0,1,0, 0,0,0,0,0,0,0,0,0])), 'rate': self.k1},
            {'name': r'$S \rightarrow G_2M$ (unstained)', 'stoich': (np.array([0,1,0, 0,0,0,0,0,0,0,0,0]), np.array([0,0,1, 0,0,0,0,0,0,0,0,0])), 'rate': self.k2},
            {'name': r'$G_2M \rightarrow 2G_1$ (unstained)', 'stoich': (np.array([0,0,1, 0,0,0,0,0,0,0,0,0]), np.array([2,0,0, 0,0,0,0,0,0,0,0,0])), 'rate': self.k3},

            {'name': r'$G_1 \rightarrow S$ (blue)', 'stoich': (np.array([0,0,0, 1,0,0, 0,0,0,0,0,0]), np.array([0,0,0, 0,1,0, 0,0,0,0,0,0])), 'rate': self.k1},
            {'name': r'$S \rightarrow G_2M$ (blue)', 'stoich': (np.array([0,0,0, 0,1,0, 0,0,0,0,0,0]), np.array([0,0,0, 0,0,1, 0,0,0,0,0,0])), 'rate': self.k2},
            {'name': r'$G_2M \rightarrow 2G_1$ (blue)', 'stoich': (np.array([0,0,0, 0,0,1, 0,0,0,0,0,0]), np.array([0,0,0, 2,0,0, 0,0,0,0,0,0])), 'rate': self.k3},

            {'name': r'$G_1 \rightarrow S$ (purple)', 'stoich': (np.array([0,0,0,0,0,0, 1,0,0, 0,0,0]), np.array([0,0,0,0,0,0, 0,1,0, 0,0,0])), 'rate': self.k1},
            {'name': r'$S \rightarrow G_2M$ (purple)', 'stoich': (np.array([0,0,0,0,0,0, 0,1,0, 0,0,0]), np.array([0,0,0,0,0,0, 0,0,1, 0,0,0])), 'rate': self.k2},
            {'name': r'$G_2M \rightarrow 2G_1$ (purple)', 'stoich': (np.array([0,0,0,0,0,0, 0,0,1, 0,0,0]), np.array([0,0,0,0,0,0, 2,0,0, 0,0,0])), 'rate': self.k3},

            {'name': r'$G_1 \rightarrow S$ (red)', 'stoich': (np.array([0,0,0,0,0,0,0,0,0, 1,0,0]), np.array([0,0,0,0,0,0,0,0,0, 0,1,0])), 'rate': self.k1},
            {'name': r'$S \rightarrow G_2M$ (red)', 'stoich': (np.array([0,0,0,0,0,0,0,0,0, 0,1,0]), np.array([0,0,0,0,0,0,0,0,0, 0,0,1])), 'rate': self.k2},
            {'name': r'$G_2M \rightarrow 2G_1$ (red)', 'stoich': (np.array([0,0,0,0,0,0,0,0,0, 0,0,1]), np.array([0,0,0,0,0,0,0,0,0, 2,0,0])), 'rate': self.k3}
            ]


    def _name(self):
        return "Double-Staining Experiment of the Eukaryotic Cell Cycle"


    def _tag_edu(self):
        '''
        Stain all cells in S phase to become EdU+.
        '''
        if np.isnan(self.x[3:6]).any():
            self.x[3] = 0
            self.x[5] = 0
            self.x[4] = self.x[1]
        else:
            self.x[4] += self.x[1]            
        self.x[1] = 0


    def _tag_brdu(self, i=1):
        '''
        Stain all cells in S phase to become BrdU+, accounting for cells already stained with EdU.
        '''
        self.pre_brdu_states[i] = self.x.copy()
        if np.isnan(self.x[6:]).any():
            self.x[6] = 0
            self.x[8] = 0
            self.x[9] = 0
            self.x[11] = 0
            self.x[7] = self.x[4]
            self.x[10] = self.x[1]
        else:
            self.x[7] += self.x[4]
            self.x[10] += self.x[1]
        self.x[4] = 0
        self.x[1] = 0

    # def run_until_n_for_initial(self, n=300):
    #     self.x[0:3] = self.x[0:3] * 0.8 # Allow the randomised start state to grow from 80% of the analytic steady state
    #     self.x[3:] = 0
    #     self.x = self.x.astype(int)
    #     while int(sum(self.x)) < n:
    #         _, j = self._tau_j()
    #         if not (j is None):
    #             s = self.stoich[j]['stoich']
    #             self.x += s[1] - s[0]
    #     # print(self.x)

    def run_simulation(self, i=1, i_max=1):
        '''
        Needs to be dynamic if experiment run time is fixed, not iterations.
        i is the index of the simulation, useful for data recording.
        '''
        self.x_series = [self.x.copy()]
        self.t_series = [self.t]
        while self.t < self.measurement_time:
            tau, j = self._tau_j()
            old_t = self.t # Should not update with self.t
            self.t += tau
            if not (j is None):
                s = self.stoich[j]['stoich']
                self.x += s[1] - s[0]
            if self.t > self.edu_time and old_t <= self.edu_time:
                self._tag_edu()
            elif self.t > self.brdu_time and old_t <= self.brdu_time:
                self._tag_brdu(i)
            self.t_series.append(self.t)
            self.x_series.append(self.x.copy())


    def run_ensemble(self, repeats, run_type='lookup', noisy_initial=False, cov=None):
        self.repeats = repeats
        self.pre_brdu_states = np.zeros((repeats, len(self.x)))
        self.end_states = np.zeros((repeats, len(self.x)))
        for i in range(repeats):
            self.run_simulation(i, repeats)
            self.end_states[i] = self.x.copy()
            self._reset(self.t_series[0], self.x_series[0], noisy_initial, cov)
        if run_type == 'small_trial':
            return self.save_data(run_type)
        self.save_data(run_type)


    def save_data(self, run_type):
        import pandas as pd
        if os.path.exists('double_staining_figs/ensemble_log.pkl'):
            df = pd.read_pickle('double_staining_figs/ensemble_log.pkl')
        else:
            df = pd.DataFrame([], columns = ["k1", "k2", "avg(k1)", "avg(k2)", "std(k1)", "std(k2)", "Initial G1", "Initial S", "Initial G2M", 
                "Lead-in time", "G1 at EdU time", "S at EdU time", "G2M at EdU time", "EdU time", "BrdU time", "Histogram data", "Whole trajectory"])
        dl_outputs = [np.sum(state.T[i*3:(i+1)*3], axis=0) for i in range(0,4) for state in self.end_states]
        new_row = [{
            "k1": self.k1,
            "k2": self.k2,
            "Cycle period": self.cycle_period,
            "Initial G1": self.initials[1][0],
            "Initial S": self.initials[1][1],
            "Initial G2M": self.initials[1][2],
            "mean EdU-BrdU-": np.mean(dl_outputs[0]),
            "mean EdU+BrdU-": np.mean(dl_outputs[1]),
            "mean EdU+BrdU+": np.mean(dl_outputs[2]),
            "mean EdU-BrdU+": np.mean(dl_outputs[3]),
            "EdU time": self.edu_time,
            "BrdU time": self.brdu_time,
            "Measurement time": self.measurement_time,
            "mean pre-BrdU state": tuple(np.mean(self.pre_brdu_state, axis=1)),
            "mean end state": tuple(np.mean(self.end_state, axis=0)),
            "std pre-BrdU state": tuple(np.std(self.pre_brdu_state, axis=1)),
            "std end state": tuple(np.std(self.end_state, axis=0)),
        }]
        df = pd.concat([df, pd.DataFrame(new_row)])
        df.to_pickle('double_staining_figs/ensemble_log.pkl')


class DualLabelingBatch(DualLabelingEnsemble):

    def save_data(self, run_type='lookup'):
        new_row = {
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "Cycle period": self.cycle_period,
            "Initial G1": self.x_series[0][0],
            "Initial S": self.x_series[0][1],
            "Initial G2M": self.x_series[0][2],
            "EdU time": self.edu_time,
            "BrdU time": self.brdu_time,
            "Wait time": self.measurement_time - self.brdu_time,
            "Repeats": self.repeats
        }
        gates = ['EdU-BrdU-', 'EdU+BrdU-', 'EdU+BrdU+', 'EdU-BrdU+']
        for i, gate in zip(range(0, 10, 3), gates):
            new_row.update({f"Mean {gate}": np.mean(np.sum(self.end_states[:, i:i+3], axis=1))})
            new_row.update({f"Std {gate}": np.std(np.sum(self.end_states[:, i:i+3], axis=1))})
        for i in range(12):
            new_row.update({f"Mean cells in {i} at end": np.mean(self.end_states[:,i])})
            new_row.update({f"Std cells in {i} at end": np.std(self.end_states[:,i])})
            if i<6:
                new_row.update({f"Mean cells in {i} at BrdU time": np.mean(self.pre_brdu_states[:,i])})
                new_row.update({f"Std cells in {i} at BrdU time": np.std(self.pre_brdu_states[:,i])})
        correlation_columns = [c for c in new_row.keys() if ('Mean' in c or 'Std' in c)]
        if run_type=='bootstrap':
            new_row.update({"Repeat Ensembles": self.repeat_ensembles})
            k1, k2, tp = self.infer_k(new_row)
            new_row.update({'Mean inferred k1': k1, 'Mean inferred k2': k2, 'Mean inferred cycle period': tp})
        self.ensemble_data = pd.concat([self.ensemble_data, pd.DataFrame([new_row])], ignore_index=True)


    def _setup_extras(self, extras):
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.jobs = extras[0]

    def _edit_stoichiometry(self):
        for i, s in enumerate(self.stoich):
            if "G_1 \\rightarrow" in s['name']:
                self.stoich[i]['rate'] = self.k1
            if "S \\rightarrow" in s['name']:
                self.stoich[i]['rate'] = self.k2
            if "G_2M \\rightarrow" in s['name']:
                self.stoich[i]['rate'] = self.k3

    def _initialise_ensemble(self, ps, noisy_initial=False, cov=None):
        self.k1 = ps['k1']
        self.k2 = ps['k2']
        self.k3 = ps['k3']
        self.start_n = ps['n']
        self.edu_time = 0 # Will set to np.inf after staining to avoid re-staining and to minimise variables
        self.brdu_time = ps['brdu_t'] # likewise
        self.measurement_time = ps['wait_t']+ps['brdu_t'] # end of simulation here
        self.cycle_period = (1/self.k1 + 1/self.k2 + 1/self.k3)
        self._edit_stoichiometry() # set rates for this run
        self.x_0 = generate_steady_state(ps['k2']/ps['k1'], ps['k3']/ps['k1'], ps['n'], padding=9)
        self.x = self.x_0.copy()
        assert (self.x_0 >= 0).all(), 'Negative terms in steady state!'
        if noisy_initial:
            # self.run_until_n_for_initial(ps['n']) # Sets self.x inside
            # self.x[0:3] += np.multiply(np.random.randn(3), np.sqrt(self.x[0:3])).round().astype(int)
            self.x[0:3] += np.random.multivariate_normal([0.0, 0.0, 0.0], cov)*self.alpha
            assert (self.x >= 0).all(), 'Negative terms in initial state! Reduce noise contribution.'
        self.x = self.x.round().astype(int)
        self.x_0 = self.x_0.round().astype(int)
        self.x_series = [self.x.copy()]

    def _reset(self, t_0, x_0, noisy_initial=False, cov=None):

        if noisy_initial:
            # self.run_until_n_for_initial(ps['n']) # Sets self.x inside
            # self.x[0:3] += np.multiply(np.random.randn(3), np.sqrt(self.x[0:3])).round().astype(int)
            fluctuation = np.random.multivariate_normal([0.0, 0.0, 0.0], cov)*self.alpha
            self.x = self.x_0 + np.pad(fluctuation, (0, 9), 'constant').round().astype(int)
            assert (self.x >= 0).all(), 'Negative terms in initial state! Reduce noise contribution.'
            self.t = t_0
        else:
            self.x = x_0
            self.t = t_0


    def run_batch(self, batch_parameters, repeats, n_parameter_combos, noisy_initial=False, cov_file = None):
        '''
        Run a collection of ensembles of simulations with varying parameters.
        '''
        self.ensemble_data = pd.DataFrame([])
        report_after_n = n_parameter_combos//10 if n_parameter_combos >= 10 else 1
        print("Starting at datetime {}...".format(str(datetime.now())))

        try:
            param_index = int(os.environ['PBS_ARRAY_INDEX'])
            local = False
        except:
            param_index = -1
            local = True
            print('Running locally')
        counter = 0

        if noisy_initial:
            cov_df = pd.read_json(cov_file)
        else:
            cov = None

        for ps in batch_parameters:
            if counter % self.jobs + 1 == param_index or local:
                # initialise to batch parameters
                if noisy_initial:
                    cov = cov_df.loc[
                        np.logical_and(np.isclose(cov_df['k1'], ps['k1']), np.isclose(cov_df['k2'], ps['k2']))
                        ]['Initial covariance matrix'].iloc[0]
                self._initialise_ensemble(ps, noisy_initial, cov)
                # run (small-ish) ensemble
                self.run_ensemble(repeats, run_type='lookup', noisy_initial=noisy_initial, cov=cov)
                # save results of ensemble (handled in overwritten save_data method)
            if ps['idx'] % report_after_n == 0:
                print(f"Completed {10*(ps['idx']//report_after_n)}% of runs at datetime {str(datetime.now())}")
            counter += 1
        self.ensemble_data.reset_index(drop=True, inplace=True)
        print("Completed all runs at {}.".format(str(datetime.now())))

    def run_erlang_batch(self, batch_parameters, repeats, n_parameter_combos, noisy_initial=False):
        '''
        Run a collection of ensembles of simulations with varying parameters.
        Splits each stage of the cell cycle into n_step separate Poisson processes
        This makes total progress through each phase more clock-like and shifts probability density away from 0 for each phase time.
        '''
        self.ensemble_data = pd.DataFrame([])
        report_after_n = n_parameter_combos//10 if n_parameter_combos >= 10 else 1
        print("Starting at datetime {}...".format(str(datetime.now())))

        try:
            param_index = int(os.environ['PBS_ARRAY_INDEX'])
            local = False
        except:
            param_index = -1
            local = True
            print('Running locally')
        counter = 0

        for ps in batch_parameters:
            if counter % self.jobs + 1 == param_index or local:
                # initialise to batch parameters
                self._initialise_ensemble(ps, noisy_initial)
                # run (small-ish) ensemble
                self.run_ensemble(repeats, run_type='lookup')
                # save results of ensemble (handled in overwritten save_data method)
            if ps['idx'] % report_after_n == 0:
                print(f"Completed {10*(ps['idx']//report_after_n)}% of runs at datetime {str(datetime.now())}")
            counter += 1
        self.ensemble_data.reset_index(drop=True, inplace=True)
        print("Completed all runs at {}.".format(str(datetime.now())))


    def infer_k(self, new_row):
        '''
        Find most likely params to have produced given cell counts.
        Distance calculated by total square relative distance from labeled cell counts
        '''
        gates = ['Mean EdU+BrdU-', 'Mean EdU-BrdU+', 'Mean EdU+BrdU+', 'Mean EdU-BrdU-']
        lookup_slice = self.lookup_df[self.lookup_df['BrdU time']==new_row['BrdU time']].copy() # Waiting time is known (so is pause before sampling, but is never varied)
        lookup_slice['Scaled Euclidean distance from obs'] = lookup_slice.apply(
            lambda lookup_row: np.sqrt(sum(
                [
                    ((lookup_row[gate] - new_row[gate])/lookup_row[gate.replace('Mean', 'Std')])**2 for gate in gates
                ]
            )),
            axis=1
        )
        # Mahalanobis distance instead of Euclidean
        # Need to redefine the look-up to include covariance instead of just std. Do it analytically.
        new_row_means = np.array([new_row[gate] for gate in gates])
        # lookup_slice['Mahalanobis distance from obs'] = lookup_slice.apply(lambda lookup_row:
                    # (new_row_means - lookup_row[gates]).T @ np.linalg.pinv(lookup_row['Covariance matrix']) @ (new_row_means - lookup_row[gates])
        # , axis=1)
        nearest_row = lookup_slice.iloc[lookup_slice['Scaled Euclidean distance from obs'].argmin()]
        # nearest_row = lookup_slice.iloc[lookup_slice['Mahalanobis distance from obs'].argmin()]
        k1, k2, tp = nearest_row['k1'], nearest_row['k2'], nearest_row['Cycle period']
        return k1, k2, tp

    def run_bootstrap(self, bootstrap_parameters, repeats, n_parameter_combos, noisy_initial=False, cov_file=None):
        '''
        Perform the bootstrapping process to extract Signal-to-Noise Ratio from 
        the noisy dual pulse nucleoside labeling simulation inference process
        '''
        self.ensemble_data = pd.DataFrame([])
        n_steps_progress = 10  # report 10%,20%,...,100%
        progress_thresholds = {round(len(bootstrap_parameters) * p / n_steps_progress): p*10 for p in range(1, n_steps_progress+1)}
        next_progress = min(progress_thresholds.keys()) if progress_thresholds else None
        print("Starting at datetime {}...".format(str(datetime.now())))

        try:
            param_index = int(os.environ['PBS_ARRAY_INDEX'])
            local = False
        except:
            param_index = -1
            local = True
            print('Running locally')
        counter = 0

        if noisy_initial:
            cov_df = pd.read_json(cov_file)
        else:
            cov = None

        for ps in bootstrap_parameters:
            if counter % self.jobs + 1 == param_index or local:
                # initialise to batch parameters
                if noisy_initial:
                    cov = cov_df.loc[
                        np.logical_and(np.isclose(cov_df['k1'], ps['k1']), np.isclose(cov_df['k2'], ps['k2']))
                        ]['Initial covariance matrix'].iloc[0]
                self.repeat_ensembles = ps['repeat_ensembles']
                for _ in range(0, ps['repeat_ensembles']):
                    self._initialise_ensemble(ps, noisy_initial, cov)
                    # run (small-ish) ensemble
                    self.run_ensemble(repeats, run_type='bootstrap', noisy_initial=noisy_initial, cov=cov) # Should save n_experiments rows that use the same parameters, so can just aggregate.
                if next_progress is not None and counter >= next_progress:
                    print(f'{progress_thresholds[next_progress]}% of parameter combinations complete.')
                    del progress_thresholds[next_progress]
                    next_progress = min(progress_thresholds.keys()) if progress_thresholds else None
            counter += 1
        self.summary_data = self.summarise_k().reset_index(drop=True)
        print("Completed all runs at {}.".format(str(datetime.now())))

    def summarise_k(self):
        ''''
        Get mean and std for each parameter combination to be saved in a Pandas DataFrame.
        '''
        self.ensemble_data['Std inferred k1'] = 0 # Need to add these columns now
        self.ensemble_data['Std inferred k2'] = 0
        compacted_df = self.ensemble_data.head(0)
        cols = self.ensemble_data.columns
        agg = {c: ['mean', 'std'] if 'Mean' in c else 'mean' for c in cols} # Choose between standard error and standard deviation later
        corr_cols = [c for c in cols if ('Mean' in c or 'Std' in c)]
        # Removed initial G1 from the groupby, as this is now variable! Total cells at start is not being varied generally.
        for gp_ps, gp_df in self.ensemble_data.groupby(["k1", "k2", "Cycle period", "EdU time", "BrdU time", "Wait time", "Repeats"]): #k1, k2, G1 uniquely determine n
            # new_row = gp_df.agg(agg)
            mean_row = pd.DataFrame(gp_df.agg(agg).iloc[0]).T
            std_row = pd.DataFrame(gp_df.agg(agg).iloc[1]).T
            mean_row[[c.replace('Mean', 'Std') for c in corr_cols if 'Mean' in c]] = np.array(np.sqrt(self.repeats)*std_row[[c for c in corr_cols if 'Mean' in c]]) # Adjust ste to std
            matrix = np.corrcoef(gp_df[corr_cols[:-2]].copy().T).astype(object)
            mean_row['Ensemble Correlations'] = [matrix]
            compacted_df = pd.concat([compacted_df, mean_row.reset_index(drop=True)], axis='index')
            # assert (mean_row.T.isna().sum() == 0).all(), 'NaN values in outputs!'       
        try:
            compacted_df['SNR k1'] = compacted_df['Mean inferred k1']/compacted_df['Std inferred k1']
        except:
            compacted_df['SNR k1'] = compacted_df['Mean inferred k1']/(compacted_df['Std inferred k1']+1E-8)
        try:
            compacted_df['SNR k2'] = compacted_df['Mean inferred k2']/compacted_df['Std inferred k2']
        except:   
            compacted_df['SNR k2'] = compacted_df['Mean inferred k2']/(compacted_df['Std inferred k2']+1E-8)
        # Get batchwise correlation coefficients in post-processing, keep ensemble-wise correlation in case

        return compacted_df.reset_index(drop=True)

'''
Utilities
'''

# Find the normalized steady state of a 3-stage Poisson process cell cycle with known k values
def generate_steady_state(K2, K3, n, padding=0):
    '''
    K2 = k2/k1, K3 = k3/k1
    '''
    T = np.array([[-1, 0, 2*K3],[1, -K2, 0],[0, K2, -K3]])
    f = lambda x, T: np.dot(T, x) + T[-1,-1] * x[-1] * x
    sol, _, ier, mesg = fsolve(f, x0=np.array([0.33,0.34,0.33]), args=(T,), full_output=True)
    if ier == 1 and sol.all() != 0.0:
        out = np.zeros((len(sol) + padding))
        out[0:len(sol)] = [round(s*n) for s in sol]
        return out
    else:
        print("No solution found.")
        return ValueError(mesg)


def generate_steady_state_erlang(K2, K3, n, n_steps=10, padding=0):
    '''
    K2 = k2/k1, K3 = k3/k1
    '''
    g1_row = np.array([1, -1, *([0]*(3*n_steps-2))])
    g1_rows = [np.roll(g1_row, i) for i in range(n_steps-1)]
    s_row = [*([0]*(n_steps)), K2, -K2, *([0]*(2*n_steps - 2))]
    s_rows = [np.roll(s_row, i) for i in range(n_steps - 1)]
    g2m_row = [*([0]*(2*n_steps)), K3, -K3, *([0]*(n_steps - 2))]
    g2m_rows = [np.roll(g2m_row, i) for i in range(n_steps - 1)]
    T = [
        np.array([-1, *([0]*(3*n_steps-2)), 2*K3]),
        *g1_rows,
        np.array([*([0]*(n_steps-1)), 1, -K2, *([0]*(2*n_steps-1))]),
        *s_rows,
        np.array([*([0]*(2*n_steps-1)), K2, -K3, *([0]*(n_steps-1))]),
        *g2m_rows
    ]
    T = np.array(T)*n_steps
    f = lambda x, T: np.dot(T, x) + T[-1,-1] * x[-1] * x
    sol, _, ier, mesg = fsolve(f, x0=np.array([0.33,0.34,0.33]*n_steps)/n_steps, args=(T,), full_output=True)
    if ier == 1 and sol.all() != 0.0:
        out = np.zeros((len(sol) + padding))
        out[0:len(sol)] = [round(s*n) for s in sol]
        return out
    else:
        print("No solution found.")
        return ValueError(mesg)
    

from scipy.linalg import null_space

def generate_steady_state_erlang_fast(
    K2, K3, n, n_steps=10, padding=0
):
    """
    Steady-state mean occupancy for a 3-phase Erlang cell cycle.

    K2 = k2 / k1
    K3 = k3 / k1
    n  = total population size
    """

    N = 3 * n_steps

    # --- build generator T ---
    T = np.zeros((N, N))

    # G1 chain
    for i in range(n_steps - 1):
        T[i, i]       -= 1
        T[i + 1, i]   += 1

    # G1 -> S
    T[n_steps, n_steps - 1] += 1
    T[n_steps - 1, n_steps - 1] -= 1

    # S chain
    for i in range(n_steps, 2*n_steps - 1):
        T[i, i]     -= K2
        T[i + 1, i] += K2

    # S -> G2M
    T[2*n_steps, 2*n_steps - 1] += K2
    T[2*n_steps - 1, 2*n_steps - 1] -= K2

    # G2M chain
    for i in range(2*n_steps, 3*n_steps - 1):
        T[i, i]     -= K3
        T[i + 1, i] += K3

    # division: G2M_last → 2 G1_0
    T[0, -1] += 2 * K3
    T[-1, -1] -= K3

    # Erlang scaling
    T *= n_steps

    # --- steady state via nullspace ---
    ns = null_space(T)
    if ns.size == 0:
        raise RuntimeError("No steady state found")

    x = ns[:, 0]
    x = np.abs(x)
    x /= x.sum()
    x *= n

    # --- output ---
    out = np.zeros(N + padding)
    out[:N] = np.round(x)

    return out



def generate_params_for_k_sweep(G1_list, S_list, edu_times, brdu_times, wait_times, periods, ns, repeat_ensembles=None):
    d=[]
    i=0
    min_G1_G2M = min(G1_list) # G1 and G2M ranges fully symmetric under this arrangement
    if repeat_ensembles is None:
            combos = itertools.product(edu_times, brdu_times, wait_times, periods, ns, S_list)
            for combo in combos:
                G1_subset = G1_list[G1_list < combo[3] - combo[-1] - min_G1_G2M] # All elements in G1_list such that their value is less than period - S time - minimum G1/G2M time.
                for G1 in G1_subset:
                    d.append({
                        "idx": i,
                        "k1": 1/G1,
                        "k2": 1/combo[-1],
                        "k3": 1/(combo[3] - combo[-1] - G1),
                        "edu_t": combo[0],
                        "brdu_t": combo[1],
                        "wait_t": combo[2],
                        "cycle_period": combo[3],
                        "n": combo[4]
                    })
                    i += 1
    else:
        combos = itertools.product(edu_times, brdu_times, wait_times, periods, ns, S_list, repeat_ensembles)
        for combo in combos:
            G1_subset = G1_list[G1_list < combo[3] - combo[-2] - min_G1_G2M]
            for G1 in G1_subset:
                d.append({
                    "idx": i,
                    "k1": 1/G1,
                    "k2": 1/combo[-2],
                    "k3": 1/(combo[3] - combo[-2] - G1),
                    "edu_t": combo[0],
                    "brdu_t": combo[1],
                    "wait_t": combo[2],
                    "cycle_period": combo[3],
                    "n": combo[4],
                    "repeat_ensembles": combo[-1]
                })
                i += 1
    return d

# Make a mesh of parameters to trial on a hexagonal mesh of a 2-simplex, as a fairer spacing
def hex_param_mesh(avg_times, dt1, dt2, ndt):
    def_dt1 = np.array([2,-1,-1])
    def_l = np.sqrt(def_dt1.dot(def_dt1))
    def_dt2 = np.array([-1, -1, 2])
    coefficients = [(i, j) for i in range(-ndt, ndt+1) for j in range(-ndt, ndt+1)]
    points = []
    for i, j in coefficients:
        if np.sqrt((i*def_dt1+j*def_dt2).dot(i*def_dt1+j*def_dt2))/def_l <= ndt:
            points.append(avg_times + i*dt1 + j*dt2)
    return 1/np.array(points)

# If we know which values we want to simulate already
def param_sweep_k_predefined(ks, edu_times, brdu_times, wait_times, periods, ns, repeat_ensembles=None):
    d = []
    i = 0
    if repeat_ensembles != None:
        combos = itertools.product(edu_times, brdu_times, wait_times, periods, ns, repeat_ensembles)
        for combo in combos:
            for k in ks:
                d.append({
                    "idx": i,
                    "k1": k[0],
                    "k2": k[1],
                    "k3": k[2],
                    "edu_t": combo[0],
                    "brdu_t": combo[1],
                    "wait_t": combo[2],
                    "cycle_period": combo[3],
                    "n": combo[4],
                    "repeat_ensembles": combo[5],
                })
                i += 1
    else:
        combos = itertools.product(edu_times, brdu_times, wait_times, periods, ns)
        for combo in combos:
            for k in ks:
                d.append({
                    "idx": i,
                    "k1": k[0],
                    "k2": k[1],
                    "k3": k[2],
                    "edu_t": combo[0],
                    "brdu_t": combo[1],
                    "wait_t": combo[2],
                    "cycle_period": combo[3],
                    "n": combo[4],
                })
                i += 1
    return d

def make_lookup(jobs=1, noisy_initial=False, cov_file=None): # Make a look-up table of cell cycle phase rates and inter-pulse waiting times
    edu_time = [0]
    brdu_time = list(range(1, 13, 1))
    wait_time = [0.5]

    period = [24]
    n = [300]
    # G1_list = np.linspace(7,17,20)
    # S_list = np.linspace(6,12,40)
    k_points = hex_param_mesh(np.array([11,8,5]), 1/np.sqrt(6)*np.array([2,-1,-1]), 1/np.sqrt(6)*np.array([-1,-1,2])*0.5, 6)

    DoubleStainBatch = DualLabelingBatch(
        t_0=0,
        x_0=np.array([0]*12),
        extras=[jobs] # Jobs
    )
    if noisy_initial:
        DoubleStainBatch.alpha = 0.2
    params = param_sweep_k_predefined(k_points, edu_time, brdu_time, wait_time, period, n)
    DoubleStainBatch.run_batch(
        batch_parameters = params,
        repeats = 1000,
        n_parameter_combos = len(params),
        noisy_initial = noisy_initial,
        cov_file = cov_file
    )
    try: # If a pbs array index is found, this means the code is being run in parallel on the cluster and there will be multiple output files
        DoubleStainBatch.ensemble_data.to_csv(f"data/DL lookup {datetime.today().date()} {int(os.environ['PBS_ARRAY_INDEX'])} {f'noisy {DoubleStainBatch.alpha}'.replace('.', 'p') if noisy_initial else ''}.csv")
    except:
        DoubleStainBatch.ensemble_data.to_csv(f"data/DL lookup {datetime.today().date()} {f'noisy {DoubleStainBatch.alpha}'.replace('.', 'p') if noisy_initial else ''}.csv")


def erlang_stoichiometries_4labels(n_steps, k1, k2, k3,
                                  labels=("unstained", "blue", "purple", "red")):
    """
    Return stoichiometry list for a 3-phase Erlang cell cycle with n_steps
    per phase, replicated for four independent labels.
    """

    n_labels = len(labels)
    n_phases = 3
    k1, k2, k3 = np.array([k1, k2, k3])*n_steps
    N = n_labels * n_phases * n_steps

    def idx(label, phase, step):
        """Linear index into state vector."""
        return label * (n_phases * n_steps) + phase * n_steps + step

    stoichs = []

    for l, lname in enumerate(labels):
        # ---- G1 chain ----
        for i in range(n_steps - 1):
            pre = np.zeros(N).astype(int)
            post = np.zeros(N).astype(int)
            pre[idx(l, 0, i)] = 1
            post[idx(l, 0, i + 1)] = 1
            stoichs.append({
                'name': rf'$G_1^{{{i+1}}}\rightarrow G_1^{{{i+2}}}$ ({lname})',
                'stoich': (pre, post),
                'rate': k1
            })
        # G1 -> S
        pre = np.zeros(N).astype(int)
        post = np.zeros(N).astype(int)
        pre[idx(l, 0, n_steps - 1)] = 1
        post[idx(l, 1, 0)] = 1
        stoichs.append({
            'name': rf'$G_1\rightarrow S$ ({lname})',
            'stoich': (pre, post),
            'rate': k1
        })
        # ---- S chain ----
        for i in range(n_steps - 1):
            pre = np.zeros(N).astype(int)
            post = np.zeros(N).astype(int)
            pre[idx(l, 1, i)] = 1
            post[idx(l, 1, i + 1)] = 1
            stoichs.append({
                'name': rf'$S^{{{i+1}}}\rightarrow S^{{{i+2}}}$ ({lname})',
                'stoich': (pre, post),
                'rate': k2
            })
        # S -> G2M
        pre = np.zeros(N).astype(int)
        post = np.zeros(N).astype(int)
        pre[idx(l, 1, n_steps - 1)] = 1
        post[idx(l, 2, 0)] = 1
        stoichs.append({
            'name': rf'$S\rightarrow G_2M$ ({lname})',
            'stoich': (pre, post),
            'rate': k2
        })
        # ---- G2M chain ----
        for i in range(n_steps - 1):
            pre = np.zeros(N).astype(int)
            post = np.zeros(N).astype(int)
            pre[idx(l, 2, i)] = 1
            post[idx(l, 2, i + 1)] = 1
            stoichs.append({
                'name': rf'$G_2M^{{{i+1}}}\rightarrow G_2M^{{{i+2}}}$ ({lname})',
                'stoich': (pre, post),
                'rate': k3
            })
        # Division: G2M_n -> 2 G1_1
        pre = np.zeros(N).astype(int)
        post = np.zeros(N).astype(int)
        pre[idx(l, 2, n_steps - 1)] = 1
        post[idx(l, 0, 0)] = 2
        stoichs.append({
            'name': rf'$G_2M\rightarrow 2G_1$ ({lname})',
            'stoich': (pre, post),
            'rate': k3
        })

    return stoichs

class DualLabelingBatchErlang(DualLabelingBatch):

    def _setup_extras(self, extras):
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.jobs = extras[0]
        self.n_steps = extras[1]

    def _stoichiometry(self):
        return erlang_stoichiometries_4labels(self.n_steps, self.k1, self.k2, self.k3)
    
    def _tag_edu(self):
        '''
        Stain all cells in S phase to become EdU+.
        '''
        self.x[4*self.n_steps:5*self.n_steps] += self.x[self.n_steps:2*self.n_steps]           
        self.x[self.n_steps:2*self.n_steps] = 0


    def _tag_brdu(self, i=1):
        '''
        Stain all cells in S phase to become BrdU+, accounting for cells already stained with EdU.
        '''
        self.pre_brdu_states[i] = self.x.copy()
        self.x[7*self.n_steps:8*self.n_steps] += self.x[4*self.n_steps:5*self.n_steps]
        self.x[10*self.n_steps:11*self.n_steps] += self.x[self.n_steps:2*self.n_steps]
        self.x[self.n_steps:2*self.n_steps] = 0
        self.x[4*self.n_steps:5*self.n_steps] = 0


    def save_data(self, run_type='lookup'):
        new_row = {
            "k1": self.k1,
            "k2": self.k2,
            "k3": self.k3,
            "Cycle period": self.cycle_period,
            "Initial G1": sum(self.x_series[0][0:self.n_steps]),
            "Initial S": sum(self.x_series[0][self.n_steps:2*self.n_steps]),
            "Initial G2M": sum(self.x_series[0][2*self.n_steps:3*self.n_steps]),
            "EdU time": self.edu_time,
            "BrdU time": self.brdu_time,
            "Wait time": self.measurement_time - self.brdu_time,
            "Repeats": self.repeats
        }
        gates = ['EdU-BrdU-', 'EdU+BrdU-', 'EdU+BrdU+', 'EdU-BrdU+']
        for i, gate in zip(range(0, 10*self.n_steps, 3*self.n_steps), gates):
            new_row.update({f"Mean {gate}": np.mean(np.sum(self.end_states[:, i:i+3*self.n_steps], axis=1))})
            new_row.update({f"Std {gate}": np.std(np.sum(self.end_states[:, i:i+3*self.n_steps], axis=1))})
        # for i in range(12):
        #     new_row.update({f"Mean cells in {i} at end": np.mean(self.end_states[:,i])})
        #     new_row.update({f"Std cells in {i} at end": np.std(self.end_states[:,i])})
        #     if i<6:
        #         new_row.update({f"Mean cells in {i} at BrdU time": np.mean(self.pre_brdu_states[:,i])})
        #         new_row.update({f"Std cells in {i} at BrdU time": np.std(self.pre_brdu_states[:,i])})
        # correlation_columns = [c for c in new_row.keys() if ('Mean' in c or 'Std' in c)]
        if run_type=='bootstrap':
            new_row.update({"Repeat Ensembles": self.repeat_ensembles})
            k1, k2, tp = self.infer_k(new_row)
            new_row.update({'Mean inferred k1': k1, 'Mean inferred k2': k2, 'Mean inferred cycle period': tp})
        self.ensemble_data = pd.concat([self.ensemble_data, pd.DataFrame([new_row])], ignore_index=True)

    def _initialise_ensemble(self, ps, noisy_initial=False, cov=None):
        self.k1 = ps['k1']
        self.k2 = ps['k2']
        self.k3 = ps['k3']
        self.edu_time = 0 # Will set to np.inf after staining to avoid re-staining and to minimise variables
        self.brdu_time = ps['brdu_t'] # likewise
        self.measurement_time = ps['wait_t']+ps['brdu_t'] # end of simulation here
        self.cycle_period = (1/self.k1 + 1/self.k2 + 1/self.k3)
        self._edit_stoichiometry() # set rates for this run
        self.x_0 = generate_steady_state_erlang(ps['k2']/ps['k1'], ps['k3']/ps['k1'], ps['n'], n_steps=self.n_steps, padding=9*self.n_steps)
        self.x = self.x_0.copy()
        assert (self.x_0 >= 0).all(), 'Negative terms in steady state!'
        if noisy_initial:
            # self.run_until_n_for_initial(ps['n']) # Sets self.x inside
            # self.x[0:3] += np.multiply(np.random.randn(3), np.sqrt(self.x[0:3])).round().astype(int)
            self.x[0:3] += np.random.multivariate_normal([0.0, 0.0, 0.0], cov)*self.alpha
            assert (self.x >= 0).all(), 'Negative terms in initial state! Reduce noise contribution.'
        self.x = self.x.round().astype(int)
        self.x_0 = self.x_0.round().astype(int)
        self.x_series = [self.x.copy()]

    def _edit_stoichiometry(self):
        k1, k2, k3 = np.array([self.k1, self.k2, self.k3])*self.n_steps
        for i, s in enumerate(self.stoich):
            if "$G_1" in s['name']:
                self.stoich[i]['rate'] = k1
            if "$S" in s['name']:
                self.stoich[i]['rate'] = k2
            if "$G_2M" in s['name']:
                self.stoich[i]['rate'] = k3


def make_lookup_erlang(n_steps=10, noisy_initial=False, jobs=1): # Make a look-up table of cell cycle phase rates and inter-pulse waiting times
    edu_time = [0]
    brdu_time = list(range(1, 13, 1))
    wait_time = [0.5]

    period = [24]
    n = [300]
    # G1_list = np.linspace(7,17,20)
    # S_list = np.linspace(6,12,40)
    k_points = hex_param_mesh(np.array([11,8,5]), 1/np.sqrt(6)*np.array([2,-1,-1]), 1/np.sqrt(6)*np.array([-1,-1,2])*0.5, 6)

    DoubleStainBatch = DualLabelingBatchErlang(
        t_0=0,
        x_0=np.array([0]*12*n_steps),
        extras=[jobs, n_steps]
    )
    if noisy_initial:
        DoubleStainBatch.alpha = 0.5
    params = param_sweep_k_predefined(k_points, edu_time, brdu_time, wait_time, period, n)
    DoubleStainBatch.run_batch(
        batch_parameters = params,
        repeats = 80,
        n_parameter_combos = len(params),
        noisy_initial = noisy_initial
    )
    try: # If a pbs array index is found, this means the code is being run in parallel on the cluster and there will be multiple output files
        DoubleStainBatch.ensemble_data.to_csv(f"data/DL lookup {datetime.today().date()} {int(os.environ['PBS_ARRAY_INDEX'])} erlang {n_steps}.csv")
    except:
        DoubleStainBatch.ensemble_data.to_csv(f"data/DL lookup {datetime.today().date()} erlang {n_steps}.csv")


def run_bootstrap(lookup_path, jobs=1, noisy_initial=False, cov_file=None, ks_as_in_lookup=False):
    edu_time = [0]
    wait_time = [0.5]
    period = [24]
    n = [300]

    DoubleStainBatch = DualLabelingBatch(
        t_0=0,
        x_0=np.array([0]*12),
        extras=[jobs] # Jobs
    )
    repeat_ensembles = [1000]

    if lookup_path[-4:] == '.csv':
        DoubleStainBatch.lookup_df = pd.read_csv(lookup_path)
    elif lookup_path[-5:] == '.json':
        DoubleStainBatch.lookup_df = pd.read_json(lookup_path)
    else:
        raise ValueError('Format of lookup table not understood. Please use CSV or JSON.')

    if ks_as_in_lookup:
        k_points = DoubleStainBatch.lookup_df[['k1', 'k2', 'k3']].copy()
        k_points = k_points[np.logical_and(np.isclose(k_points['k1'], k_points['k1'].mean(), rtol=5E-2), np.isclose(k_points['k2'], k_points['k2'].mean(), rtol=5E-2))].to_numpy()
    else:
        k_points = hex_param_mesh(np.array([11,8,5]), 1/np.sqrt(6)*np.array([2,-1,-1]), 1/np.sqrt(6)*np.array([-1,-1,2])*0.5, 6)
        # k_points = hex_param_mesh(np.array([11,8,5]), 1/np.sqrt(6)*np.array([2,-1,-1]), 1/np.sqrt(6)*np.array([-1,-1,2])*0.5, 2)
        # G1_list = np.linspace(7,17,20)
        # S_list = np.linspace(6,12,40)
    if 'Cycle period' not in DoubleStainBatch.lookup_df.columns:
        DoubleStainBatch.lookup_df['Cycle period'] = 1/DoubleStainBatch.lookup_df['k1'] + 1/DoubleStainBatch.lookup_df['k2'] + 1/DoubleStainBatch.lookup_df['k3']

    brdu_time = DoubleStainBatch.lookup_df['BrdU time'].unique()
    params = param_sweep_k_predefined(k_points, edu_time, brdu_time, wait_time, period, n, repeat_ensembles)
    if noisy_initial:
        DoubleStainBatch.alpha = 0.2
    DoubleStainBatch.run_bootstrap(
        bootstrap_parameters = params,
        repeats = 3,
        n_parameter_combos = len(params),
        noisy_initial = noisy_initial,
        cov_file = cov_file
    )
    try:
        DoubleStainBatch.summary_data.to_json(f"data/DL bootstrap {datetime.today().date()} {int(os.environ['PBS_ARRAY_INDEX'])} {f'noisy {DoubleStainBatch.alpha}'.replace('.', 'p') if noisy_initial else ''} 10rep.json")
    except:
        DoubleStainBatch.summary_data.to_json(f"data/DL bootstrap {datetime.today().date()} {f'noisy {DoubleStainBatch.alpha}'.replace('.', 'p') if noisy_initial else ''} 10rep.json")
    # Saving these in json to preserve np array and not lose all readability & keep library version generality

def run_bootstrap_erlang(lookup_path, jobs=1, n_steps=10, noisy_initial=False):
    edu_time = [0]
    brdu_time = list(range(1, 13, 1))
    wait_time = [0.5]

    period = [24]
    n = [300]
    # G1_list = np.linspace(7,17,20)
    # S_list = np.linspace(6,12,40)
    # k_points = hex_param_mesh(np.array([11,8,5]), 1/np.sqrt(6)*np.array([2,-1,-1]), 1/np.sqrt(6)*np.array([-1,-1,2])*0.5, 6)

    DoubleStainBatch = DualLabelingBatchErlang(
        t_0=0,
        x_0=np.array([0]*12*n_steps),
        extras=[jobs, n_steps]
    )
    repeat_ensembles = [100]
    # params = param_sweep_k_predefined(k_points, edu_time, brdu_time, wait_time, period, n, repeat_ensembles)
    if lookup_path[-4:] == '.csv':
        DoubleStainBatch.lookup_df = pd.read_csv(lookup_path)
    elif lookup_path[-5:] == '.json':
        DoubleStainBatch.lookup_df = pd.read_json(lookup_path)
    else:
        raise ValueError('Format of lookup table not understood. Please use CSV or JSON.')
    if 'Wait time' not in DoubleStainBatch.lookup_df.columns:
        DoubleStainBatch.lookup_df['Wait time'] = 0.5
    if 'Cycle period' not in DoubleStainBatch.lookup_df.columns:
        DoubleStainBatch.lookup_df['Cycle period'] = 24.0
    params = [df_row.to_dict() for _, df_row in DoubleStainBatch.lookup_df[['k1', 'k2', 'k3', 'BrdU time', 'Wait time']].iterrows()]
    for idx in range(len(params)):
        params[idx].update({
            "edu_t": 0.0,
            "brdu_t": params[idx]['BrdU time'],
            "wait_t": params[idx]['Wait time'],
            "cycle_period": 24.0,
            "n": 300,
            "repeat_ensembles": repeat_ensembles[0]
        })
    params = [p for p in params if np.logical_and(np.isclose(p['k1'], DoubleStainBatch.lookup_df['k1'].mean(), rtol=5E-2), np.isclose(p['k2'], DoubleStainBatch.lookup_df['k2'].mean(), rtol=5E-2))]
    print(len(params))
    params = np.random.choice(params, len(params), replace=False)
    if noisy_initial:
        DoubleStainBatch.alpha = 0.5
    DoubleStainBatch.run_bootstrap(
        bootstrap_parameters = params,
        repeats = 3,
        n_parameter_combos = len(params),
        noisy_initial = noisy_initial
    )
    try:
        DoubleStainBatch.summary_data.to_json(f"data/DL bootstrap {datetime.today().date()} {int(os.environ['PBS_ARRAY_INDEX'])} erlang {n_steps} redo.json")
    except:
        DoubleStainBatch.summary_data.to_json(f"data/DL bootstrap {datetime.today().date()} erlang {n_steps} redo.json")
    # Saving these in json to preserve np array and not lose all readability & keep library version generality


def run_demo_experiment(lookup_path='DPNL lookup.json'):
    edu_time = [0]
    wait_time = [0.5]
    period = [24]
    n = [300]

    DoubleStainBatch = DualLabelingBatch(
        t_0=0,
        x_0=np.array([0]*12),
        extras=[1] 
    )
    repeat_ensembles = [10]

    if lookup_path[-4:] == '.csv':
        DoubleStainBatch.lookup_df = pd.read_csv(lookup_path)
    elif lookup_path[-5:] == '.json':
        DoubleStainBatch.lookup_df = pd.read_json(lookup_path)
    else:
        raise ValueError('Format of lookup table not understood. Please use CSV or JSON.')

    k_points = np.array([np.array([1/12, 1/8, 1/4])]*15) # 5 repeats of the same parameters, middle-of-the-range
    k_points += 1E-8*np.random.rand(15,3)

    if 'Cycle period' not in DoubleStainBatch.lookup_df.columns:
        DoubleStainBatch.lookup_df['Cycle period'] = 1/DoubleStainBatch.lookup_df['k1'] + 1/DoubleStainBatch.lookup_df['k2'] + 1/DoubleStainBatch.lookup_df['k3']

    brdu_time = [1, 3, 8]
    params = param_sweep_k_predefined(k_points, edu_time, brdu_time, wait_time, period, n, repeat_ensembles)

    DoubleStainBatch.run_bootstrap(
        bootstrap_parameters = params,
        repeats = 1,
        n_parameter_combos = len(params),
    )

    DoubleStainBatch.summary_data.to_json(f"data/DL demo {datetime.today().date()}.json")
    # Saving these in json to preserve np array and not lose all readability & keep library version generality



if __name__ == '__main__':
    # print('Current working directory:', os.getcwd())
    # make_lookup(noisy_initial=True, cov_file='data/DL analytic cov noisy.json') # Makes a new look-up table from noisy initial conditions
    # run_bootstrap('data/DL lookup 2026-01-14 noisy 0p2.csv', noisy_initial=True, cov_file='data/DL analytic cov noisy.json', ks_as_in_lookup=True)
    # make_lookup_erlang(n_steps=4, noisy_initial=False)
    # df = pd.read_json('data/DL bootstrap 2025-12-20 erlang 4.json')
    # df = df.loc[np.logical_or(np.logical_or(df['SNR k1']>100.0, df['SNR k2']>100.0), np.logical_or(df['SNR k1'].isna(), df['SNR k2'].isna()))]
    # run_bootstrap_erlang('data/DL analytic cov erlang 4.json', n_steps=4, noisy_initial=False)
    # print('Done!')

    ### Inference without a fixed period, all within 1 hour of 24 hours

    # run_bootstrap('data/DL analytic cov templated.json', noisy_initial=False, cov_file='data/DL analytic cov templated.json', ks_as_in_lookup=True)
    # run_demo_experiment()

    ### Increasing the number of repeats to 10
    run_bootstrap('DPNL lookup.json', noisy_initial=False, ks_as_in_lookup=True)
