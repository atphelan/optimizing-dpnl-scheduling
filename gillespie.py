'''
Generic Gillespie simulation base class and ensemble of simulations base class.
Simulates a prespecified set of two- or one- body reactions under a few assumptions.
'''
import numpy as np
import matplotlib.pyplot as plt
import warnings
import copy

class Reaction():
    def __init__(self, t_0, x_0, plot=True, extras=None):
        self.x = x_0
        self.x = self.x.astype('int64')
        self.t = t_0
        self.intflag = False
        self.initials = [t_0, x_0, extras]
        self.plot = plot
        self.dim = len(x_0)
        self._setup_extras(extras)
        self.stoich = self._stoichiometry()
        self.M = len(self.stoich)

    def _species(self):
        return NotImplementedError()

    def _setup_extras(self, extras):
        '''
        Any other init bits specific to this model, optional
        '''
        pass

    def _stoichiometry(self):
        return NotImplementedError()

    def _name(self):
        return NotImplementedError()

    def _get_a(self):
        if self.x.dtype != np.dtype('int64') and not self.intflag:
            self.intflag = True
            warnings.warn("System state not stored as 64-bit integers. Truncating now.")
            self.x = self.x.astype('int64')
        a = np.zeros(self.M+1)
        for i, s in enumerate(self.stoich):
            if 0 in self.x[np.where(s['stoich'][0] > 0)[0]] or np.isnan(self.x[np.where(s['stoich'][0] > 0)[0]]).any():
                a[i] = 0
                continue
            if 1 in s['stoich'][0]: # Implies internal single-body process or a collision of different type individuals
                r = self.x[np.where(s['stoich'][0]==1)[0]]
            elif 2 in s['stoich'][0]: # Only other possibility: collision of same type of individual
                r = self.x[np.where(s['stoich'][0]==2)[0]]
                r *= r-1
            else:
                a[i] = 0
                continue
            assert (r >= 0).any(), "Negative rate!"
            a[i] = s['rate']*np.prod(r)
        a[-1] = np.sum(a)
        return a

    def _tau_j(self):
        a = self._get_a() # Get the current propensities for each reaction
        if a[-1] == 0:
            return 0, None
        r = np.random.random(2)    
        tau = -1/a[-1] * np.log(r[0])
        j = np.where(np.cumsum(a[:-1]) > r[1]*a[-1])[0][0]
        return tau, j

    def plot_trajectory(self):
        fig, ax = plt.subplots()
        ax.plot(self.t_series, self.x_series, label=self._species())
        plt.title(self._name())
        plt.xlabel('Time')
        plt.ylabel('Counts of species')
        ax.legend(bbox_to_anchor=(1,1), loc='upper left')
        plt.tight_layout()
        plt.show()
        

    def run_simulation(self, n):
        self.x_series = np.zeros((n+1, self.dim))
        self.t_series = np.zeros(n+1)
        self.x_series[0] = self.x
        self.t_series[0] = self.t
        for i in range(1, n+1):
            tau, j = self._tau_j()
            self.t += tau
            if not (j is None):
                s = self.stoich[j]['stoich']
                self.x -= s[0] - s[1]
            self.t_series[i] = self.t
            self.x_series[i] = self.x
        if self.plot == True:
            self.plot_trajectory()


class TestReaction(Reaction):

    def test_boundary_cases(self):
        ''' Try BCs at 0, 1, and np.inf maybe'''
        pass

    def _stoichiometry(self):
        # A+A -> B, B+C -> 2C, C -> 2A
        return [
            {'name': 'A+A->B', 'stoich': (np.array([2,0,0]), np.array([0,1,1])), 'rate': 1},
            {'name': 'B+C->2C', 'stoich': (np.array([0,1,1]), np.array([0,0,2])), 'rate': 1},
            {'name': 'C->2A', 'stoich': (np.array([0,0,1]), np.array([2,0,0])), 'rate': 1},
        ]
    
    def _species(self):
        # x = [A, B, C].T
        return ['A', 'B', 'C']

    def _name(self):
        return "Test System"

class TestDecay(Reaction):

    def test_boundary_cases(self):
        pass

    def _stoichiometry(self):
        return [{'name': 'A->0', 'stoich': (np.array([1]), np.array([0])), 'rate': 1}]

    
    def _species(self):
        # x = [A, B, C].T
        return ['A']

    def _name(self):
        return "Noisy decay problem"
    
    def infer_k(self, n):
        return -np.log(1/n)/self.t


class ReactionEnsemble(Reaction):
    '''
    A collection of reactions. Can use this to estimate the standard deviation or verify expectations
    '''

    def run_ensemble(self, N, n): 
        data = []
        times = []
        for _ in range(N):
            self.run_simulation(n)
            data.append(self.x_series.copy())
            times.append(self.t_series.copy())
            self._reset(self.t_series[0], self.x_series[0])
        self.plot_results(times, data, colours={'mean': None, 'envelope': None})


    def _reset(self, t_0, x_0):
        '''
        Restore initial conditions to start another Reaction in the ensemble
        '''
        self.x = x_0
        self.t = t_0


    def _setup_extras(self, extras):
        self.stain_counts = []
        self.initials = [copy.copy(self.t), copy.copy(self.x), extras]

    def run_simulation(self, n):
        self.x_series = np.zeros((self.dim, n+1)).T
        self.t_series = np.zeros(n+1)
        self.x_series[0] = self.initials[1]
        self.t_series[0] = self.initials[0]
        for i in range(1, n+1):
            tau, j = self._tau_j()
            self.t += tau
            s = self.stoich[j]['stoich']
            self.x -= s[0] - s[1]
            self.t_series[i] = self.t
            self.x_series[i] = self.x
        # print(self.x_series)

    def plot_results(self, times, data, colours, labels=None, envelope_mode = 'stderr'):
        '''
        Calculate mean trajectory and envelope, plot.

        data: numpy array (N, nt, n_species)
        '''
        # Warning to self: the time series are not the same. You should make iterpolation functions before mean
        t_f = np.max(times)
        plot_times = np.linspace(np.min(self.t_series), t_f, 500)
        interpolated_data = []
        for t_sim, sim in zip(times, data):
            sim = sim.T
            state_trajectory = []
            for species_trajectory in sim:
                state_trajectory.append(np.interp(x=plot_times, xp=t_sim, fp=species_trajectory))
            interpolated_data.append(np.array(state_trajectory).T)
        interpolated_data = np.array(interpolated_data)
        mean = np.mean(interpolated_data, axis=0)
        if envelope_mode == 'stderr':
            stderr = np.std(interpolated_data, axis=0)/np.sqrt(interpolated_data.shape[0])
            upper = mean + stderr*2
            lower = mean - stderr*2
        elif envelope_mode == 'range':
            upper = np.nanmax(interpolated_data, axis=0)
            lower = np.nanmin(interpolated_data, axis=0)
        else:
            ValueError("Please use 'stderr' or 'range' envelope_mode")

        fig, ax = plt.subplots()
        ax.plot(plot_times, mean, color=colours['mean'], label=labels)  
        for i in range(self.dim):
            lower_i = lower.T[i].T
            upper_i = upper.T[i].T
            if colours['envelope'] is None:
                ax.fill_between(plot_times, lower_i, upper_i, alpha=0.5)
            else:
                ax.fill_between(plot_times, lower_i, upper_i, color=colours['envelope'][i], alpha=0.5)

        plt.show()

class TestEnsemble(ReactionEnsemble):

    def test_boundary_cases(self):
        ''' Try BCs at 0, 1, and np.inf maybe'''
        pass

    def _stoichiometry(self):
        # A+A -> B, B+C -> 2C, C -> 2A
        return [
            {'name': 'A+A->B', 'stoich': (np.array([2,0,0]), np.array([0,1,1])), 'rate': 1},
            {'name': 'B+C->2C', 'stoich': (np.array([0,1,1]), np.array([0,0,2])), 'rate': 1},
            {'name': 'C->2A', 'stoich': (np.array([0,0,1]), np.array([2,0,0])), 'rate': 1},
        ]
    
    def _species(self):
        # x = [A, B, C].T
        return ['A', 'B', 'C']

    def _name(self):
        return "Test Ensemble of Systems"
    
class TestEnsembleDecay(ReactionEnsemble):
    def _stoichiometry(self):
        return [{'name': 'A->0', 'stoich': (np.array([1]), np.array([0])), 'rate': 1}]

    def _species(self):
        # x = [A, B, C].T
        return ['A']

    def _name(self):
        return "Noisy decay problem"
    
    def run_ensemble(self, N, n): 
        '''
        N: integer, size of ensemble
        n: integer, number of reactions allowed to happen per simulation in ensemble
        '''
        data = []
        times = []
        n0 = copy.copy(self.x[0])
        for _ in range(N):
            self.run_simulation(n)
            data.append(self.x_series.copy())
            times.append(self.t_series.copy())
            self._reset(self.t_series[0], self.x_series[0])
        if self.plot:
            self.plot_results(times, data, colours={'mean': None, 'envelope': None})
        k = self.infer_k(n, n0, np.array(times))
        print('Mean k: ', np.mean(k))
        print('Std k: ', np.std(k))
    
    def infer_k(self, n, n0, times):
        return -np.log((n0-n)/(n0))/times[:,-1]
    

class ReactionBatch(ReactionEnsemble):
    
    def run_batch(self):
        pass

    def plot_results(self):
        pass

if __name__ == '__main__':
    # # # Unit tests # # #
    # Test = TestReaction(0, np.array([0,0,0]), plot=False)
    # Test.run_simulation(10)
    # assert (Test.x_series == np.zeros((Test.dim, 11)).T).all() # Test 0: All 0

    # Test = TestReaction(0, np.array([1,0,0]), plot=False)
    # Test.run_simulation(5)
    # assert (Test.x_series == np.array([[[1,0,0]]*6])).all() # Test 1: 1 A, not enough for any reactions

    # Test = TestReaction(0, np.array([0,1,0]), plot=False)
    # Test.run_simulation(3)
    # assert (Test.x_series == np.array([[[0,1,0]]*4])).all() # Test 2: 1 B, no C for reaction

    # Test = TestReaction(0, np.array([0,0,1]), plot=False)
    # Test.run_simulation(1)
    # assert (Test.x_series == np.array([[0,0,1], [2,0,0]])).all() # Test 3: 1 C, do one 1 C -> 2A reaction

    # Test = TestReaction(0, np.array([2,0,0]), plot=False)
    # Test.run_simulation(1)
    # assert (Test.x_series == np.array([[2,0,0], [0,1,1]])).all() # Test 4: 2 A, do one A + A -> B + C reaction

    # Test = TestReaction(0, np.array([0,1,1]), plot=False)
    # Test.run_simulation(1)
    # assert (Test.x_series == np.array([[0,1,1], [0,0,2]])).all() # Test 5: 1 B and 1 C, do one B + C -> 2C reaction
    # # Seems to go wrong sometimes?

    # Test = TestReaction(0, np.array([2,2,2]), plot=True)
    # Test.run_simulation(100)
    # assert (Test.x_series >= 0).all() # Test 6: All reactions possible, run the simulation for a while. Check no deductions past 0.

    # TestEns = TestEnsemble(0, np.array([15,15,15]))
    # TestEns.run_ensemble(3, 1)

    # TestDec = TestDecay(0, np.array([100]), plot=False)
    # TestDec.run_simulation(100)
    # print('k = ', TestDec.infer_k(100))

    TestEnsDec = TestEnsembleDecay(0, np.array([100]), plot=False)
    TestEnsDec.run_ensemble(25, 50)