from evtools.sim_tools.utils import (
    split_registers, get_neighbourhood, get_active_qubits, get_expectation_values,
    FinishedJob, I, X, Y, Z
)
from qiskit import QuantumCircuit
from mthree.probability import quasi_to_probs
import numpy as np
import json
from warnings import filterwarnings
import statsmodels.api as sm
filterwarnings("ignore", category=DeprecationWarning)

class ProcessEVCDRMultiple:
    """
    """
    ancilla_neighbourhood_size = 3
    max_hamming_distance = 2
    force_zero_Z_shift = True
    normalize_clifford_purity = True
    n_resamples = 10_000
    use_old_regression = False

    def __init__(self, folder_index, backend, batches):
        """
        """
        
        self.batches = batches
        self.directory = f'data/EVCDR/{folder_index}'
        self.backend = backend

        self.list_num_clifford_samples   = {}
        self.list_num_non_clifford_gates = {}
        self.clifford_exact              = {}
        self.standard_exact              = {}
        self.job_ids                     = {}

        for b in self.batches:
            submission_filename = f'{self.directory}/{backend}_submission_{b}.json'
        
            with open(submission_filename,'r') as infile:
                submission_data = json.load(infile)
                self.steps = submission_data['steps']
                self.times = submission_data['times']
                self.step_size = submission_data['step_size']

                for key,val in enumerate(submission_data['num_clifford_samples']):
                    key += 1
                    self.list_num_clifford_samples.setdefault(key,{})
                    self.list_num_clifford_samples[key][b] = val
                for key,val in enumerate(submission_data['num_non_clifford_gates']):
                    key += 1
                    self.list_num_non_clifford_gates.setdefault(key,{})
                    self.list_num_non_clifford_gates[key][b] = val
                for key,val in submission_data['clifford_exact'].items():
                    self.clifford_exact.setdefault(int(key),{})
                    self.clifford_exact[int(key)][b] = val
                for key,val in submission_data['job_ids'].items():
                    self.job_ids.setdefault(int(key),{})
                    self.job_ids[int(key)][b] = val

    def load_job(self, step, batch):
        """
        """
        self.qpu_job = FinishedJob(f'{self.directory}/{self.backend}_{batch}/step_{step}.txt')
        self.measurement_circuits = dict(zip(['X','Y','Z'], self.qpu_job.inputs['circuits'][:3]))
        self.step = step
        self.num_clifford_samples   = self.list_num_clifford_samples[step][batch]
        self.num_non_clifford_gates = self.list_num_non_clifford_gates[step][batch]
        self._clifford_ideal_vals   = np.asarray(self.clifford_exact[step][batch])
        
        print(f'>> loaded qpu job at step={step}, batch={batch}')

    def retrieve_results(self):
        """
        """
        assert self.qpu_job.status().name == 'DONE', 'Job not finished'
        result = self.qpu_job.result()
        XYZ_bases = ['X','Y','Z']+['X']*self.num_clifford_samples+['Z']*self.num_clifford_samples
        if hasattr(result, 'quasi_dists'):
            counts_out = []
            assert len(XYZ_bases)==len(result.quasi_dists)
            for basis,counts in zip(XYZ_bases, result.quasi_dists):
                counts = quasi_to_probs(counts)[0]
                cregs = self.measurement_circuits[basis].cregs[::-1]
                n_measured_bits = sum([reg.size for reg in cregs])
                counts = {np.binary_repr(int(measint), n_measured_bits):val for measint,val in counts.items()}
                counts_out.append(split_registers(counts, cregs))
            return counts_out
        
    def postselect_measurements(self, counts, basis):
        """
        """
        measurement_map = {}
        for op in self.measurement_circuits[basis]:
            if op.operation.name == 'measure':
                measurement = op.clbits[0]
                reg = measurement.register.name
                if reg.find('sys') != -1:
                    measurement_map[op.qubits[0].index] = measurement.index

        ancilla_qubits = {}
        system_qubits = {}
        for op in self.measurement_circuits[basis]:
            if op.operation.name == 'measure':
                measurement = op.clbits[0]
                reg = measurement.register.name
                if reg.find('anc') != -1:
                    ancilla_index = reg.split('_')[-1]
                    ancilla_qubits[ancilla_index] = op.qubits[0].index
                if reg.find('sys') != -1:
                    system_index = reg.split('_')[-1]
                    system_qubits.setdefault(system_index, [])
                    system_qubits[system_index].append(op.qubits[0].index)

        mbits, freqs = zip(*counts.items())
        freqs = np.array(freqs)
        ancilla_measurements = [np.array([[int(bit) for bit in bitstr] for bitstr in m]) for m in zip(*[m.split(' ')[::2] for m in mbits])]
        compute_measurements = [np.array([[int(bit) for bit in bitstr] for bitstr in m]) for m in zip(*[m.split(' ')[1::2] for m in mbits])]

        postselection_masks = []
        for (cluster_index, ancilla), meas in zip(ancilla_qubits.items(), compute_measurements[::-1]):
            neighbourhood = get_neighbourhood(
                self.measurement_circuits[basis], 
                ancilla, self.ancilla_neighbourhood_size
            )
            neighbourhood_indices = [measurement_map[n] for n in neighbourhood]
            mask_zero_in_neighbourhood = ~np.any(meas[:,::-1][:,neighbourhood_indices], axis=1) # check if bits should be reversed
            mask_within_hamming_distance = np.sum(meas, axis=1)<=self.max_hamming_distance 
            postselection_masks.append(mask_zero_in_neighbourhood & mask_within_hamming_distance)
            
        ancilla_regs = self.measurement_circuits[basis].cregs[1::2]
        compute_regs = self.measurement_circuits[basis].cregs[::2]
        assert [anc.size for anc in ancilla_regs[::-1]] == [anc.shape[1] for anc in ancilla_measurements], 'Ancilla measurements and registers contain inconsistent numbers of qubits.'
        assert [com.size for com in compute_regs[::-1]] == [com.shape[1] for com in compute_measurements], 'Compute measurements and registers contain inconsistent numbers of qubits.'

        final_measurements = {}
        for areg,ameas,pmask in zip(ancilla_regs[::-1],ancilla_measurements,postselection_masks[::-1]):
            index = int(areg.name.split('_')[-1])
            final_measurements[index] = []
            ameas_post = ameas[pmask]
            freqs_post = freqs[pmask]
            for row in ameas_post.T[::-1]: # CHECK: reverse order?
                final_measurements[index].append(
                    {0: np.sum(freqs_post[row == 0]), 1:np.sum(freqs_post[row == 1])}
                )

        return final_measurements
        
    def _process_measurements(self, counts, basis):
        """
        """
        post_meas = self.postselect_measurements(counts, basis)
        _,meas_list = zip(*sorted(post_meas.items()))
        meas_list = [a for b in meas_list for a in b]
        return np.array([[ev_meas.get(0,0), ev_meas.get(1,0)] for ev_meas in meas_list])

    def bootstrap_postselected_data(self, _01_vals):
        """
        """
        vals, p0 = get_expectation_values(_01_vals)
        n_shots = (p0 * self.qpu_job.inputs['run_options']['shots']).astype(int)
        shape = list(_01_vals.shape); shape[-1]=1
        normalized = _01_vals/p0.reshape(shape)
        if normalized.ndim == 2:
            n_shots    = n_shots.reshape(1,-1)
            normalized = normalized.reshape(1,normalized.shape[0],2)
        resampled_vals = []
        for _n_shots,_normalized in zip(n_shots,normalized):
            _resampled_vals = []
            for ns, prob in zip(_n_shots,_normalized):
                _01_resampled = np.vstack([np.random.multinomial(ns, prob) for _ in range(self.n_resamples)])/ns
                bs_expvals,_ = get_expectation_values(_01_resampled)
                _resampled_vals.append(bs_expvals)
            _resampled_vals = np.vstack(_resampled_vals)
            resampled_vals.append(_resampled_vals)
        return vals, p0, np.asarray(resampled_vals)
        # deviations = np.array(deviations).reshape(n_shots.shape)
        # deviations[abs(deviations) < 1e-10] = 1e-10 # limit zero values for WLS
        # return deviations

    def process_measurements(self):
        """
        """
        # retrieve the qpu job results
        XYZ_counts = self.retrieve_results()
        print('\t>> retrieved measurement data')
        # the first three count dictionaries correspond with the XYZ
        # basis measurements of the full echo verification circuit
        _X01_vals = self._process_measurements(XYZ_counts.pop(0), 'X')
        _Y01_vals = self._process_measurements(XYZ_counts.pop(0), 'Y')
        _Z01_vals = self._process_measurements(XYZ_counts.pop(0), 'Z')
        # process the measurement data to get expectation values
        self._X_vals, self._X_p0, self._X_vals_bootstrapped = self.bootstrap_postselected_data(_01_vals=_X01_vals)
        self._Y_vals, self._Y_p0, self._Y_vals_bootstrapped = self.bootstrap_postselected_data(_01_vals=_Y01_vals)
        self._Z_vals, self._Z_p0, self._Z_vals_bootstrapped = self.bootstrap_postselected_data(_01_vals=_Z01_vals)
        # the remaining results are Clifford training data, X basis first then Z basis
        _X01_vals_clifford = np.vstack([self._process_measurements(c,'X').reshape(1,-1,2) for c in XYZ_counts[:self.num_clifford_samples]])
        _Z01_vals_clifford = np.vstack([self._process_measurements(c,'Z').reshape(1,-1,2) for c in XYZ_counts[self.num_clifford_samples:]])
        # process the measurement data to get expectation values
        self._X_vals_clifford, self._X_p0_clifford, self._X_vals_clifford_bootstrapped = self.bootstrap_postselected_data(_01_vals=_X01_vals_clifford)
        self._Z_vals_clifford, self._Z_p0_clifford, self._Z_vals_clifford_bootstrapped = self.bootstrap_postselected_data(_01_vals=_Z01_vals_clifford)
        print('\t>> extracted noisy expectation values')
        
    def process_batches(self, step):

        self.X_p0 = []
        self.Y_p0 = []
        self.Z_p0 = []
        self.X_vals = []
        self.Y_vals = []
        self.Z_vals = []
        self.X_vals_bootstrapped = []
        self.Y_vals_bootstrapped = []
        self.Z_vals_bootstrapped = []
        self.X_p0_clifford = []
        self.Z_p0_clifford = []
        self.X_vals_clifford = []
        self.Z_vals_clifford = []
        self.X_vals_clifford_bootstrapped = []
        self.Z_vals_clifford_bootstrapped = []
        self.clifford_ideal_vals = []

        for b in self.batches:
            self.load_job(step,b)
            self.process_measurements()

            self.X_p0.append(self._X_p0)
            self.Y_p0.append(self._Y_p0)
            self.Z_p0.append(self._Z_p0)
            self.X_vals.append(self._X_vals)
            self.Y_vals.append(self._Y_vals)
            self.Z_vals.append(self._Z_vals)
            self.X_vals_bootstrapped.append(self._X_vals_bootstrapped)
            self.Y_vals_bootstrapped.append(self._Y_vals_bootstrapped)
            self.Z_vals_bootstrapped.append(self._Z_vals_bootstrapped)
            self.X_p0_clifford.append(self._X_p0_clifford)
            self.Z_p0_clifford.append(self._Z_p0_clifford)
            self.X_vals_clifford.append(self._X_vals_clifford)
            self.Z_vals_clifford.append(self._Z_vals_clifford)
            self.X_vals_clifford_bootstrapped.append(self._X_vals_clifford_bootstrapped)
            self.Z_vals_clifford_bootstrapped.append(self._Z_vals_clifford_bootstrapped)
            self.clifford_ideal_vals.append(
                np.tile(self._clifford_ideal_vals.reshape(-1,1), self._X_vals_clifford.shape[1])
            )
            
        self.X_p0                         = np.hstack(self.X_p0)
        self.Y_p0                         = np.hstack(self.Y_p0)
        self.Z_p0                         = np.hstack(self.Z_p0)
        self.X_vals                       = np.hstack(self.X_vals)
        self.Y_vals                       = np.hstack(self.Y_vals)
        self.Z_vals                       = np.hstack(self.Z_vals)
        self.X_vals_bootstrapped          = np.hstack(self.X_vals_bootstrapped)[0]
        self.Y_vals_bootstrapped          = np.hstack(self.Y_vals_bootstrapped)[0]
        self.Z_vals_bootstrapped          = np.hstack(self.Z_vals_bootstrapped)[0]
        self.X_p0_clifford                = np.hstack(self.X_p0_clifford)
        self.Z_p0_clifford                = np.hstack(self.Z_p0_clifford)
        self.X_vals_clifford              = np.hstack(self.X_vals_clifford)
        self.Z_vals_clifford              = np.hstack(self.Z_vals_clifford)
        self.X_vals_clifford_bootstrapped = np.hstack(self.X_vals_clifford_bootstrapped)
        self.Z_vals_clifford_bootstrapped = np.hstack(self.Z_vals_clifford_bootstrapped)
        self.clifford_ideal_vals          = np.hstack(self.clifford_ideal_vals)
        self.X_deviations                 = np.std(self.X_vals_bootstrapped, axis=-1); self.X_deviations[abs(self.X_deviations) < 1e-10] = 1e-10
        self.Y_deviations                 = np.std(self.Y_vals_bootstrapped, axis=-1); self.Y_deviations[abs(self.Y_deviations) < 1e-10] = 1e-10
        self.Z_deviations                 = np.std(self.Z_vals_bootstrapped, axis=-1); self.Z_deviations[abs(self.Z_deviations) < 1e-10] = 1e-10
        self.X_deviations_clifford        = np.std(self.X_vals_clifford_bootstrapped, axis=-1); self.X_deviations_clifford[abs(self.X_deviations_clifford) < 1e-10] = 1e-10
        self.Z_deviations_clifford        = np.std(self.Z_vals_clifford_bootstrapped, axis=-1); self.Z_deviations_clifford[abs(self.Z_deviations_clifford) < 1e-10] = 1e-10

    def clifford_fitting(self):
        """
        """
        self.X_vals_clifford_ideal = (1-self.clifford_ideal_vals**2)/(1+self.clifford_ideal_vals**2)
        self.Z_vals_clifford_ideal = 2*self.clifford_ideal_vals/(1+self.clifford_ideal_vals**2)

        if self.normalize_clifford_purity:
            normalization = self.purity(
                _X_vals=self.X_vals_clifford, _Y_vals=0, 
                _Z_vals=self.Z_vals_clifford
            ).reshape(self.X_vals_clifford.shape)
        else:
            normalization = 1

        if self.num_clifford_samples == 1:
            self.clifford_X_fitted = [np.poly1d([0,self.X_vals_clifford_ideal[0,0]]) for _ in range(self.X_vals_clifford.shape[1])]
            self.clifford_Z_fitted = [np.poly1d([0,self.Z_vals_clifford_ideal[0,0]]) for _ in range(self.Z_vals_clifford.shape[1])]
            self.X_sosr = np.zeros(len(self.clifford_X_fitted))
            self.Z_sosr = np.zeros(len(self.clifford_Z_fitted))
        else:
            if not self.use_old_regression:
                self.clifford_X_fitted = []
                self.X_sosr = []
                for noisy,ideal,stdev in zip(
                        (self.X_vals_clifford/normalization).T,
                        self.X_vals_clifford_ideal.T,
                        self.X_deviations_clifford.T
                    ):
                    exog = sm.add_constant(ideal)
                    wlsfit = sm.WLS(endog=noisy, exog=exog, weights=1/np.square(stdev)).fit()
                    c,m = wlsfit.params
                    self.clifford_X_fitted.append(np.poly1d([1/m,-c/m])) # inverse
                    self.X_sosr.append(wlsfit.rsquared)
                self.X_sosr = np.array(self.X_sosr)

                self.clifford_Z_fitted = []
                self.Z_sosr = []
                for noisy,ideal,stdev in zip(
                        (self.Z_vals_clifford/normalization).T,
                        self.Z_vals_clifford_ideal.T,
                        self.Z_deviations_clifford.T
                    ):
                    exog = sm.add_constant(ideal)
                    if self.force_zero_Z_shift:
                        exog[:,0]=0
                    wlsfit = sm.WLS(endog=noisy, exog=exog, weights=1/np.square(stdev)).fit()
                    c,m = wlsfit.params
                    self.clifford_Z_fitted.append(np.poly1d([1/m,-c/m])) # inverse
                    self.Z_sosr.append(wlsfit.rsquared)
                self.Z_sosr = np.array(self.Z_sosr)
            else:
                self.clifford_X_fitted = []
                self.X_sosr = []
                for noisy,ideal,stdev in zip(
                        (self.X_vals_clifford/normalization).T,
                        self.X_vals_clifford_ideal.T,
                        self.X_deviations_clifford.T
                    ):
                    xnoise = sm.add_constant(noisy)
                    wlsfit = sm.WLS(endog=ideal, exog=xnoise, weights=1/np.square(stdev)).fit()
                    self.clifford_X_fitted.append(np.poly1d(wlsfit.params[::-1]))
                    self.X_sosr.append(wlsfit.rsquared)
                self.X_sosr = np.array(self.X_sosr)

                self.clifford_Z_fitted = []
                self.Z_sosr = []
                for noisy,ideal,stdev in zip(
                        (self.Z_vals_clifford/normalization).T,
                        self.Z_vals_clifford_ideal.T,
                        self.Z_deviations_clifford.T
                    ):
                    xnoise = sm.add_constant(noisy)
                    wlsfit = sm.WLS(endog=ideal, exog=xnoise, weights=1/np.square(stdev)).fit()
                    self.clifford_Z_fitted.append(np.poly1d(wlsfit.params[::-1]))
                    self.Z_sosr.append(wlsfit.rsquared)
                self.Z_sosr = np.array(self.Z_sosr)

    def n_postselected(self):
        """
        """
        return (self.X_p0+self.Y_p0+self.Z_p0)/3 * self.qpu_job.inputs['run_options']['shots']

    def get_clifford_interpolated_data(self, use_bs=False):
        """
        """
        if self.normalize_clifford_purity:
            normalization = self.purity(_Y_vals=0)
        else:
            normalization = np.ones_like(self.purity(_Y_vals=0))

        if use_bs:
            X_vals = self.X_vals_bootstrapped/normalization.reshape(-1,1)
            Z_vals = self.Z_vals_bootstrapped/normalization.reshape(-1,1)
        else:
            X_vals = self.X_vals/normalization
            Z_vals = self.Z_vals/normalization
        X_interpolated = np.array([f(x) for f,x in zip(self.clifford_X_fitted,X_vals)])
        Z_interpolated = np.array([f(z) for f,z in zip(self.clifford_Z_fitted,Z_vals)])
        X_interpolated[X_interpolated>+1]=+1
        X_interpolated[X_interpolated< 0]=0
        Z_interpolated[Z_interpolated>+1]=+1
        Z_interpolated[Z_interpolated<-1]=-1
        return X_interpolated, Z_interpolated
    
    def ancilla_state(self, _X_vals=None, _Y_vals=None, _Z_vals=None):
        """
        """
        if _X_vals is None:
            _X_vals = self.X_vals
        if _Y_vals is None:
            _Y_vals = self.Y_vals
        if _Z_vals is None:
            _Z_vals = self.Z_vals
        _X_vals = np.asarray(_X_vals)
        _Y_vals = np.asarray(_Y_vals)
        _Z_vals = np.asarray(_Z_vals)
        
        rhos = (
            I + 
            _X_vals.reshape(-1,1,1) * X + 
            _Y_vals.reshape(-1,1,1) * Y +
            _Z_vals.reshape(-1,1,1) * Z
        ) / 2 # ancilla density matrices
        return rhos

    def purity(self, _X_vals=None, _Y_vals=None, _Z_vals=None):
        """
        """
        purity = np.array([np.trace(p@p) for p in self.ancilla_state(_X_vals,_Y_vals,_Z_vals)]).real
        purity[purity>1]=1
        return purity

    def raw_estimate(self):
        """
        """
        exp_X_reciprocal        = np.mean((1+self.X_vals_bootstrapped)**-1, axis=-1)
        exp_X_reciprocal_square = np.mean((1+self.X_vals_bootstrapped)**-2, axis=-1)
        exp_Z                   = np.mean(self.Z_vals_bootstrapped, axis=-1)
        exp_Z_square            = np.mean(self.Z_vals_bootstrapped**2, axis=-1)
        estimates = self.Z_vals/(1+self.X_vals)
        deviation = np.sqrt(exp_Z_square*exp_X_reciprocal_square-(exp_Z*exp_X_reciprocal)**2)
        return estimates, deviation/np.sqrt(self.n_postselected())
     
    def purity_normalized_estimate(self):
        """
        """
        exp_X_reciprocal        = np.mean((1+self.X_vals_bootstrapped)**-1, axis=-1)
        exp_X_reciprocal_square = np.mean((1+self.X_vals_bootstrapped)**-2, axis=-1)
        exp_Z                   = np.mean(self.Z_vals_bootstrapped, axis=-1)
        exp_Z_square            = np.mean(self.Z_vals_bootstrapped**2, axis=-1)
        estimates = self.Z_vals/(self.purity()+self.X_vals)
        deviation = np.sqrt(exp_Z_square*exp_X_reciprocal_square-(exp_Z*exp_X_reciprocal)**2)
        return estimates, deviation/np.sqrt(self.n_postselected())

    def spectral_purification_estimate(self, Z_threshold=0.1):
        """
        """
        X_vals_pure = []
        Z_vals_pure = []
        for rho in self.ancilla_state(_Y_vals=0):
            eigvals, eigvecs = np.linalg.eig(rho)
            dominant_eigvec = eigvecs[:,np.argmax(eigvals)].reshape([-1,1])
            exp_X = (dominant_eigvec.T.conjugate() @ X @ dominant_eigvec)[0,0]
            exp_Z = (dominant_eigvec.T.conjugate() @ Z @ dominant_eigvec)[0,0]  
            X_vals_pure.append(exp_X)
            Z_vals_pure.append(exp_Z)
        X_vals_pure = np.array(X_vals_pure).real
        X_vals_pure[X_vals_pure<0]=0
        Z_vals_pure = np.array(Z_vals_pure).real
        Z_vals_pure[abs(self.Z_vals)<Z_threshold] = self.Z_vals[abs(self.Z_vals)<Z_threshold]
        
        exp_X_reciprocal        = np.mean((1+self.X_vals_bootstrapped)**-1, axis=-1)
        exp_X_reciprocal_square = np.mean((1+self.X_vals_bootstrapped)**-2, axis=-1)
        exp_Z                   = np.mean(self.Z_vals_bootstrapped, axis=-1)
        exp_Z_square            = np.mean(self.Z_vals_bootstrapped**2, axis=-1)
        estimates = Z_vals_pure / (1+X_vals_pure)
        deviation = np.sqrt(exp_Z_square*exp_X_reciprocal_square-(exp_Z*exp_X_reciprocal)**2)
        return estimates, deviation/np.sqrt(self.n_postselected()) 

    def estimate_depolarization_rate(self):
        """
        """
        num_qubits_by_cluster = {}
        for creg in self.measurement_circuits['X'].cregs:
            cluster_index = int(creg.name.split('_')[-1])
            num_qubits_by_cluster.setdefault(cluster_index, 0) 
            num_qubits_by_cluster[cluster_index] += creg.size
        self.d = 2**np.array(num_qubits_by_cluster[0])
        p0 = (self.X_p0 + self.Y_p0 + self.Z_p0)/3
        delta = self.d * p0 * (1 - self.purity()) / (1 + np.sqrt( 2 * self.purity() - 1 ))
        return delta

    def depolarization_biased_estimate(self):
        """
        """
        delta = self.estimate_depolarization_rate()
        return self.raw_estimate() * (1 + 2*delta/(self.d*(1-delta))), None
        
    def clifford_interpolated_estimate(self):
        """
        """
        X_interpolated, Z_interpolated = self.get_clifford_interpolated_data()
        X_int_bs, Z_int_bs = self.get_clifford_interpolated_data(use_bs=True)
        exp_X_reciprocal        = np.mean((1+X_int_bs)**-1, axis=-1)
        exp_X_reciprocal_square = np.mean((1+X_int_bs)**-2, axis=-1)
        exp_Z                   = np.mean(Z_int_bs,    axis=-1)
        exp_Z_square            = np.mean(Z_int_bs**2, axis=-1)
        deviation = np.sqrt(exp_Z_square*exp_X_reciprocal_square-(exp_Z*exp_X_reciprocal)**2)
        estimates = Z_interpolated/(1+X_interpolated)
        return estimates, deviation/np.sqrt(self.n_postselected())  

    def clifford_interpolated_estimate_Z_only(self):
        """
        """
        X_interpolated, Z_interpolated = self.get_clifford_interpolated_data()
        X_int_bs, Z_int_bs = self.get_clifford_interpolated_data(use_bs=True)
        exp_Z_reciprocal        = np.mean((1+np.sqrt(1-Z_int_bs**2))**-1, axis=-1)
        exp_Z_reciprocal_square = np.mean((1+np.sqrt(1-Z_int_bs**2))**-2, axis=-1)
        exp_Z                   = np.mean(Z_int_bs,    axis=-1)
        exp_Z_square            = np.mean(Z_int_bs**2, axis=-1)
        deviation = np.sqrt(exp_Z_square*exp_Z_reciprocal_square-(exp_Z*exp_Z_reciprocal)**2)
        estimates = Z_interpolated/(1+np.sqrt(1-Z_interpolated**2))
        return estimates, deviation/np.sqrt(self.n_postselected())   

    def clifford_interpolated_estimate_X_only(self):
        """
        """
        X_interpolated, Z_interpolated = self.get_clifford_interpolated_data()
        X_int_bs, Z_int_bs = self.get_clifford_interpolated_data(use_bs=True)
        exp_X_reciprocal        = np.mean(np.sqrt(1+X_int_bs)**-1, axis=-1)
        exp_X_reciprocal_square = np.mean(np.sqrt(1+X_int_bs)**-2, axis=-1)
        exp_X                   = np.mean(np.sqrt(1-X_int_bs),     axis=-1)
        exp_X_square            = np.mean(np.sqrt(1-X_int_bs)**2,  axis=-1)
        deviation = np.sqrt(exp_X_square*exp_X_reciprocal_square-(exp_X*exp_X_reciprocal)**2)
        estimates = np.sign(Z_interpolated) * np.sqrt((1-X_interpolated)/(1+X_interpolated))
        return estimates, deviation/np.sqrt(self.n_postselected())

    def get_results(self, neighbourhood_list=None, hamming_list=None, force_Z_shift_list=None):
        """
        """

        self.results = {
            'times':[],
            'steps':[],
            'purities':{},
            'depolarization_rates':{},
            'circuit_depths':{},
            'circuit_nonlocal_depths':{},
            'active_qubits':{},
            'expvals':{
                'raw':{},
                'exact':self.standard_exact,
                'density_purified':{},
                'purity_normalized':{},
                'clifford_averaged':{},
                'clifford_interpolated':{},
                'clifford_interpolated_X_only':{},
                'clifford_interpolated_Z_only':{},
                'depolarization_biased':{}
            },
            'clifford_data':{
                'X':{'noisy':{},'stdev':{},'ideal':{},'curve':{},'sosr':{}},
                'Z':{'noisy':{},'stdev':{},'ideal':{},'curve':{},'sosr':{}},
            },
            'standard_data':{
                'X':{'expval':{},'p0':{}},
                'Y':{'expval':{},'p0':{}},
                'Z':{'expval':{},'p0':{}}
            }
        }

        for index,step in enumerate(self.steps):
            
            if neighbourhood_list is not None:
                self.ancilla_neighbourhood_size = neighbourhood_list[index]
            if hamming_list is not None:
                self.max_hamming_distance = hamming_list[index]
            if force_Z_shift_list is not None:
                self.force_zero_Z_shift = force_Z_shift_list[index]

            message = f'| Processing step number {step} |'
            print('-'*len(message));print(message);print('-'*len(message))
            # load in and process the data
            self.process_batches(step=step)
            try:
                self.clifford_fitting()
                self.results['active_qubits'][str(step)]  = get_active_qubits(self.measurement_circuits['X'])
                self.results['circuit_depths'][str(step)] = self.measurement_circuits['X'].depth()
                qc = self.measurement_circuits['X']
                _qc = QuantumCircuit(qc.num_qubits)
                for inst in qc.data:
                    if len(inst.qubits) == 2: _qc.data.append(inst)
                self.results['circuit_nonlocal_depths'][str(step)] = _qc.depth()
                # evaluate estimates
                self.results['purities'][str(step)]                         = self.purity()
                self.results['depolarization_rates'][str(step)]             = self.estimate_depolarization_rate()
                self.results['expvals']['raw'][str(step)]                   = self.raw_estimate()
                self.results['expvals']['density_purified'][str(step)]      = self.spectral_purification_estimate()
                self.results['expvals']['purity_normalized'][str(step)]     = self.purity_normalized_estimate()
                # self.results['expvals']['clifford_averaged'][str(step)]     = self.clifford_averaged_estimate()
                self.results['expvals']['clifford_interpolated'][str(step)] = self.clifford_interpolated_estimate()
                self.results['expvals']['clifford_interpolated_Z_only'][str(step)] = self.clifford_interpolated_estimate_Z_only()
                self.results['expvals']['clifford_interpolated_X_only'][str(step)] = self.clifford_interpolated_estimate_X_only()
                self.results['expvals']['depolarization_biased'][str(step)] = self.depolarization_biased_estimate()
                # store the clifford training data
                self.results['clifford_data']['X']['noisy'][str(step)]  = self.X_vals_clifford
                self.results['clifford_data']['Z']['noisy'][str(step)]  = self.Z_vals_clifford
                self.results['clifford_data']['X']['stdev'][str(step)]  = self.X_deviations_clifford
                self.results['clifford_data']['Z']['stdev'][str(step)]  = self.Z_deviations_clifford
                self.results['clifford_data']['X']['ideal'][str(step)]  = self.X_vals_clifford_ideal
                self.results['clifford_data']['Z']['ideal'][str(step)]  = self.Z_vals_clifford_ideal
                self.results['clifford_data']['X']['curve'][str(step)]  = self.clifford_X_fitted
                self.results['clifford_data']['Z']['curve'][str(step)]  = self.clifford_Z_fitted
                self.results['clifford_data']['X']['sosr'][str(step)]   = self.X_sosr
                self.results['clifford_data']['Z']['sosr'][str(step)]   = self.Z_sosr
                # store the standard echo verification expecttion values
                self.results['standard_data']['X']['expval'][str(step)] = self.X_vals
                self.results['standard_data']['X']['p0'][str(step)] = self.X_p0
                self.results['standard_data']['Y']['expval'][str(step)] = self.Y_vals
                self.results['standard_data']['Y']['p0'][str(step)] = self.Y_p0
                self.results['standard_data']['Z']['expval'][str(step)] = self.Z_vals
                self.results['standard_data']['Z']['p0'][str(step)] = self.Z_p0
                self.results['times'].append(self.times[index])
                self.results['steps'].append(step)
            except:
                print('Failed to process, continuing to next step...')