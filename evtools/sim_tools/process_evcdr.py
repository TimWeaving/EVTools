from .utils import split_registers, get_active_qubits, get_neighbourhood, FinishedJob, I, X, Y, Z
from qiskit_ibm_runtime import QiskitRuntimeService
from mthree.probability import quasi_to_probs
import numpy as np

class ProcessEVCDR:
    """
    """
    ancilla_neighbourhood_size = 2
    max_hamming_distance = 2

    def __init__(self, folder_index, filename,  use_local_data=False):
        """
        """
        
        self.use_local_data = use_local_data
        self.directory = f'data/EVCDR/{folder_index}'
        self.identifier = filename.replace('_submission','')
        submission_filename = f'{self.directory}/{filename}.json'
        
        if not use_local_data:
            self.service = QiskitRuntimeService(channel='ibm_quantum')

        with open(submission_filename,'r') as infile:
            import json
            submission_data = json.load(infile)
        
        self.steps = submission_data['steps']
        self.times = submission_data['times']
        self.step_size = submission_data['step_size']
        self.job_ids = submission_data['job_ids']
        self.list_num_clifford_samples = submission_data['num_clifford_samples']
        self.list_num_non_clifford_gates = submission_data['num_non_clifford_gates']
        self.standard_exact = submission_data['standard_exact']
        self.clifford_exact = submission_data['clifford_exact']

    def load_job(self, index):
        """
        """
        if self.use_local_data:
            self.qpu_job = FinishedJob(f'{self.directory}/{self.identifier}/step_{index}.txt')
        else:
            self.qpu_job = self.service.job(self.job_ids[str(index)])
        self.measurement_circuits = dict(zip(['X','Y','Z'], self.qpu_job.inputs['circuits'][:3]))
        self.clifford_ideal_vals  = np.asarray(self.clifford_exact[str(index)])
        print('>> loaded qpu job')

    def retrieve_results(self):
        """
        """
        from qiskit_aer import AerJob
        if isinstance(self.qpu_job, AerJob):
            import time
            while self.qpu_job.status().name not in ['DONE', 'ERROR', 'CANCELLED']:
                time.sleep(0.1)
        if self.qpu_job == None:
            raise ValueError('No job has been submitted to the QPU.')
        status = self.qpu_job.status().name
        if status == 'ERROR':
            raise ValueError('The job errored - inspect qpu_job.error_message for further details.')
        elif status == 'QUEUED':
            raise ValueError('The job is still queued - exiting measurement processing.')
        elif status == 'RUNNING':
            est_runtime = self.qpu_job.usage_estimation['quantum_seconds']
            raise ValueError(f'The job is currently running - estimated runtime of {est_runtime:.0f} seconds.')
        elif status == 'DONE':
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
            else:
                return result.get_counts()
        else:
            raise ValueError('Unknown status of QPU job.')
        
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
        
    def process_measurements(self):
        """
        """
        # retrieve the qpu job results
        XYZ_counts = self.retrieve_results()
        print('>> retrieved measurement data')
        # the first three count dictionaries correspond with the XYZ
        # basis measurements of the full echo verification circuit
        _X01_vals = self._process_measurements(XYZ_counts.pop(0), 'X')
        _Y01_vals = self._process_measurements(XYZ_counts.pop(0), 'Y')
        _Z01_vals = self._process_measurements(XYZ_counts.pop(0), 'Z')
        # process the measurement data to get expectation values
        X0 = _X01_vals[:,0]
        X1 = _X01_vals[:,1]
        Y0 = _Y01_vals[:,0]
        Y1 = _Y01_vals[:,1]
        Z0 = _Z01_vals[:,0]
        Z1 = _Z01_vals[:,1]
        self._X_p0 = X0 + X1
        self._Y_p0 = Y0 + Y1
        self._Z_p0 = Z0 + Z1
        self._X_vals = (X0 - X1) / self._X_p0
        self._Y_vals = (Y0 - Y1) / self._Y_p0
        self._Z_vals = (Z0 - Z1) / self._Z_p0
        # the remaining results are Clifford training data, X basis first then Z basis
        _X01_vals_clifford = np.vstack([self._process_measurements(c,'X').reshape(1,-1,2) for c in XYZ_counts[:self.num_clifford_samples]])
        _Z01_vals_clifford = np.vstack([self._process_measurements(c,'Z').reshape(1,-1,2) for c in XYZ_counts[self.num_clifford_samples:]])
        # process the measurement data to get expectation values
        X0_clifford = _X01_vals_clifford[:,:,0]
        X1_clifford = _X01_vals_clifford[:,:,1]
        Z0_clifford = _Z01_vals_clifford[:,:,0]
        Z1_clifford = _Z01_vals_clifford[:,:,1]
        self._X_p0_clifford = X0_clifford + X1_clifford
        self._Z_p0_clifford = Z0_clifford + Z1_clifford
        self._X_vals_clifford = (X0_clifford - X1_clifford) / self._X_p0_clifford
        self._Z_vals_clifford = (Z0_clifford - Z1_clifford) / self._Z_p0_clifford
        print('>> extracted noisy expectation values')
        
    def clifford_fitting(self):
        """
        """
        assert self.clifford_ideal_vals is not None, 'Need to generate Clifford data first'
        self._X_vals_clifford_ideal = (1-self.clifford_ideal_vals**2)/(1+self.clifford_ideal_vals**2)
        self._Z_vals_clifford_ideal = 2*self.clifford_ideal_vals/(1+self.clifford_ideal_vals**2)
        # clifford_purities = self.purity(
        #     _X_vals=self._X_vals_clifford,
        #     _Y_vals=0,
        #     _Z_vals=self._Z_vals_clifford
        # )
        if self.num_clifford_samples == 1:

            self.clifford_X_fitted = [np.poly1d([0,self._X_vals_clifford_ideal[0]]) for _ in range(self._X_vals_clifford.shape[1])]
            self.clifford_Z_fitted = [np.poly1d([0,self._Z_vals_clifford_ideal[0]]) for _ in range(self._Z_vals_clifford.shape[1])]
            self.X_sosr = np.zeros(len(self.clifford_X_fitted))
            self.Z_sosr = np.zeros(len(self.clifford_Z_fitted))

        else:

            self.clifford_X_fitted = []
            self.X_sosr = []
            for cluster_clifford_val in self._X_vals_clifford.T:
                coeffs, sosr, rank, sv, rcond = np.polyfit(cluster_clifford_val, self._X_vals_clifford_ideal, deg=1, full=True)
                self.clifford_X_fitted.append(np.poly1d(coeffs))
                if len(sosr)!=0:
                    self.X_sosr.append(sosr[0])
                else:
                    self.X_sosr.append(0)
            self.X_sosr = np.array(self.X_sosr)

            self.clifford_Z_fitted = []
            self.Z_sosr = []
            for cluster_clifford_val in self._Z_vals_clifford.T:
                coeffs, sosr, rank, sv, rcond = np.polyfit(cluster_clifford_val, self._Z_vals_clifford_ideal, deg=1, full=True)
                self.clifford_Z_fitted.append(np.poly1d(coeffs))
                if len(sosr)!=0:
                    self.Z_sosr.append(sosr[0])
                else:
                    self.Z_sosr.append(0)
            self.Z_sosr = np.array(self.Z_sosr)

        print('>> completed clifford fitting procedure')

    def ancilla_state(self, _X_vals=None, _Y_vals=None, _Z_vals=None):
        """
        """
        if _X_vals is None:
            _X_vals = self._X_vals
        if _Y_vals is None:
            _Y_vals = self._Y_vals
        if _Z_vals is None:
            _Z_vals = self._Z_vals
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
        return np.array([np.trace(p@p) for p in self.ancilla_state(_X_vals,_Y_vals,_Z_vals)]).real
        
    def raw_estimate(self):
        """
        """
        return self._Z_vals/(1+self._X_vals)
    
    def purity_normalized_estimate(self):
        """
        """
        return self._Z_vals/(self.purity()+self._X_vals)
    
    def spectral_purification_estimate(self, Z_threshold=0.2):
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
        Z_vals_pure[abs(self._Z_vals)<Z_threshold] = self._Z_vals[abs(self._Z_vals)<Z_threshold]
        return Z_vals_pure / (1+X_vals_pure)
    
    def estimate_depolarization_rate(self):
        """
        """
        num_qubits_by_cluster = {}
        for creg in self.measurement_circuits['X'].cregs:
            cluster_index = int(creg.name.split('_')[-1])
            num_qubits_by_cluster.setdefault(cluster_index, 0) 
            num_qubits_by_cluster[cluster_index] += creg.size
        self.d = 2**np.array(list(num_qubits_by_cluster.values()))
        p0 = (self._X_p0 + self._Y_p0 + self._Z_p0)/3
        delta = self.d * p0 * (1 - self.purity()) / (1 + np.sqrt( 2 * self.purity() - 1 ))
        return delta

    def depolarization_biased_estimate(self):
        """
        """
        delta = self.estimate_depolarization_rate()
        return self.raw_estimate() * (1 + 2*delta/(self.d*(1-delta)))
    
    def get_clifford_interpolated_data(self):
        """
        """
        X_interpolated = np.array([f(x) for f,x in zip(self.clifford_X_fitted,self._X_vals)])
        Z_interpolated = np.array([f(z) for f,z in zip(self.clifford_Z_fitted,self._Z_vals)])
        X_interpolated[X_interpolated>+1]=+1
        X_interpolated[X_interpolated< 0]=0
        Z_interpolated[Z_interpolated>+1]=+1
        Z_interpolated[Z_interpolated<-1]=-1
        return X_interpolated, Z_interpolated
        
    def clifford_interpolated_estimate(self):
        """
        """
        X_interpolated, Z_interpolated = self.get_clifford_interpolated_data()
        return Z_interpolated/(1+X_interpolated)
    
    def clifford_interpolated_estimate_Z_only(self):
        """
        """
        X_interpolated, Z_interpolated = self.get_clifford_interpolated_data()
        return Z_interpolated/(1+np.sqrt(1-Z_interpolated**2))

    def clifford_interpolated_estimate_X_only(self):
        """
        """
        X_interpolated, Z_interpolated = self.get_clifford_interpolated_data()
        return np.sign(Z_interpolated) * np.sqrt((1-X_interpolated)/(1+X_interpolated))
    
    # def clifford_averaged_estimate(self):
    #     X_averaged = self._X_vals/self.clifford_X_lambda
    #     Z_averaged = self._Z_vals/self.clifford_Z_lambda
    #     return Z_averaged/(1+X_averaged)
    
    def get_results(self, neighbourhood_list=None, hamming_list=None):
        """
        """

        self.results = {
            'times':[],
            'steps':[],
            'purities':{},
            'depolarization_rates':{},
            'circuit_depths':{},
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
                'X':{'noisy':{},'ideal':{},'curve':{},'sosr':{}},
                'Z':{'noisy':{},'ideal':{},'curve':{},'sosr':{}},
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

            self.num_clifford_samples = self.list_num_clifford_samples[index]
            self.num_non_clifford_gates = self.list_num_non_clifford_gates[index]
            message = f'| Processing step number {step} |'
            print('-'*len(message));print(message);print('-'*len(message))
            # load in and process the data
            self.load_job(step)
            self.process_measurements()
            try:
                self.clifford_fitting()
                
                self.results['active_qubits'][str(step)]  = get_active_qubits(self.measurement_circuits['X'])
                self.results['circuit_depths'][str(step)] = self.measurement_circuits['X'].depth()
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
                self.results['clifford_data']['X']['noisy'][str(step)]  = self._X_vals_clifford
                self.results['clifford_data']['Z']['noisy'][str(step)]  = self._Z_vals_clifford
                self.results['clifford_data']['X']['ideal'][str(step)]  = self._X_vals_clifford_ideal
                self.results['clifford_data']['Z']['ideal'][str(step)]  = self._Z_vals_clifford_ideal
                self.results['clifford_data']['X']['curve'][str(step)]  = self.clifford_X_fitted
                self.results['clifford_data']['Z']['curve'][str(step)]  = self.clifford_Z_fitted
                self.results['clifford_data']['X']['sosr'][str(step)]   = self.X_sosr
                self.results['clifford_data']['Z']['sosr'][str(step)]   = self.Z_sosr
                # store the standard echo verification expecttion values
                self.results['standard_data']['X']['expval'][str(step)] = self._X_vals
                self.results['standard_data']['X']['p0'][str(step)] = self._X_p0
                self.results['standard_data']['Y']['expval'][str(step)] = self._Y_vals
                self.results['standard_data']['Y']['p0'][str(step)] = self._Y_p0
                self.results['standard_data']['Z']['expval'][str(step)] = self._Z_vals
                self.results['standard_data']['Z']['p0'][str(step)] = self._Z_p0
                self.results['times'].append(self.times[index])
                self.results['steps'].append(step)
            except:
                print('Failed to process, continuing to next step...')