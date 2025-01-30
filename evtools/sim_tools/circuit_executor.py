from .embed_cluster_circuits import EmbedClusterCircuits
from .utils import split_registers, get_active_qubits, unitary_2x2, FinishedJob, I, X, Y, Z
from symmer import PauliwordOp
from symmer.evolution.circuit_symmerlator import CircuitSymmerlator
from scipy.optimize import minimize, NonlinearConstraint
from qiskit import ClassicalRegister, transpile
from qiskit_ibm_runtime import Sampler
import mthree as m3
import numpy as np



class CircuitExecutor(EmbedClusterCircuits):

    n_shots = 2 ** 10
    spectral_purification  = True
    max_clifford_cycles = 200
    max_non_clifford_gates = 10
    max_unique_clifford_vals = 10


    def __init__(self, circuit, backend, session=None, offline=False):
        super().__init__(circuit, backend)
        self.measurement_circuits = None
        # self.mem = None
        self.qpu_job = None
        if offline:
            self.sampler = backend
        else:
            if session is not None:
                self.sampler = Sampler(session=session)
            else:
                self.sampler = Sampler(backend=backend)
        self.clifford_ideal_vals = None
        
    # def initialize_m3(self):
    #     """
    #     """
    #     self.mem = m3.M3Mitigation(system=self.backend)
    #     self.mem.cals_from_system()

    # def apply_m3_correction(self, counts):
    #     """
    #     """
    #     split_sizes = [reg.size for reg in self.measurement_circuits['Z'].cregs[::-1]]
    #     counts_mem = self.mem.apply_correction(
    #         counts=join_registers(counts), 
    #         qubits=list(get_active_qubits(self.sys_qc))
    #     ).nearest_probability_distribution()
    #     counts_mem = split_registers(counts_mem, split_sizes)
    #     return counts_mem

    def _get_measurement_circuit(self, basis='Z'):
        """
        """
        assert basis in ['X', 'Y', 'Z'], 'Measurement basis must be X,Y or Z.'
        meas_qc = self.sys_qc.copy()
        for i in self.clusters.keys():
            compute = self.compute_positions[i]
            ancilla = self.ancilla_positions[i]
            cr_com = ClassicalRegister(len(compute), name=f'sys_{i}')
            cr_anc = ClassicalRegister(len(ancilla), name=f'anc_{i}')
            # change-of-basis on ancilla qubits
            if basis == 'Y':
                for q in ancilla:
                    meas_qc.sdg(q)
            if basis != 'Z':
                for q in ancilla:
                    meas_qc.h(q)
            meas_qc.add_register(cr_com)
            meas_qc.add_register(cr_anc)
            meas_qc.measure(compute, cr_com)
            meas_qc.measure(ancilla, cr_anc)
        meas_qc = transpile(
            meas_qc,
            optimization_level=1, 
            basis_gates=self.config.basis_gates, 
            coupling_map=self.config.coupling_map
        )
        return meas_qc
    
    def get_measurement_circuits(self):
        """
        """
        self.measurement_circuits = {
            'X':self._get_measurement_circuit('X'),
            'Y':self._get_measurement_circuit('Y'),
            'Z':self._get_measurement_circuit('Z'),
        }
        
    def prepare_circuits(self, pauli_measurement_dict):
        """
        """
        _,self.pauli_measurement_list = zip(*sorted(pauli_measurement_dict.items()))
        self.pauli_measurement_list = [a for b in self.pauli_measurement_list for a in b] # flatten
        self.build_echo_verification_circuits(pauli_measurement_dict)
        self.transpile_clusters()
        self.embed_clusters()
        self.get_measurement_circuits()
    
    def get_supported_params(self, params):
        """
        """
        assert self.sys_qc is not None, 'Need to run CircuitExecutor.prepare_circuits first'
        params = np.asarray(params, dtype=float)
        supported_params = np.array([p.index for p in self.sys_qc.parameters])
        return params[supported_params]
    
    def generate_clifford_training_circuits(self, params):
        """
        """
        n_qubits = self.primitive_qc.num_qubits
        params = np.asarray(params, dtype=float)
        params[:n_qubits]*=(-self.h*self.step_size)
        params[n_qubits:]*=(-self.J*self.step_size)
        supported_params = np.array([p.index for p in self.sys_qc.parameters])
        num_suppored_params = len(supported_params)
        _params = params[supported_params]
        num_non_clifford_gates = min(
            int(np.ceil(num_suppored_params/2)), 
            self.max_non_clifford_gates
        )

        clifford_ideal_vals = []
        clifford_X_circuits = []
        clifford_Z_circuits = []

        num_unique_vals = 0
        count = 0
        while count < self.max_clifford_cycles and num_unique_vals < self.max_unique_clifford_vals:
            count += 1
            _params_temp = _params.copy()
            index_clifford = np.sort(np.random.choice(np.arange(_params.shape[0]), _params.shape[0]-num_non_clifford_gates, replace=False))
            _params_temp[index_clifford] = np.round(_params[index_clifford]*2/np.pi)*np.pi/2
            _params_temp = _params_temp/_params
            params_temp  = np.zeros_like(params)
            params_temp[supported_params] = _params_temp
            self.primitive_cliffordized = self.primitive_qc.bind_parameters(params_temp)
            clifford_val = CircuitSymmerlator.from_qiskit(self.primitive_cliffordized).evaluate(
                PauliwordOp.from_list([self.pauli_measurement_list[0]])
            )
            if clifford_val not in clifford_ideal_vals and abs(clifford_val)>1e-3:
                clifford_ideal_vals.append(clifford_val)
                clifford_X_circuits.append(self.measurement_circuits['X'].bind_parameters(_params_temp))
                clifford_Z_circuits.append(self.measurement_circuits['Z'].bind_parameters(_params_temp))
                num_unique_vals += 1
                
        self.clifford_ideal_vals = np.array(clifford_ideal_vals).real
        self._num_unique_clifford_vals = len(self.clifford_ideal_vals)
        self._num_non_clifford_gates   = num_non_clifford_gates

        return clifford_X_circuits, clifford_Z_circuits
    
    def submit_to_backend(self, params):
        """
        """
        # self.clifford_handling = False
        _params = self.get_supported_params(params)
        X_bound = self.measurement_circuits['X'].bind_parameters(_params)
        Y_bound = self.measurement_circuits['Y'].bind_parameters(_params)
        Z_bound = self.measurement_circuits['Z'].bind_parameters(_params)
        X_bound_clifford, Z_bound_clifford = self.generate_clifford_training_circuits(params)
        self.qpu_job = self.sampler.run(
            [X_bound,Y_bound,Z_bound]+X_bound_clifford+Z_bound_clifford, 
            shots=self.n_shots
        )
        print('Submitted job to backend.')

    # def load_qpu_job(self, job_id=None, filename=None):
    #     """
    #     """
    #     if job_id is not None:
    #         self.qpu_job = self.backend.service.job(job_id)
    #     else:
    #         assert filename is not None
    #         self.qpu_job = FinishedJob(filename)
    #     self.measurement_circuits = dict(zip(['X','Y','Z'], self.qpu_job.inputs['circuits']))