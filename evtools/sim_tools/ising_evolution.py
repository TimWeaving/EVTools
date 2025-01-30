from .circuit_executor import CircuitExecutor
from symmer import PauliwordOp
from symmer.evolution.circuit_symmerlator import CircuitSymmerlator
from qiskit.circuit import ParameterVector, QuantumCircuit
import numpy as np

class IsingEvolution(CircuitExecutor):

    num_steps = 3 
    final_time = 1
    calculate_exact = False

    J = -2
    h = 1
    
    def __init__(self, 
            backend, 
            cluster,
            obs_str,
            n_qubits,
            spin_couplings,
            session=None, 
            offline=False
        ) -> None :
        """ cluster of form {0:{"compute":[qubits], "ancilla":[qubits]}, 1 ...}
        """
        self.backend = backend
        self.cluster = cluster
        self.obs_str = obs_str
        self.session = session
        self.offline = offline
        self.n_qubits = n_qubits
        self.spin_couplings = spin_couplings

    def update_circuit(self, step_number):
        """
        """
        pvec = ParameterVector('P', self.n_qubits+len(self.spin_couplings))
        qc = QuantumCircuit(self.n_qubits)
        for _ in range(step_number):
            for i in range(self.n_qubits):
                qc.rx(-self.h*self.step_size*pvec[i], i)
            for param_index,(i,j) in enumerate(self.spin_couplings):
                if j<i:
                    i,j=j,i
                qc.cx(i,j)
                qc.rz(-self.J*self.step_size*pvec[param_index+self.n_qubits], j)
                qc.cx(i,j)

        super().__init__(
            circuit=qc, 
            backend=self.backend, 
            session=self.session, 
            offline=self.offline
        )
        for _,cluster in self.cluster.items():
            assert len(cluster['ancilla']) == 1, 'Currently only supports single-ancilla'
            self.add_cluster(
                compute_qubits=cluster['compute'],
                ancilla_qubits=cluster['ancilla']
            )
        paulistring_measurement_dict = {
            i:[self.obs_str] for i in self.cluster.keys()
        }
        self.prepare_circuits(paulistring_measurement_dict)

    def submit_circuits(self, folder_index):
        """
        """
        self.step_size = 2*self.final_time/self.num_steps
        steps = np.arange(1, self.num_steps+1)
        times = steps*self.step_size/2
        
        self.job_history   = {}
        self.clifford_data = {'times':times,'steps':steps,'ideal_data':{}}
        self.standard_data = {'times':times,'steps':steps,'expvals':{'exact':[]}}
        num_unique_clifford_vals = []
        num_non_clifford_gates   = []

        for step in steps:
            print(f'*** submitting step number {step} ***')
            self.update_circuit(step_number=step)
            self.submit_to_backend([1]*self.primitive_qc.num_parameters)        
            self.job_history[step] = self.qpu_job.job_id()
            self.clifford_data['ideal_data'][step] = self.clifford_ideal_vals
            qc_bound = self.primitive_qc.bind_parameters([1]*self.primitive_qc.num_parameters)
            if self.calculate_exact:
                self.standard_data['expvals']['exact'].append(
                    CircuitSymmerlator.from_qiskit(qc_bound).evaluate(
                        PauliwordOp.from_list([self.obs_str])
                    )
                )
            num_unique_clifford_vals.append(self._num_unique_clifford_vals)
            num_non_clifford_gates.append(self._num_non_clifford_gates)
    
        data_out = {
            'backend':self.backend.name,
            'clusters':self.cluster,
            'strengths':{'coupling':self.J, 'field':self.h},
            'step_size':self.step_size,
            'steps':list(map(int,self.standard_data['steps'].tolist())),
            'times':self.standard_data['times'].astype(float).tolist(),
            'job_ids':dict(zip(map(int,self.standard_data['steps'].tolist()), self.job_history.values())),
            'num_clifford_samples':num_unique_clifford_vals,
            'num_non_clifford_gates':num_non_clifford_gates,
            'standard_exact':dict(zip(map(int,self.standard_data['steps'].tolist()), np.array(self.standard_data['expvals']['exact']).real)),
            'clifford_exact':{int(i):l.tolist() for i,l in self.clifford_data['ideal_data'].items()}
        }
        with open(f'data/EVCDR/{folder_index}/{self.backend.name}_submission.json','w') as outfile:
            import json
            json.dump(data_out, outfile)