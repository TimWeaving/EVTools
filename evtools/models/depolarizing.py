from cached_property import cached_property
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.opflow import CircuitStateFn
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.quantum_info import Statevector
from matplotlib import pyplot as plt
import numpy as np
from itertools import product
from functools import reduce

def tensor(factors, qubits=None, n_qubits=None):
    if qubits is not None:
        assert n_qubits is not None
        assert len(factors) == len(qubits)
        _factors = [I]*n_qubits
        for f,q in zip(factors, qubits):
            _factors[q] = f
        factors = _factors
    return reduce(np.kron, factors)

I = np.array([[1,0],[0,1]])
Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])

class DepolarizingModel:

    p = 0
    n_samples=10_000
    balance_Y = True
    target_qubits = []
    error_rates   = []

    def __init__(self, circuit):
        self.primitive_qc = circuit
        self.rho  = None
        self.circ = None
        self.n_qubits = circuit.num_qubits
        self.d = 2**(self.n_qubits+1)
    
    @cached_property
    def ancilla_X(self):
        anc_had = np.eye(2**self.n_qubits)
        anc_had = np.kron(np.array([[1,1],[1,-1]])/np.sqrt(2), anc_had)
        return anc_had
    
    @cached_property
    def ancilla_Y(self):
        anc_had = np.eye(2**self.n_qubits)
        anc_had = np.kron(
            np.array([[1,0],[0,1j]]) @ np.array([[1,1],[1,-1]])/np.sqrt(2), 
            anc_had
        )
        return anc_had
    
    @cached_property
    def zero_projector(self):
        zero_proj = np.zeros([2**self.n_qubits, 2**self.n_qubits])
        zero_proj[0,0] = 1
        zero_proj = np.kron(I, zero_proj)
        return zero_proj

    def reverse_qubit_order(self, q):
        """ Qiskit ordering is reversed
        """
        return self.primitive_qc.num_qubits - 1 - q
    
    def build_circuit(self, paulistring):
        """
        """
        ev_circuit = QuantumCircuit()
        qr = QuantumRegister(self.primitive_qc.num_qubits, 'sys')
        ar = AncillaRegister(1, 'anc')
        ev_circuit.add_register(qr)
        ev_circuit.add_register(ar)

        ev_circuit.compose(self.primitive_qc, qubits=qr, inplace=True)
        ev_circuit.barrier()
        non_identity_indices = [(q,P) for q,P in enumerate(paulistring) if P!='I']
        # change-of-basis
        for q,P in non_identity_indices:
            q = self.reverse_qubit_order(q)
            if P == 'Y':
                ev_circuit.sdg(qr[q])
            if P != 'Z':
                ev_circuit.h(qr[q])
        ev_circuit.barrier()
        # store parity in ancilla qubit
        indices_to_couple,_ = zip(*non_identity_indices)
        indices_to_couple = list(map(self.reverse_qubit_order, indices_to_couple))
        qubit_pairs = list(zip(indices_to_couple[:-1], indices_to_couple[1:]))
        for q1,q2 in qubit_pairs:
            ev_circuit.cx(q1,q2)
        ev_circuit.cx(indices_to_couple[-1],ar[0])
        for q1,q2 in qubit_pairs[::-1]:
            ev_circuit.cx(q1,q2)
        ev_circuit.barrier()
        # undo change-of-basis
        for q,P in non_identity_indices:
            q = self.reverse_qubit_order(q)
            if P != 'Z':
                ev_circuit.h(qr[q])
            if P == 'Y':
                ev_circuit.s(qr[q])
        ev_circuit.barrier()
        ev_circuit.compose(self.primitive_qc.inverse(), qubits=qr, inplace=True)
        self.circ = ev_circuit
    
    def build_rho(self):
        """
        """
        psi = Statevector(self.circ).data.reshape(-1,1)
        self.rho = np.outer(psi, psi.conj().T)

    def depolarize(self):
        I = np.eye(*self.rho.shape)
        return self.rho * (1-self.p) + I * self.p/self.rho.shape[0]
    
    def apply_pauli_errors(self):
        assert len(self.error_rates) == 4**len(self.target_qubits)-1
        assert sum(self.error_rates) <= 1
        _tensor = lambda factors:tensor(factors, qubits=self.target_qubits, n_qubits=self.n_qubits+1)
        pauli_tensors = list(map(_tensor, product([I,X,Y,Z], repeat=len(self.target_qubits))))
        depolarized = self.depolarize()
        rho_error_applied = sum([(P@depolarized@P)*p for p,P in zip(self.error_rates,pauli_tensors[1:])])
        return depolarized*(1-sum(self.error_rates)) + rho_error_applied
    
    def sample_state(self, basis='Z'):
        rho = self.apply_pauli_errors()
        if basis == 'X':
            rho = self.ancilla_X.conj().T @ rho @ self.ancilla_X
        elif basis == 'Y':
            rho = self.ancilla_Y.conj().T @ rho @ self.ancilla_Y
        else:
            assert basis=='Z', 'Invald basis supplied'
        prob_dist = abs(rho.diagonal().real)
        samples = np.random.multinomial(self.n_samples, prob_dist)
        return {
            np.binary_repr(index, self.n_qubits+1):freq for index,freq in enumerate(samples) if freq != 0
        }
    
    def estimate_rdm(self, sampled=True):
        """
        """
        if sampled:
            X_samples = self.sample_state(basis='X')
            Y_samples = self.sample_state(basis='Y')
            Z_samples = self.sample_state(basis='Z')
            Y0  = Y_samples.get('0'+'0'*self.n_qubits, 0)
            Y1  = Y_samples.get('1'+'0'*self.n_qubits, 0)
            if self.balance_Y:
                bal0 = (1 + Y1/Y0) / 2
                bal1 = (1 + Y0/Y1) / 2
            else:
                bal0=1
                bal1=1
            Y0 *= bal0
            Y1 *= bal1
            X0  = X_samples.get('0'+'0'*self.n_qubits, 0)*bal0
            X1  = X_samples.get('1'+'0'*self.n_qubits, 0)*bal1
            Z0  = Z_samples.get('0'+'0'*self.n_qubits, 0)*bal0
            Z1  = Z_samples.get('1'+'0'*self.n_qubits, 0)*bal1
            p0X = (X0 + X1)
            p0Y = (Y0 + Y1)
            p0Z = (Z0 + Z1)
            self.p0 = (p0X + p0Y + p0Z)/3/self.n_samples
            trXrho = (X0 - X1) / p0X
            trYrho = (Y0 - Y1) / p0Y
            trZrho = (Z0 - Z1) / p0Z
            rho_reduced = (I + X*trXrho + Y*trYrho + Z*trZrho)/2
        else:
            rho_project = self.zero_projector @ self.apply_pauli_errors() @ self.zero_projector
            self.p0 = np.trace(rho_project)
            rho_reduced = rho_project[rho_project.nonzero()].reshape(2,2)/self.p0
        self.purity = np.trace(rho_reduced @ rho_reduced)
        self.rho_reduced = rho_reduced
        
    def estimate_epsilson(self):
        quad = (2*self.purity-1+np.sqrt(2*self.purity-1))/(1-self.purity)
        eps = self.d * self.p0 / (2 + quad)
        return eps
     
    def EV_estimate(self, rho=None):
        if rho is None:
            rho = self.rho_reduced
        return np.trace(Z@rho)/(1+np.trace(X@rho))
    
    def EV_spectral_estimate(self, rho=None):
        if rho is None:
            rho = self.rho_reduced
        eigvals, eigvecs = np.linalg.eig(rho)
        dominant_eigvec = eigvecs[:,np.argmax(eigvals)].reshape(-1,1)
        dominant_rho = np.outer(dominant_eigvec, dominant_eigvec.conj().T)
        return self.EV_estimate(rho=dominant_rho)