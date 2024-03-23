import numpy as np

from bin.data_3_part import generate_2_entanglement

if __name__ == "__main__":
    n_qubit = 4
    num_of_quantum_state = 1000

    entanglement_2_qubit = list()
    entanglement_2_qubit.append(0.5 * (np.outer(np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]).conj())))
    for _ in range(num_of_quantum_state):
        entanglement_2_qubit.append(generate_2_entanglement())

