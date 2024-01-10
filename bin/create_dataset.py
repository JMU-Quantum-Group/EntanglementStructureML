import numpy as np

from partition_tools import generate_k_producible_partitions


def random_state(n):
    real_current_matrix = np.random.rand(n, n)
    real_current_matrix = np.matrix(real_current_matrix)

    imaginary_current_matrix = np.random.rand(n, n)
    imaginary_current_matrix = np.matrix(imaginary_current_matrix)

    current_matrix = real_current_matrix + 1j * imaginary_current_matrix

    state = current_matrix * np.transpose(np.conj(current_matrix))
    return np.array(state / state.trace())


def create_dataset(n_qubit, num_of_quantum_state, partition_list):
    matrix_size = 2 ** n_qubit
    matrix_list = list()
    for i in range(num_of_quantum_state):
        current_matrix = random_state(matrix_size)


if __name__ == "__main__":
    dataset_name = "2-prod"
    k = 2
    n_qubit = 4
    num_of_quantum_state = 5000

    partition_list = generate_k_producible_partitions(n_qubit, k)
