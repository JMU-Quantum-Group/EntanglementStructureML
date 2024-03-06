import numpy as np


def random_state(n_qubit):
    n = 2**n_qubit
    real_current_matrix = np.random.rand(n, n)
    real_current_matrix = np.matrix(real_current_matrix)

    imaginary_current_matrix = np.random.rand(n, n)
    imaginary_current_matrix = np.matrix(imaginary_current_matrix)

    current_matrix = real_current_matrix + 1j * imaginary_current_matrix

    state = current_matrix * np.transpose(np.conj(current_matrix))
    return np.array(state / state.trace())


def create_dataset(n_qubit, num_of_quantum_state):
    matrix_size = 2 ** n_qubit
    matrix_list = list()
    for i in range(num_of_quantum_state):
        current_matrix = random_state(matrix_size)
        matrix_list.append(current_matrix)


def select_elements(matrix, num_element):
    matrix_copy = np.copy(matrix)

    # Calculate the magnitude of each element
    magnitude = np.abs(matrix_copy)

    # Set the lower triangular elements to 0
    magnitude[np.tril_indices(magnitude.shape[0])] = 0

    # Get the indices of the n elements with the largest magnitude
    indices = np.dstack(np.unravel_index(np.argsort(magnitude.ravel()), magnitude.shape))[0]

    # Filter out the indices of the diagonal elements
    indices = [index for index in indices if index[0] != index[1]]

    # Return the indices of the n non-diagonal elements with the largest magnitude
    return indices[-num_element:]


if __name__ == "__main__":
    dataset_name = "2-prod"
    k = 2
    n_qubit = 4
    num_of_quantum_state = 5000

    # partition_list = generate_k_producible_partitions(n_qubit, k)

    current_matrix = random_state(n_qubit)
    print(current_matrix)
    elements = select_elements(current_matrix, 5)
    print(elements)
