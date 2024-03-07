import time

import numpy as np

from Full_Sep_SDP import FullSepSDP
from GD_SDP import GD_SDP
from partition_tools import generate_k_producible_partitions, generate_k_partitionable_partitions


def compute_all_4_qubit(rho):
    print("----------- full sep -------------")
    partition_4_part = generate_k_partitionable_partitions(4, 4)
    current_class = FullSepSDP(4, 1000, rho, partition_4_part, 1)
    current_class.train(200)
    p_value_full_sep = current_class.sdp()
    print("Full Sep:", p_value_full_sep)

    print("----------- 3 part -------------")
    partition_3_part = generate_k_partitionable_partitions(4, 3)
    current_class = GD_SDP(4, 300, rho, partition_3_part, 1)
    current_class.train(200)
    p_value_3_part = current_class.sdp()
    print("3 part:", p_value_3_part)

    print("----------- 2 prod -------------")
    partition_2_prod = generate_k_producible_partitions(4, 2)
    current_class = GD_SDP(4, 1000, rho, partition_2_prod, 1)
    current_class.train(200)
    p_value_2_prod = current_class.sdp()
    print("2 prod:", p_value_2_prod)

    print("----------- 2 part -------------")
    partition_2_part = generate_k_partitionable_partitions(4, 2)
    current_class = GD_SDP(4, 300, rho, partition_2_part, 1)
    current_class.train(200)
    p_value_2_part = current_class.sdp()
    print("2 part:", p_value_2_part)

    return [p_value_full_sep, p_value_3_part, p_value_2_prod, p_value_2_part]


def random_state(n_qubit):
    n = 2 ** n_qubit
    real_current_matrix = np.random.rand(n, n)
    real_current_matrix = np.matrix(real_current_matrix)

    imaginary_current_matrix = np.random.rand(n, n)
    imaginary_current_matrix = np.matrix(imaginary_current_matrix)

    current_matrix = real_current_matrix + 1j * imaginary_current_matrix

    state = current_matrix * np.transpose(np.conj(current_matrix))
    return np.array(state / state.trace())


def create_dataset(n_qubit, num_of_quantum_state):
    matrix_list = list()
    label_list = list()
    for i in range(num_of_quantum_state):
        current_matrix = random_state(n_qubit)
        bound_result = compute_all_4_qubit(current_matrix)
        matrix_list.append(current_matrix)


def decimal_to_binary(n, length):
    return bin(n)[2:].zfill(length)


def select_elements(matrix, num_element):
    matrix_copy = np.copy(matrix)

    # Calculate the magnitude of each element
    magnitude = np.abs(matrix_copy)

    # Set the lower triangular elements to 0
    magnitude[np.tril_indices(magnitude.shape[0])] = 0

    # Get the indices of the n elements with the largest magnitude
    indices = np.dstack(np.unravel_index(np.argsort(magnitude.ravel()), magnitude.shape))[0]

    # Filter out the indices of the diagonal elements
    indices = [index for index in indices if
               index[0] != index[1] or ((index[1] - index[0]) & (index[1] - index[0] - 1) == 0)]

    # Return the indices of the n non-diagonal elements with the largest magnitude
    return indices[-num_element:]


if __name__ == "__main__":
    dataset_name = "2-prod"
    k = 2
    n_qubit = 4
    num_of_quantum_state = 5000

    current_matrix = random_state(n_qubit)
    current_matrix_2 = random_state(n_qubit)

    print(current_matrix)
    print(current_matrix_2)
    print()
    print(current_matrix_2 - current_matrix)

    start_time = time.time()
    print(compute_all_4_qubit(current_matrix))
    end_time = time.time()
    print("time:", end_time - start_time)

    start_time_2 = time.time()
    print(compute_all_4_qubit(current_matrix_2))
    end_time_2 = time.time()
    print("time:", end_time_2 - start_time_2)



    # print("2 prod:", p_value_2_prod)
    #
    # elements = select_elements(current_matrix, 50)
    # # elements = [[0, 15]]
    # # elements = [[1, 2], [1, 4], [1, 8], [2, 4], [2, 8], [4, 8]]
    # P = list()
    # left_sum = 0
    # for element in elements:
    #     P.append([decimal_to_binary(element[0], n_qubit), decimal_to_binary(element[1], n_qubit)])
    #     left_sum += np.abs(current_matrix[element[0], element[1]])
    #     print(left_sum)
    #
    # print(P)
    # result_list = compute_all_producible(P, n_qubit)
    # for result_item in result_list:
    #     print(result_item)
    #     t = symbols('t')
    #     expression = 0
    #     for item in result_item:
    #         current_index = int(item[0], 2)
    #         expression += item[1] * (
    #                     t * np.abs(current_matrix[current_index, current_index]) + (1 - t) / (2 ** n_qubit))
    #         # if current_index & current_index - 1 == 0 and current_index != 0:
    #         #     expression += item[1] * (t * 0.25 + (1 - t) / 16)
    #         # else:
    #         #     expression += item[1] * (1 - t) / 16
    #     inequality = GreaterThan(t * left_sum, expression)
    #     # print(inequality)
    #
    #     solution = solve_univariate_inequality(inequality, t)
    #     print("solution:", solution)
