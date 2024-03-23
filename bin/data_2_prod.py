import numpy as np
from sympy import symbols, GreaterThan, solve_univariate_inequality
import sympy

from bin.create_dataset import select_elements, decimal_to_binary
from bin.data_3_part import generate_2_entanglement, get_exchange_matrix, handle_u_matrix
from bin.data_full_sep import generate_train_matrix
from partition_tools import generate_k_partitionable_partitions
from tools import compute


def compute_2_prod(current_state):
    elements = select_elements(current_state, 5)
    P = list()
    left_sum = 0
    for element in elements:
        P.append([decimal_to_binary(element[0], n_qubit), decimal_to_binary(element[1], n_qubit)])
        left_sum += np.abs(current_state[element[0], element[1]])

    partition_3_part = generate_k_partitionable_partitions(4, 3)
    result_item = compute(P, partition_3_part)
    t = symbols('t')
    expression = 0
    for item in result_item:
        current_index = int(item[0], 2)
        expression += item[1] * (t * np.abs(current_state[current_index, current_index]) + (1 - t) / 16)
    inequality = GreaterThan(t * left_sum, expression)

    solution = solve_univariate_inequality(inequality, t)
    if not isinstance(solution.args[0].lhs, sympy.core.symbol.Symbol):
        return solution.args[0].lhs
    else:
        return 10


if __name__ == "__main__":
    n_qubit = 4
    num_of_quantum_state = 1000
    total_quantum_state = 10000
    I = np.eye(16) / 16

    exchange_matrix, exchange_matrix_np = get_exchange_matrix(n_qubit)

    entanglement_2_qubit = list()
    entanglement_2_qubit.append(0.5 * (np.outer(np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]).conj())))
    for _ in range(num_of_quantum_state):
        entanglement_2_qubit.append(generate_2_entanglement())

    prod_2_data_list = list()
    pure_state = list()
    for index_1 in range(num_of_quantum_state):
        random_index = np.random.randint(num_of_quantum_state, size=2)
        current_state = np.kron(entanglement_2_qubit[random_index[0]], entanglement_2_qubit[random_index[0]])
        current_state_list = list()
        current_state_list.append(current_state)
        current_state_list.append(exchange_matrix_np[0][2 - 0 - 1] * current_state * exchange_matrix_np[0][2 - 0 - 1])
        current_state_list.append(exchange_matrix_np[0][3 - 0 - 1] * current_state * exchange_matrix_np[0][3 - 0 - 1])

        for current_state in current_state_list:
            new_state = handle_u_matrix(current_state)
            pure_state.append(new_state)

            train_matrix = generate_train_matrix(new_state)
            prod_2_data_list.append(train_matrix)

            p_value = compute_2_prod(new_state)
            if p_value < 1:
                the_matrix = np.float64(p_value) * train_matrix + np.float64(1 - p_value) * I
                prod_2_data_list.append(the_matrix)

    while len(prod_2_data_list) < total_quantum_state:
        a = np.random.randint(1, 30)

        b = np.random.randint(len(pure_state), size=a)

        c = np.random.dirichlet(np.ones(a), size=1)[0]

        convex_state = sum(c[i] * pure_state[b[i]] for i in range(a))
        result, p_value = compute_2_prod(convex_state)
        if p_value < 1:
            train_matrix = generate_train_matrix(convex_state)
            prod_2_data_list.append(train_matrix)

            the_matrix = np.float64(p_value) * train_matrix + np.float64(1 - p_value) * I
            prod_2_data_list.append(the_matrix)

    labels = [2] * len(prod_2_data_list)
    np.save('prod_2_states.npy', prod_2_data_list)
    np.save('prod_2_labels.npy', labels)

    # np.save('prod_2_states_test.npy', prod_2_data_list)
    # np.save('prod_2_labels_test.npy', labels)

