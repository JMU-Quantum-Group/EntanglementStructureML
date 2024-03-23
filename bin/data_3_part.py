import math
import random

import numpy as np
import torch
from sympy import symbols, GreaterThan, solve_univariate_inequality

from Full_Sep_SDP import sigma_x, sigma_z, sigma_y
from bin.data_full_sep import random_point_on_sphere, generate_train_matrix


def checkout_2_entanglement_state(rho):
    rho_ppt = [[rho[0, 0], rho[0, 1], rho[2, 0], rho[2, 1]],
               [rho[1, 0], rho[1, 1], rho[3, 0], rho[3, 1]],
               [rho[0, 2], rho[0, 3], rho[2, 2], rho[2, 3]],
               [rho[1, 2], rho[1, 3], rho[3, 2], rho[3, 3]]]

    if np.all(np.linalg.eigvals(np.array(rho_ppt)) >= 0):
        return False
    else:
        return True


def swap_chars(s, i, j):
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)


def generate_unitary(alpha, beta, theta):
    # 计算复数的指数
    exp_i_alpha = np.exp(1j * alpha)
    exp_i_beta = np.exp(1j * beta)

    # 计算余弦和正弦
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 生成酉矩阵
    u = np.array([
        [exp_i_alpha * cos_theta, exp_i_beta * sin_theta],
        [-np.conj(exp_i_beta) * sin_theta, np.conj(exp_i_alpha) * cos_theta]
    ])

    return u


def get_u_list(u_numbers):
    u_list = list()
    for i in range(4):
        u_list.append(generate_unitary(u_numbers[i * 3], u_numbers[i * 3 + 1], u_numbers[i * 3 + 2]))
    return u_list


def swap_bits(a, b, i):
    # 获取 a 和 b 的第 i 位
    a_i = (a >> i) & 1
    b_i = (b >> i) & 1

    # 如果 a 和 b 的第 i 位不同，我们需要交换它们
    if a_i != b_i:
        # 创建一个只有第 i 位为 1 的掩码
        mask = 1 << i

        # 使用异或操作交换 a 和 b 的第 i 位
        a ^= mask
        b ^= mask

    return a, b


def checkout_quantum_state(current_state):
    result = False
    p_value = -1
    for i in range(16):
        for j in range(i + 1, 16):
            if (j - i) & (j - i - 1) == 0:
                continue
            for k in range(4):
                m, n = swap_bits(i, j, k)
                if m != i and n != j and np.abs(current_state[i, j]) - 0.5 * (
                        np.abs(current_state[m, m]) + np.abs(current_state[n, n])) > 0:
                    result = True
                    t = symbols('t')
                    inequality = GreaterThan(t * np.abs(current_state[i, j]), 0.5 * (
                            t * (np.abs(current_state[m, m]) + np.abs(current_state[n, n])) + (1 - t) / 16))
                    solution = solve_univariate_inequality(inequality, t)
                    if p_value == -1:
                        p_value = solution.args[0].args[0]
                    elif solution.args[0].args[0] < p_value:
                        p_value = solution.args[0].args[0]

    return result, p_value


def handle_u_matrix(current_state):
    u_numbers = np.random.uniform(0, 2 * np.pi, 12)
    u_list = get_u_list(u_numbers)
    u_matrix = np.kron(u_list[0], np.kron(u_list[1], np.kron(u_list[2], u_list[3])))
    new_state = u_matrix @ current_state @ np.transpose(u_matrix.conj())
    return new_state


def get_exchange_matrix(n_qubit):
    exchange_matrix = list()
    exchange_matrix_np = list()
    for num1 in range(n_qubit):
        temp_matrix_list = list()
        temp_matrix_list_np = list()
        for num2 in range(num1 + 1, n_qubit):
            the_matrix = np.zeros([2 ** n_qubit, 2 ** n_qubit])
            for number in range(2 ** n_qubit):
                number_str = format(number, '0{}b'.format(n_qubit))
                number_str = swap_chars(number_str, num1, num2)
                number_23 = int(number_str, 2)
                the_matrix[number, number_23] = 1
            temp_matrix_list.append(torch.tensor(the_matrix, dtype=torch.complex128))
            temp_matrix_list_np.append(np.matrix(the_matrix))
        exchange_matrix.append(temp_matrix_list)
        exchange_matrix_np.append(temp_matrix_list_np)
    return exchange_matrix, exchange_matrix_np


def generate_2_entanglement():
    r1 = random.random()
    r2 = random.random()
    r3 = random.random()
    r4 = random.random()

    s = r1 ** 2 + r2 ** 2 + r3 ** 2 + r4 ** 2
    scale = math.sqrt(s)
    a = r1 / scale
    b = r2 / scale
    c = r3 / scale
    d = r4 / scale

    current_part = np.outer(np.array([a + b * 1j, 0, 0, c + d * 1j]),
                            np.array([a + b * 1j, 0, 0, c + d * 1j]).conj())
    return current_part


if __name__ == "__main__":
    n_qubit = 4
    num_of_quantum_state = 1000
    I = np.eye(16) / 16

    exchange_matrix, exchange_matrix_np = get_exchange_matrix(n_qubit)
    entanglement_2_qubit = list()
    entanglement_2_qubit.append(0.5 * (np.outer(np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]).conj())))
    for _ in range(num_of_quantum_state):
        entanglement_2_qubit.append(generate_2_entanglement())

    part_3_data_list = list()
    pure_state = list()
    for index_1 in range(num_of_quantum_state):
        x_1, y_1, z_1 = random_point_on_sphere()
        current_qubit_1 = 0.5 * (np.eye(2) + x_1 * sigma_x + y_1 * sigma_y + z_1 * sigma_z)

        x_2, y_2, z_2 = random_point_on_sphere()
        current_qubit_2 = 0.5 * (np.eye(2) + x_2 * sigma_x + y_2 * sigma_y + z_2 * sigma_z)

        current_state_list = list()

        current_state_list.append(
            np.kron(current_qubit_1, np.kron(current_qubit_2, entanglement_2_qubit[index_1])))  # 1|2|3,4
        current_state_list.append(exchange_matrix_np[0][2 - 0 - 1] * np.kron(current_qubit_1, np.kron(current_qubit_2,
                                                                                                      entanglement_2_qubit[
                                                                                                          index_1])) *
                                  exchange_matrix_np[0][2 - 0 - 1])
        current_state_list.append(exchange_matrix_np[1][2 - 1 - 1] * np.kron(current_qubit_1, np.kron(current_qubit_2,
                                                                                                      entanglement_2_qubit[
                                                                                                          index_1])) *
                                  exchange_matrix_np[1][2 - 1 - 1])
        current_state_list.append(exchange_matrix_np[0][3 - 0 - 1] * np.kron(current_qubit_1, np.kron(current_qubit_2,
                                                                                                      entanglement_2_qubit[
                                                                                                          index_1])) *
                                  exchange_matrix_np[0][3 - 0 - 1])
        current_state_list.append(exchange_matrix_np[1][3 - 1 - 1] * np.kron(current_qubit_1, np.kron(current_qubit_2,
                                                                                                      entanglement_2_qubit[
                                                                                                          index_1])) *
                                  exchange_matrix_np[1][3 - 1 - 1])
        current_state_list.append(np.kron(entanglement_2_qubit[index_1], np.kron(current_qubit_1, current_qubit_2)))

        for current_state in current_state_list:
            new_state = handle_u_matrix(current_state)
            pure_state.append(new_state)

            result, p_value = checkout_quantum_state(new_state)
            train_matrix = generate_train_matrix(new_state)
            part_3_data_list.append(train_matrix)
            if result:
                the_matrix = np.float64(p_value) * train_matrix + np.float64(1 - p_value) * I
                part_3_data_list.append(the_matrix)

        print("index:", index_1)

    convex_state_list = list()
    for _ in range(num_of_quantum_state):
        a = np.random.randint(1, 50)

        b = np.random.randint(len(pure_state), size=a)

        c = np.random.dirichlet(np.ones(a), size=1)[0]

        convex_state = sum(c[i] * pure_state[b[i]] for i in range(a))
        result, p_value = checkout_quantum_state(convex_state)
        if result:
            train_matrix = generate_train_matrix(convex_state)
            part_3_data_list.append(train_matrix)

            I = np.eye(16) / 16
            the_matrix = np.float64(p_value) * train_matrix + np.float64(1 - p_value) * I
            part_3_data_list.append(the_matrix)
        print("convex index:", _)

    labels = [1] * len(part_3_data_list)

    np.save('part_3_states.npy', part_3_data_list)
    np.save('part_3_labels.npy', labels)
    #
    # np.save('part_3_states_test.npy', part_3_data_list)
    # np.save('part_3_labels_test.npy', labels)
