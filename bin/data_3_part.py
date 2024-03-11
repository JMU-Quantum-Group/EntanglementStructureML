import numpy as np
import torch
from picos import Constant
from sympy import symbols, GreaterThan, solve_univariate_inequality

from Full_Sep_SDP import sigma_x, sigma_z, sigma_y
from bin.data_full_sep import random_point_on_sphere


def checkout_2_entanglement_state(rho):
    rho_picos = Constant("rho_picos", rho, (4, 4))
    rho_pt = rho_picos.partial_transpose(0)

    print(np.array(rho_pt.value[0]))

    if np.all(np.linalg.eigvals(np.array(rho_pt.value)) >= 0):
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


def handle_state(current_state, part_3_data_list):
    upper_triangular_real = np.triu(np.real(current_state))
    lower_triangular_imag = np.tril(np.imag(current_state))
    handle_matrix = np.array(upper_triangular_real + lower_triangular_imag)
    part_3_data_list.append(handle_matrix)

    u_numbers = np.random.uniform(0, 2 * np.pi, 12)
    u_list = get_u_list(u_numbers)
    u_matrix = np.kron(u_list[0], np.kron(u_list[1], np.kron(u_list[2], u_list[3])))
    new_state = u_matrix @ current_state @ np.transpose(u_matrix.conj())
    result, p_value = checkout_quantum_state(new_state)
    if result:
        I = np.eye(16) / 16
        the_matrix = np.float64(p_value) * handle_matrix + np.float64(1 - p_value) * I
        part_3_data_list.append(the_matrix)


if __name__ == "__main__":
    n_qubit = 4
    num_of_quantum_state = 10

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

    part_3_data_list = list()
    for _ in range(num_of_quantum_state):
        x_1, y_1, z_1 = random_point_on_sphere()
        current_qubit_1 = 0.5 * (np.eye(2) + x_1 * sigma_x + y_1 * sigma_y + z_1 * sigma_z)

        x_2, y_2, z_2 = random_point_on_sphere()
        current_qubit_2 = 0.5 * (np.eye(2) + x_2 * sigma_x + y_2 * sigma_y + z_2 * sigma_z)

        current_part = 0.5 * (np.outer(np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]).conj()))

        # print(checkout_2_entanglement_state(current_part))

        current_state_list = list()

        # 1,4  1,3, 2,4

        current_state_list.append(np.kron(current_qubit_1, np.kron(current_qubit_2, current_part)))  # 1|2|3,4
        current_state_list.append(
            exchange_matrix_np[0][3 - 0 - 1] * np.kron(current_qubit_1, np.kron(current_qubit_2, current_part)) *
            exchange_matrix_np[0][3 - 0 - 1])
        current_state_list.append(
            exchange_matrix_np[0][2 - 0 - 1] * np.kron(current_qubit_1, np.kron(current_qubit_2, current_part)) *
            exchange_matrix_np[0][2 - 0 - 1])
        current_state_list.append(
            exchange_matrix_np[1][2 - 1 - 1] * np.kron(current_qubit_1, np.kron(current_qubit_2, current_part)) *
            exchange_matrix_np[1][2 - 1 - 1])

        current_state_list.append(np.kron(current_qubit_1, np.kron(current_part, current_qubit_2)))  # 1|2,3|4
        current_state_list.append(
            exchange_matrix_np[0][1 - 0 - 1] * np.kron(current_qubit_1, np.kron(current_part, current_qubit_2)) *
            exchange_matrix_np[0][1 - 0 - 1])
        current_state_list.append(
            exchange_matrix_np[1][3 - 1 - 1] * np.kron(current_qubit_1, np.kron(current_part, current_qubit_2)) *
            exchange_matrix_np[1][3 - 1 - 1])

        current_state_list.append(np.kron(current_qubit_2, np.kron(current_qubit_1, current_part)))
        current_state_list.append(
            exchange_matrix_np[0][3 - 0 - 1] * np.kron(current_qubit_2, np.kron(current_qubit_1, current_part)) *
            exchange_matrix_np[0][3 - 0 - 1])
        current_state_list.append(
            exchange_matrix_np[0][2 - 0 - 1] * np.kron(current_qubit_2, np.kron(current_qubit_1, current_part)) *
            exchange_matrix_np[0][2 - 0 - 1])
        current_state_list.append(
            exchange_matrix_np[1][2 - 1 - 1] * np.kron(current_qubit_2, np.kron(current_qubit_1, current_part)) *
            exchange_matrix_np[1][2 - 1 - 1])

        current_state_list.append(np.kron(current_qubit_2, np.kron(current_part, current_qubit_1)))
        current_state_list.append(
            exchange_matrix_np[0][1 - 0 - 1] * np.kron(current_qubit_2, np.kron(current_part, current_qubit_1)) *
            exchange_matrix_np[0][1 - 0 - 1])
        current_state_list.append(
            exchange_matrix_np[1][3 - 1 - 1] * np.kron(current_qubit_2, np.kron(current_part, current_qubit_1)) *
            exchange_matrix_np[1][3 - 1 - 1])

        current_state_list.append(np.kron(current_part, np.kron(current_qubit_2, current_qubit_1)))
        current_state_list.append(
            exchange_matrix_np[0][3 - 0 - 1] * np.kron(current_part, np.kron(current_qubit_2, current_qubit_1)) *
            exchange_matrix_np[0][3 - 0 - 1])
        current_state_list.append(
            exchange_matrix_np[1][3 - 1 - 1] * np.kron(current_part, np.kron(current_qubit_2, current_qubit_1)) *
            exchange_matrix_np[1][3 - 1 - 1])
        current_state_list.append(
            exchange_matrix_np[1][2 - 1 - 1] * np.kron(current_part, np.kron(current_qubit_2, current_qubit_1)) *
            exchange_matrix_np[1][2 - 1 - 1])

        current_state_list.append(np.kron(current_part, np.kron(current_qubit_1, current_qubit_2)))
        current_state_list.append(
            exchange_matrix_np[0][3 - 0 - 1] * np.kron(current_part, np.kron(current_qubit_1, current_qubit_2)) *
            exchange_matrix_np[0][3 - 0 - 1])
        current_state_list.append(
            exchange_matrix_np[1][3 - 1 - 1] * np.kron(current_part, np.kron(current_qubit_1, current_qubit_2)) *
            exchange_matrix_np[1][3 - 1 - 1])
        current_state_list.append(
            exchange_matrix_np[1][2 - 1 - 1] * np.kron(current_part, np.kron(current_qubit_1, current_qubit_2)) *
            exchange_matrix_np[1][2 - 1 - 1])

        for current_state in current_state_list:
            handle_state(current_state, part_3_data_list)

        print("index:", _)

    labels = [1] * len(part_3_data_list)

    # np.save('part_3_states.npy', part_3_data_list)
    # np.save('part_3_labels.npy', labels)

    np.save('part_3_states_test.npy', part_3_data_list)
    np.save('part_3_labels_test.npy', labels)
