import numpy as np

from Full_Sep_SDP import sigma_x, sigma_y, sigma_z
from bin.data_full_sep import random_point_on_sphere


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
    for i in range(16):
        for j in range(i + 1, 16):
            if (j - i) & (j - i - 1) == 0:
                continue
            for k in range(4):
                m, n = swap_bits(i, j, k)
                # print(m, n, i, j, k)
                if m != i and n != j and np.abs(current_state[i, j]) - 0.5 * (
                        np.abs(current_state[m, m]) + np.abs(current_state[n, n])) > 0:
                    print(m, n, i, j, k)
                    result = True
                    return result

    return result


def handle_state(current_state):
    u_numbers = np.random.uniform(0, 2 * np.pi, 12)
    u_list = get_u_list(u_numbers)
    u_matrix = np.kron(u_list[0], np.kron(u_list[1], np.kron(u_list[2], u_list[3])))
    new_state = u_matrix @ current_state @ np.transpose(u_matrix.conj())
    if checkout_quantum_state(new_state):
        print("True")
    else:
        print("False")
    print()
    # print(new_state[0, 15], new_state[1, 1], new_state[14, 14])
    # print(np.abs(new_state[0, 15]) - 0.5 * (np.abs(new_state[1, 1]) + np.abs(new_state[14, 14])))
    # print(new_state[0, 15], new_state[2, 2], new_state[13, 13])
    # print(np.abs(new_state[0, 15]) - 0.5 * (np.abs(new_state[2, 2]) + np.abs(new_state[13, 13])))
    # print(new_state[0, 15], new_state[4, 4], new_state[11, 11])
    # print(np.abs(new_state[0, 15]) - 0.5 * (np.abs(new_state[4, 4]) + np.abs(new_state[11, 11])))
    # print(new_state[0, 15], new_state[7, 7], new_state[8, 8])
    # print(np.abs(new_state[0, 15]) - 0.5 * (np.abs(new_state[7, 7]) + np.abs(new_state[8, 8])))
    # print()


if __name__ == "__main__":
    n_qubit = 4
    num_of_quantum_state = 50

    part_3_data_list = list()
    for _ in range(num_of_quantum_state):
        x_1, y_1, z_1 = random_point_on_sphere()
        current_qubit_1 = 0.5 * (np.eye(2) + x_1 * sigma_x + y_1 * sigma_y + z_1 * sigma_z)

        x_2, y_2, z_2 = random_point_on_sphere()
        current_qubit_2 = 0.5 * (np.eye(2) + x_2 * sigma_x + y_2 * sigma_y + z_2 * sigma_z)

        current_part = 0.5 * (np.outer(np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]).conj()))

        current_state_list = list()

        current_state_list.append(np.kron(current_qubit_1, np.kron(current_qubit_2, current_part)))
        current_state_list.append(np.kron(current_qubit_1, np.kron(current_part, current_qubit_2)))

        current_state_list.append(np.kron(current_qubit_2, np.kron(current_qubit_1, current_part)))
        current_state_list.append(np.kron(current_qubit_2, np.kron(current_part, current_qubit_1)))

        current_state_list.append(np.kron(current_part, np.kron(current_qubit_2, current_qubit_1)))
        current_state_list.append(np.kron(current_part, np.kron(current_qubit_1, current_qubit_2)))

        for current_state in current_state_list:
            handle_state(current_state)
