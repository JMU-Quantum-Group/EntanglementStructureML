import numpy as np

from Full_Sep_SDP import sigma_x, sigma_y, sigma_z


def random_state(n_qubit):
    n = 2 ** n_qubit
    real_current_matrix = np.random.rand(n, n)
    real_current_matrix = np.matrix(real_current_matrix)

    imaginary_current_matrix = np.random.rand(n, n)
    imaginary_current_matrix = np.matrix(imaginary_current_matrix)

    current_matrix = real_current_matrix + 1j * imaginary_current_matrix

    state = current_matrix * np.transpose(np.conj(current_matrix))
    return np.array(state / state.trace())


def random_point_on_sphere():
    theta = 2 * np.pi * np.random.random()
    phi = np.arccos(2 * np.random.random() - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)  # 计算z坐标
    return x, y, z


if __name__ == "__main__":
    n_qubit = 4
    num_of_quantum_state = 500

    print("----------- full sep -------------")
    full_sep_state_list = list()
    for _ in range(num_of_quantum_state):
        qubit_list = list()
        for i in range(n_qubit):
            x, y, z = random_point_on_sphere()
            current_qubit = 0.5 * (np.eye(2) + x * sigma_x + y * sigma_y + z * sigma_z)
            qubit_list.append(current_qubit)
        current_state = np.kron(qubit_list[0], np.kron(qubit_list[1], np.kron(qubit_list[2], qubit_list[3])))

        upper_triangular_real = np.triu(np.real(current_state))
        lower_triangular_imag = np.tril(np.imag(current_state))
        result = np.array(upper_triangular_real + lower_triangular_imag)

        full_sep_state_list.append(result)

    for _ in range(num_of_quantum_state):
        # 生成一个1到15的随机数设为a
        a = np.random.randint(1, num_of_quantum_state)

        # 生成一个长度为a的list，标记为b，里面每个数是0到499
        b = np.random.randint(num_of_quantum_state, size=a)

        # 生成一个长度为a的list，标记为c，每个数是0到1的小数，然后总和为1
        c = np.random.dirichlet(np.ones(a), size=1)[0]

        result = sum(c[i] * full_sep_state_list[b[i]] for i in range(a))

        full_sep_state_list.append(result)

    # not_full_sep_state_list = list()
    # for _ in range(2 * num_of_quantum_state):
    #     current_state = random_state(n_qubit)
    #     not_full_sep_state_list.append(current_state)
    #
    # matrices = full_sep_state_list + not_full_sep_state_list

    labels = [0] * (2 * num_of_quantum_state)

    # np.save('full_sep_states.npy', full_sep_state_list)
    # np.save('full_sep_labels.npy', labels)
    np.save('full_sep_states_test.npy', full_sep_state_list)
    np.save('full_sep_labels_test.npy', labels)

