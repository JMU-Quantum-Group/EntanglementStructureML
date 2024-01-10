from collections import deque

import numpy as np
import picos as pic
import torch


def bubble_sort_steps(nums):
    steps = list()
    for i in range(len(nums)):
        for j in range(len(nums) - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                steps.append([j, j + 1])
    return steps


def swap_chars(s, i, j):
    lst = list(s)
    lst[i], lst[j] = lst[j], lst[i]
    return ''.join(lst)


class GD_SDP(object):
    def __init__(self, N, n_points, rho, partition_list, r, lr=0.01):
        self.N = N
        self.n_points = n_points
        self.rho = rho
        self.r = r
        self.white_noise = torch.eye(2 ** N) / 2 ** N
        self.beta = torch.flatten(torch.tensor(rho, dtype=torch.complex128) - self.white_noise)
        self.beta_norm = torch.norm(self.beta)
        self.partition_list = partition_list
        self.lr = lr

        self.exchange_matrix = list()
        self.exchange_matrix_np = list()
        for num1 in range(self.N):
            temp_matrix_list = list()
            temp_matrix_list_np = list()
            for num2 in range(num1 + 1, self.N):
                the_matrix = np.zeros([2 ** self.N, 2 ** self.N])
                for number in range(2 ** self.N):
                    number_str = format(number, '0{}b'.format(self.N))
                    number_str = swap_chars(number_str, num1, num2)
                    number_23 = int(number_str, 2)
                    the_matrix[number, number_23] = 1
                temp_matrix_list.append(torch.tensor(the_matrix, dtype=torch.complex128))
                temp_matrix_list_np.append(np.matrix(the_matrix))
            self.exchange_matrix.append(temp_matrix_list)
            self.exchange_matrix_np.append(temp_matrix_list_np)

        self.exchange_list = list()
        self.partition_max_part_info_list = list()
        for partition in self.partition_list:
            concatenated_list = [element for sublist in partition.partition_by_list for element in sublist]
            self.exchange_list.append(bubble_sort_steps(concatenated_list))
            info = dict()
            info['len'] = partition.producible
            for index in range(len(partition.partition_by_list)):
                if len(partition.partition_by_list[index]) == partition.producible:
                    info['part'] = partition.partition_by_list[index]
                    info['index'] = index
            self.partition_max_part_info_list.append(info)

        self.ml_queue = deque(maxlen=1)

    def train(self, epoch):
        weights_normalized = torch.tensor(
            [1 / (self.n_points * len(self.partition_list))] * (self.n_points * len(self.partition_list)))
        opti_list = list()
        L_list = list()
        for index_point in range(self.n_points):
            point_index_list = list()
            for partition in self.partition_list:
                current_L = list()
                for part in partition.partition_by_list:
                    L_temp = torch.randn(2 ** len(part), 2 ** len(part), dtype=torch.complex128)
                    L_temp.requires_grad_(True)
                    current_L.append(L_temp)
                opti_list += current_L
                point_index_list.append(current_L)
            L_list.append(point_index_list)

        optimizer = torch.optim.Adam(opti_list, lr=self.lr)

        param = 1000

        for epoch in range(epoch):
            target = torch.zeros(2 ** self.N, 2 ** self.N, dtype=torch.complex128)

            sdp_torch_list = list()

            for point_index in range(self.n_points):
                point_sdp_torch_list = list()
                for partition_index in range(len(self.partition_list)):
                    partition_sdp_torch_list = list()
                    current_L = L_list[point_index][partition_index]

                    L_0 = torch.matmul(current_L[0], current_L[0].conj().t())
                    L_0 = L_0 / L_0.trace()
                    L_1 = torch.matmul(current_L[1], current_L[1].conj().t())
                    L_1 = L_1 / L_1.trace()

                    partition_sdp_torch_list.append(L_0)
                    partition_sdp_torch_list.append(L_1)

                    current_target = torch.kron(L_0, L_1)
                    for L_index in range(2, len(current_L)):
                        L = torch.matmul(current_L[L_index], current_L[L_index].conj().t())
                        L = L / L.trace()
                        partition_sdp_torch_list.append(L)
                        current_target = torch.kron(current_target, L)

                    point_sdp_torch_list.append(partition_sdp_torch_list)

                    for exchange_pair in self.exchange_list[partition_index]:
                        current_target = torch.matmul(torch.matmul(
                            self.exchange_matrix[exchange_pair[0]][exchange_pair[1] - exchange_pair[0] - 1],
                            current_target), self.exchange_matrix[exchange_pair[0]][
                            exchange_pair[1] - exchange_pair[0] - 1])

                    current_target = weights_normalized[
                                         point_index * len(self.partition_list) + partition_index] * current_target
                    target += current_target

                sdp_torch_list.append(point_sdp_torch_list)

            alpha = torch.flatten(target - self.white_noise)
            scalar = torch.norm(torch.matmul(self.beta.T, alpha)) / (self.beta_norm ** 2)

            matrix_1 = target - torch.tensor(self.rho, dtype=torch.complex128)
            target_loss = param * torch.abs(torch.sqrt(torch.trace(torch.matmul(matrix_1.conj().t(), matrix_1))))
            # target_loss = param * torch.norm(target - torch.tensor(self.rho, dtype=torch.complex128))

            matrix_2 = target - (
                        scalar * torch.tensor(self.rho, dtype=torch.complex128) + (1 - scalar) * self.white_noise)
            real_distance = param * torch.abs(torch.sqrt(torch.trace(torch.matmul(matrix_2.conj().t(), matrix_2))))
            # real_distance = param * torch.norm(
            #     target - (scalar * torch.tensor(self.rho, dtype=torch.complex128) + (1 - scalar) * self.white_noise))

            if real_distance > self.r:
                optimizer.zero_grad()
                real_distance.backward()
                optimizer.step()
                print(
                    f'Epoch {epoch} distance: Scalar = {scalar.item()}, Scalar Loss = {target_loss.item()}, real Distance Loss = {real_distance.item()}')
            else:
                optimizer.zero_grad()
                target_loss.backward()
                optimizer.step()
                print(
                    f'Epoch {epoch} scalar: Scalar = {scalar.item()}, Scalar Loss = {target_loss.item()}, real Distance Loss = {real_distance.item()}')

            result = list()
            for point_index in range(self.n_points):
                temp_point_list = list()
                for partition_index in range(len(self.partition_list)):
                    current_L = list()
                    info = self.partition_max_part_info_list[partition_index]
                    for part_index in range(len(self.partition_list[partition_index].partition_by_list)):
                        if part_index == info['index']:
                            continue
                        else:
                            current_L.append(sdp_torch_list[point_index][partition_index][part_index].detach().numpy())
                    temp_point_list.append(current_L)
                result.append(temp_point_list)

            if real_distance < self.r:
                self.ml_queue.append(result)

    def sdp(self):
        x_list = self.ml_queue[0]

        prob = pic.Problem()
        rho_list = list()
        for i in range(self.n_points):
            rho_list.append([])

        p = pic.RealVariable("p", 1)

        for i in range(self.n_points):
            for j in range(len(self.partition_list)):
                info = self.partition_max_part_info_list[j]
                sdp_rho = pic.HermitianVariable('rho_' + str(i) + '_' + str(j), 2 ** info['len'])
                rho_list[i].append(sdp_rho)
                prob.add_constraint(sdp_rho >> 0)

        rho_next = None
        for point_index in range(self.n_points):
            for partition_index in range(len(self.partition_list)):
                info = self.partition_max_part_info_list[partition_index]
                rho_left = None
                rho_right = None
                x_list_index = 0
                for part_index in range(len(self.partition_list[partition_index].partition_by_list)):
                    if part_index < info['index']:
                        if rho_left is None:
                            rho_left = x_list[point_index][partition_index][x_list_index]
                            x_list_index += 1
                        else:
                            rho_left = np.kron(rho_left, x_list[point_index][partition_index][x_list_index])
                    elif part_index > info['index']:
                        if rho_right is None:
                            rho_right = x_list[point_index][partition_index][x_list_index]
                            x_list_index += 1
                        else:
                            rho_right = np.kron(rho_right, x_list[point_index][partition_index][x_list_index])
                    else:
                        continue

                if rho_left is None:
                    current_rho = rho_list[point_index][partition_index]
                else:
                    current_rho = rho_left @ rho_list[point_index][partition_index]

                if rho_right is not None:
                    current_rho = current_rho @ rho_right

                for exchange_pair in self.exchange_list[partition_index]:
                    current_rho = self.exchange_matrix_np[exchange_pair[0]][
                                      exchange_pair[1] - exchange_pair[0] - 1] * current_rho * \
                                  self.exchange_matrix_np[exchange_pair[0]][exchange_pair[1] - exchange_pair[0] - 1]

                if rho_next is None:
                    rho_next = current_rho
                else:
                    rho_next += current_rho

        prob.add_constraint(rho_next == (p * self.rho + ((1 - p) / (2 ** self.N)) * np.eye(2 ** self.N)))
        prob.set_objective("max", p)
        prob.solve(solver="mosek", primals=True)
        return p.value
