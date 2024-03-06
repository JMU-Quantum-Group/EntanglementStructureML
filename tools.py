import copy
import itertools
import logging
import random
from fractions import Fraction
from itertools import product

import pandas as pd

from dancing_links import DancingLinks
from partition_tools import generate_k_stretchable_partitions, generate_k_producible_partitions, \
    generate_k_partitionable_partitions

logging.basicConfig(filename='app.log', filemode='a', format='%(message)s', level=logging.INFO)


def compute_all_stretchable(P, N):
    for k in range(1 - N, N - 1):
        logging.info(str(N) + "partite" + str(k) + "-str:")
        compute(P, generate_k_stretchable_partitions(N, k))
        logging.info("----------------------------------------------------------------------")


def compute_all_producible(P, N):
    for k in range(1, N):
        logging.info(str(N) + "partite" + str(k) + "-prod:")
        compute(P, generate_k_producible_partitions(N, k))
        logging.info("----------------------------------------------------------------------")


def compute_all_partitionable(P, N):
    for k in range(N - 1, 1, -1):
        logging.info(str(N) + "partite" + str(k) + "-part:")
        compute(P, generate_k_partitionable_partitions(N, k))
        logging.info("----------------------------------------------------------------------")


def recursion_generate_min_union(M, current_index, current_result, min_count):
    result_list = list()
    sorted_list = sorted(M[current_index], key=lambda m: len(set(m).union(current_result)))

    if current_index + 1 == len(M):
        for temp_result in sorted_list:
            if len(set(temp_result).union(current_result)) <= min_count:
                min_count = len(set(temp_result).union(current_result))
                result_list.append([temp_result])
            else:
                break
    else:
        for temp_result in sorted_list:
            if len(set(temp_result).union(current_result)) <= min_count:
                recursion_result_list = recursion_generate_min_union(M, current_index + 1,
                                                                     current_result.union(set(temp_result)), min_count)
                if len(recursion_result_list) > 0:
                    temp_recursion = set()
                    for m in recursion_result_list[0]:
                        temp_recursion = temp_recursion.union(set(m))

                    if len(temp_recursion.union(current_result).union(temp_result)) < min_count:
                        min_count = len(temp_recursion.union(current_result).union(temp_result))
                        result_list.clear()
                        result_list.extend(
                            [[temp_result] + recursion_result for recursion_result in recursion_result_list])
                    elif len(temp_recursion.union(current_result).union(temp_result)) == min_count:
                        result_list.extend(
                            [[temp_result] + recursion_result for recursion_result in recursion_result_list])

    return result_list


def m_n_part(m, n, part):
    result = str()
    start_index = 0
    for index in part:
        result += m[start_index: index - 1]
        result += n[index - 1]
        start_index = index
    result += m[start_index:]
    return result


def frac_to_latex(frac):
    if frac.denominator == 1:
        return str(frac.numerator)
    else:
        return r'\frac{{{}}}{{{}}}'.format(frac.numerator, frac.denominator)


def recursion_mini_cover_list(pair_set_cover_dict, need_cover_set, mini_count):
    cover_list = list()
    sorted_keys_dict = dict()
    for key in pair_set_cover_dict:
        if pair_set_cover_dict[key] & need_cover_set == need_cover_set:
            cover_list.append([key])
        elif len(pair_set_cover_dict[key] & need_cover_set) > 0:
            sorted_keys_dict[key] = len(pair_set_cover_dict[key] & need_cover_set)

    if len(cover_list) > 0 or mini_count == 1 or len(need_cover_set) > sum(sorted_keys_dict.values()):
        return cover_list

    sorted_keys = [k for k, v in sorted(sorted_keys_dict.items(), key=lambda item: item[1], reverse=True)]
    temp_pair_set_cover_dict = copy.deepcopy(pair_set_cover_dict)
    for key in sorted_keys:
        next_need_cover_set = need_cover_set - pair_set_cover_dict[key]
        del temp_pair_set_cover_dict[key]
        current_cover_list = recursion_mini_cover_list(temp_pair_set_cover_dict, next_need_cover_set, mini_count - 1)
        if len(current_cover_list) > 0:
            if len(current_cover_list[0]) + 1 < mini_count:
                mini_count = len(current_cover_list[0]) + 1
                cover_list.clear()
                cover_list.extend([cover_item + [key] for cover_item in current_cover_list])
            elif len(current_cover_list[0]) + 1 == mini_count:
                cover_list.extend([cover_item + [key] for cover_item in current_cover_list])

    return cover_list


def compute(P, Gamma, greedy=False, dim=2):
    qs_candidate_list = list()

    # compute the candidate matrix
    for partition in Gamma:
        partition_candidate_list = dict()
        partition_candidate_list['Partitions'] = partition.partition_by_str
        for quantum_pair in P:
            quantum_state_set = set()
            for part in partition.partition_by_list:
                quantum_state_set.add(".".join(sorted([m_n_part(quantum_pair[0], quantum_pair[1], part),
                                                       m_n_part(quantum_pair[1], quantum_pair[0], part)])))
            partition_candidate_list[".".join(sorted(quantum_pair))] = list(quantum_state_set)
        qs_candidate_list.append(partition_candidate_list)

    # select the only one choice
    qs_candidate = pd.DataFrame(qs_candidate_list)
    column_pair_dict = dict()
    for column in qs_candidate.columns:
        if column == "Partitions":
            continue
        column_pair_set = set()
        for pair_list in qs_candidate[column]:
            if len(pair_list) == 1:
                column_pair_set.add(pair_list[0])
            if len(pair_list) == 2 and column in pair_list:
                column_pair_set.add(pair_list[0] if column == pair_list[1] else pair_list[1])
        column_pair_dict[column] = column_pair_set

    # 计算 P 的系数
    union_max = dict()
    for index, line in qs_candidate.iterrows():
        union_add = dict()
        for column in qs_candidate.columns:
            if column == "Partitions":
                continue
            if len(line[column]) == 1 and line[column][0] == column:
                quantum = column.split(".")
                if quantum[0] not in union_add:
                    union_add[quantum[0]] = 0
                if quantum[1] not in union_add:
                    union_add[quantum[1]] = 0
                union_add[quantum[0]] += Fraction(1, 2)
                union_add[quantum[1]] += Fraction(1, 2)
        for key in union_add:
            if key not in union_max:
                union_max[key] = 0
            if union_max[key] < union_add[key]:
                union_max[key] = union_add[key]

    union_pair_list = list()
    columns_list = list()
    # 选取最小集合
    for column in qs_candidate.columns:
        if column == "Partitions":
            continue

        union_pair_set = set()
        need_to_cover_list = list()
        for pair_list in qs_candidate[column]:
            pair_set = set(pair_list)
            if column in pair_set:
                pair_set.remove(column)
            if len(pair_set & column_pair_dict[column]) == 0:
                union_pair_set = union_pair_set.union(pair_set)
                need_to_cover_list.append(pair_set)

        if len(union_pair_set) != 0:
            pair_set_cover_dict = dict()
            for pair in union_pair_set:
                pair_set_cover_dict[pair] = set()
                for index in range(len(need_to_cover_list)):
                    if pair in need_to_cover_list[index]:
                        pair_set_cover_dict[pair].add(index)

            cover_list = recursion_mini_cover_list(pair_set_cover_dict, set(range(len(need_to_cover_list))),
                                                   float('inf'))
            union_pair_list.append([list(column_pair_dict[column]) + cover_item for cover_item in cover_list])
            columns_list.append(column)
        else:
            union_pair_list.append([list(column_pair_dict[column])])
            columns_list.append(column)

    min_union_pair_list = recursion_generate_min_union(union_pair_list, 0, set(), float('inf'))
    min_sum_value = float('inf')
    best_choice = None
    best_qs_value_dict = None
    for min_union_pair in min_union_pair_list:
        temp_candidate = qs_candidate.copy()
        current_choice, current_sum_value, current_best_qs_value_dict = select_best_choice(min_union_pair,
                                                                                           temp_candidate, columns_list,
                                                                                           union_max, greedy)
        if current_sum_value < min_sum_value:
            best_choice = current_choice
            min_sum_value = current_sum_value
            best_qs_value_dict = current_best_qs_value_dict

    # latex best choice
    best_choice = pd.DataFrame(best_choice)
    logging.info(best_choice.to_csv())
    logging.info(min_sum_value)

    str_list = list()
    sorted_items = sorted(best_qs_value_dict.items(), key=lambda x: x[1], reverse=True)
    # 按顺序输出值和对应的键
    for value, group in itertools.groupby(sorted_items, key=lambda x: x[1]):
        keys = sorted([item[0] for item in group])
        if len(keys) > 1:
            str_list.append(frac_to_latex(value) + " \\times (" + " + ".join(
                ["\\rho_{" + str(int(key, dim) + 1) + "," + str(int(key, dim) + 1) + "}" for key in keys]) + ")")
        else:
            str_list.append(frac_to_latex(value) + " \\times " + " + ".join(
                ["\\rho_{" + str(int(key, dim) + 1) + "," + str(int(key, dim) + 1) + "}" for key in keys]))

    logging.info(
        " + ".join(["|\\rho_{" + str(int(item[0], dim) + 1) + "," + str(int(item[1], dim) + 1) + "}|" for item in
                    P]) + " \\le " + " + ".join(str_list))


def select_best_choice(min_union_pair, qs_candidate, columns_list, union_max, greedy=False):
    partition_list = qs_candidate['Partitions']
    qs_candidate = qs_candidate.drop('Partitions', axis=1)
    for col in qs_candidate.columns:
        # 获取限定范围
        r = min_union_pair[columns_list.index(col)]
        # 应用函数，删除不在限定范围内的值
        qs_candidate[col] = qs_candidate[col].apply(lambda x: [i for i in x if i in r])

    best_choices_list = list()
    best_choice_limit = 30
    for a in range(len(qs_candidate.values)):
        temp_union_max = union_max.copy()
        no_change_column_list = list()

        for key in temp_union_max:
            temp_union_max[key] = int(temp_union_max[key] / Fraction(1, 2))

        for temp_index in range(len(qs_candidate.values[a])):
            if len(qs_candidate.values[a][temp_index]) == 1 and qs_candidate.columns[temp_index] in \
                    qs_candidate.values[a][temp_index]:
                key1, key2 = qs_candidate.columns[temp_index].split('.')
                temp_union_max[key1] -= 1
                temp_union_max[key2] -= 1

        no_change_column_dict = dict()
        for temp_index in range(len(qs_candidate.values[a])):
            if len(qs_candidate.values[a][temp_index]) > 1:
                key1, key2 = qs_candidate.columns[temp_index].split('.')
                if key1 in temp_union_max and key2 in temp_union_max and temp_union_max[key1] > 0 and temp_union_max[
                    key2] > 0:
                    no_change_column_list.append([key1, key2])
                    no_change_column_dict[key1] = no_change_column_dict.get(key1, 0) + 1
                    no_change_column_dict[key2] = no_change_column_dict.get(key2, 0) + 1

        remove_key_list = list()
        for key in temp_union_max:
            if temp_union_max[key] == 0 or key not in no_change_column_dict:
                remove_key_list.append(key)
            elif temp_union_max[key] > no_change_column_dict[key]:
                temp_union_max[key] = no_change_column_dict[key]
        for key in remove_key_list:
            del temp_union_max[key]

        not_handled = True
        current_qs_candidate_list = list()
        if sum(temp_union_max.values()) > 0 and len(no_change_column_list) > 0:
            no_change_list = no_change_candidate(no_change_column_list, temp_union_max, greedy)
            if len(no_change_list) > 0:
                not_handled = False
                for no_change in no_change_list:
                    current_qs_candidate = copy.deepcopy(qs_candidate.values[a])
                    for temp_index in range(len(current_qs_candidate)):
                        if qs_candidate.columns[temp_index] in no_change:
                            current_qs_candidate[temp_index] = [qs_candidate.columns[temp_index]]
                        elif len(current_qs_candidate[temp_index]) > 1 and qs_candidate.columns[temp_index] in \
                                current_qs_candidate[temp_index]:
                            current_qs_candidate[temp_index].remove(qs_candidate.columns[temp_index])
                    current_qs_candidate_list.append(current_qs_candidate)

        if not_handled:
            current_qs_candidate = copy.deepcopy(qs_candidate.values[a])
            for temp_index in range(len(current_qs_candidate)):
                if len(current_qs_candidate[temp_index]) > 1 and qs_candidate.columns[temp_index] in \
                        current_qs_candidate[temp_index]:
                    current_qs_candidate[temp_index].remove(qs_candidate.columns[temp_index])
            current_qs_candidate_list.append(current_qs_candidate)

        current_best_choices = list()
        for current_qs_candidate in current_qs_candidate_list:
            choices = product(*current_qs_candidate)

            min_value = float('inf')
            max_count = 0
            for choice in choices:
                choice_list = list(choice)
                qs_result = handle_choice(choice_list)

                value = max(qs_result.values())
                count = len(qs_result.keys())
                if (value < min_value and count > max_count) or (count > max_count and value == min_value) or (
                        value < min_value and count == max_count):
                    current_best_choices.clear()
                    current_best_choices.append(choice_list)
                    min_value = value
                    max_count = count
                elif value == min_value and count == max_count:
                    current_best_choices.append(choice_list)
                    if len(current_best_choices) == best_choice_limit:
                        break
        best_choices_list.append(current_best_choices)

    # select 10 choice randomly, find the best one
    result_choice = None
    min_sum_value = float('inf')
    best_qs_value_dict = None
    count = 0
    while count < 10:
        qs_value_dict = dict()
        current_choice = dict()
        current_choice['Partitions'] = partition_list
        for col in columns_list:
            current_choice[col] = list()
        for best_choice in best_choices_list:
            choice = best_choice[random.randint(0, len(best_choice) - 1)]
            qs_value_dict = merge_dicts(qs_value_dict, handle_choice(choice))
            for i in range(len(columns_list)):
                current_choice[columns_list[i]].append(choice[i])
        choice_coefficients_sum = sum(value for key, value in qs_value_dict.items() if key not in union_max.keys())

        if choice_coefficients_sum < min_sum_value:
            min_sum_value = choice_coefficients_sum
            result_choice = current_choice
            best_qs_value_dict = qs_value_dict
        count += 1

    return result_choice, min_sum_value, best_qs_value_dict


def no_change_candidate(column_list, union_max_count, greedy):
    target_union_max_count_list = list()
    # use dancing links
    if sum(union_max_count.values()) % 2 != 0:
        for key in union_max_count:
            current_union_max_count = copy.deepcopy(union_max_count)
            current_union_max_count[key] -= 1
            target_union_max_count_list.append(current_union_max_count)
    else:
        target_union_max_count_list.append(union_max_count)

    no_change_list = list()
    for temp_union_max_count in target_union_max_count_list:
        current_index = 0
        for key in sorted(temp_union_max_count.keys()):
            count = int(temp_union_max_count[key])
            temp_union_max_count[key] = [current_index, current_index + count]
            current_index = current_index + count
        dlx = DancingLinks(current_index, max_solution=30)
        for column in column_list:
            row_list = list()
            for index_1 in range(temp_union_max_count[column[0]][0], temp_union_max_count[column[0]][1]):
                for index_2 in range(temp_union_max_count[column[1]][0], temp_union_max_count[column[1]][1]):
                    row_list.append([index_1, index_2])
            dlx.append_same_row(row_list)

        for solution in dlx.search():
            temp_no_change_item = list()
            for key_list in solution:
                key_item = list()
                for key in sorted(temp_union_max_count.keys()):
                    if temp_union_max_count[key][0] <= key_list[len(key_item)] < temp_union_max_count[key][1]:
                        key_item.append(key)
                    if len(key_item) == len(key_list):
                        break
                temp_no_change_item.append('.'.join(key_item))
            no_change_list.append(temp_no_change_item)
            if len(no_change_list) >= 30:
                break

    return no_change_list


def handle_choice(choice_list):
    result_dict = dict()
    for pair in choice_list:
        state_1, state_2 = pair.split('.')
        result_dict[state_1] = result_dict.get(state_1, 0) + Fraction(1, 2)
        result_dict[state_2] = result_dict.get(state_2, 0) + Fraction(1, 2)
    return result_dict


def merge_dicts(d1, d2):
    result = {}
    for key, value in d1.items():
        result[key] = value
    for key, value in d2.items():
        if key in result:
            result[key] = max(result[key], value)
        else:
            result[key] = value
    return result
