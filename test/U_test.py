import numpy as np

import numpy as np

def find_boundary(arr):
    # 检查数组是否满足条件：前面全是0，后面全是1
    if np.all(arr[:np.where(arr==1)[0][0]] == 0) and np.all(arr[np.where(arr==1)[0][0]:] == 1):
        return np.where(arr==1)[0][0]
    else:
        return False

# 测试代码
arr = np.array([0, 0, 0, 1, 1, 1])
print(find_boundary(arr))  # 输出：3

arr = np.array([0, 0, 1, 0, 1, 1])
print(find_boundary(arr))  # 输出：False


# 生成一个0到15的随机数设为a
a = np.random.randint(16)

# 生成一个长度为a的list，标记为b，里面每个数是0到499
b = np.random.randint(500, size=a)

# 生成一个长度为a的list，标记为c，每个数是0到1的小数，然后总和为1
c = np.random.dirichlet(np.ones(a), size=1)[0]

# 假设我们有一个矩阵列表matrix_list
matrix_list = [np.random.rand(3, 3) for _ in range(500)]

# 用b作为角标，去取某个矩阵列表里的矩阵，乘上c
result = sum(c[i] * matrix_list[b[i]] for i in range(a))

print(result)

# import torch
# from torch.autograd import Variable
#
# # 初始化四个实数参数
# params = Variable(torch.randn(4), requires_grad=True)
# optimizer = torch.optim.SGD([params], lr=0.01)
#
# for i in range(1000):
#     optimizer.zero_grad()
#
#     # 将参数转化为复数酉矩阵
#     a, b, c, d = params
#     U = torch.tensor([[torch.cos(a) * torch.exp(1j * b), torch.sin(a) * torch.exp(1j * c)],
#                       [-torch.sin(a) * torch.exp(-1j * c), torch.cos(a) * torch.exp(-1j * d)]])
#
#     # 计算损失函数，这里我们使用矩阵的迹
#     loss = -torch.trace(U)
#
#     # 计算梯度
#     loss.backward()
#
#     # 更新参数
#     optimizer.step()
#
# print(U)
