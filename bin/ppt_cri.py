import numpy as np

# 假设我们有两个只包含0和1的numpy标签数组
labels1 = np.array([0, 1, 0, 1, 1, 0, 1, 1, 0, 1])
labels2 = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])

# 根据条件创建一个新的标签数组
# new_labels = np.where(labels1 == 0, 0, np.where(labels2 == 1, 2, 1))

new_labels = np.where(labels2 == 1, 2, np.where(labels1 == 0, 0, 1))

# 打印新的标签数组
print(new_labels)
