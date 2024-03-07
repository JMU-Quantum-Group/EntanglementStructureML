import numpy as np

# 生成50个随机矩阵
matrices = [np.random.rand(10, 10) for _ in range(50)]

# 生成50个随机标签
labels = np.random.randint(0, 2, 50)  # 这里生成的是0或1的标签，你可以根据需要修改

# 保存到npy文件
np.save('matrices.npy', matrices)
np.save('labels.npy', labels)
