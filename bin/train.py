import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
from torch.utils.data import TensorDataset

# 加载数据
matrices = np.load('matrices.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

# matrix_list = list()
# for matrix in matrices:
#     matrix_list.append(torch.from_numpy(matrix))
#
# # 将 tensor 列表转换为 TensorDataset
# tensor_dataset = TensorDataset(torch.stack(matrix_list))
#
# # 然后，你可以使用 DataLoader 来创建一个 dataloader
# dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)

# 转换为 PyTorch 张量
matrices = torch.tensor(matrices, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# 创建数据加载器
dataset = TensorDataset(matrices, labels)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

train_dataset, test_dataset = Data.random_split(dataset, [1500, 500])
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.softmax(self.layer5(x))
        return x


# 定义输入和输出
input_dim = 256
hidden_dim = 512
output_dim = 2

# 创建模型
model = Net(input_dim, hidden_dim, output_dim)

# 定义优化器，包含L2正则化
optimizer = torch.optim.Adam(model.parameters())

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练神经网络
for epoch in range(10):  # 这里我们只做100个epoch，你可以根据需要调整
    mean_loss = 0.0
    count = 0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        count += 1

    print(f'Epoch {epoch + 1}/{10000}, Loss: {mean_loss / count}')

model.eval()  # 切换到评估模式
correct = 0
total = 0
with torch.no_grad():  # 不需要计算梯度
    for data, target in test_loader:
        output = model(data)
        # 计算并打印损失
        # loss = criterion(output, target)
        _, predicted = torch.max(output.data, 1)
        print(predicted, target)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print('Accuracy of the network on the test data: %d %%' % (100 * correct / total))
