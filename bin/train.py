import numpy as np
import torch
import torch.utils.data as Data
from torch import nn
from torch.utils.data import TensorDataset

# 加载数据
full_sep_data = np.load('full_sep_states.npy', allow_pickle=True)
full_sep_labels = np.load('full_sep_labels.npy', allow_pickle=True)

part_3_data = np.load('part_3_states.npy', allow_pickle=True)
part_3_labels = np.load('part_3_labels.npy', allow_pickle=True)

# full_sep_data = full_sep_data[:2500]
# full_sep_labels = full_sep_labels[:2500]

matrices = np.concatenate((full_sep_data, part_3_data))
labels = np.concatenate((full_sep_labels, part_3_labels))

# 转换为 PyTorch 张量
matrices = torch.tensor(matrices, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# start test data
full_sep_test_data = np.load('full_sep_states_test.npy', allow_pickle=True)
full_sep_test_labels = np.load('full_sep_labels_test.npy', allow_pickle=True)
part_3_test_data = np.load('part_3_states_test.npy', allow_pickle=True)
part_3_test_labels = np.load('part_3_labels_test.npy', allow_pickle=True)
test_matrices = np.concatenate((full_sep_test_data, part_3_test_data))
test_labels = np.concatenate((full_sep_test_labels, part_3_test_labels))
test_matrices = torch.tensor(test_matrices, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(matrices, labels)
test_dataset = TensorDataset(test_matrices, test_labels)
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# train_size = int(0.7 * matrices.shape[0])
# train_dataset, test_dataset = Data.random_split(dataset, [train_size, matrices.shape[0] - train_size])
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.softmax(self.layer3(x))
        return x


# 定义输入和输出
input_dim = 256
output_dim = 2

# 创建模型
model = Net(input_dim, output_dim)

# 定义优化器，包含L2正则化
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 训练神经网络
for epoch in range(100):  # 这里我们只做100个epoch，你可以根据需要调整
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
    if mean_loss / count < 0.32:
        break

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

index_list = [1, 2, 4, 8]
w_state = np.zeros(16)
for index in index_list:
    w_state[index] = 0.5
w_state = np.outer(w_state, w_state)

rho_w_list = list()
number_list = list()
for i in range(70):
    number = 0.05 + i * 0.01
    rho_w_list.append(number * w_state + ((1 - number) / 16) * np.eye(16))
    number_list.append(number)

with torch.no_grad():
    rho_w_state = torch.tensor(rho_w_list, dtype=torch.float32)
    output = model(rho_w_state)
    # 计算并打印损失
    # loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)
    print(number_list)
    print(predicted)
