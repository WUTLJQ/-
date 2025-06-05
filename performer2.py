import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from random import uniform
import time
# 1. 数据预处理
data = pd.read_csv(r'D:\论文1\光伏发电功率预测\运行数据\数据集3.csv')  # 读取CSV数据
data=data[:2000]
# 提取特征和目标变量
features = data.drop('Active_Power', axis=1)  #axis=1表示按列，axis=0表示按行
target = data['Active_Power']

# 数据标准化
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

features_scaled = scaler_features.fit_transform(features)
target_scaled = scaler_target.fit_transform(target.values.reshape(-1, 1))

# 转换为张量
features_tensor = torch.tensor(features_scaled, dtype=torch.float32)
target_tensor = torch.tensor(target_scaled, dtype=torch.float32)

# 创建时间序列数据集
sequence_length = 10  # 可以根据需求调整，每个输入数据包含多少个历史时间步的数据来预测一个目标值。
X, y = [], []
for i in range(len(features_tensor) - sequence_length):
    X.append(features_tensor[i:i + sequence_length])
    y.append(target_tensor[i + sequence_length])

X = torch.stack(X)
y = torch.stack(y)

# 划分训练集和测试集
train_size = int(0.6 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# 2. 定义PerformerAttention模块
class PerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, kernel_ratio=0.5):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.kernel_ratio = kernel_ratio
        self.kernel_dim = int(dim * kernel_ratio)

        self.query_projection = nn.Linear(dim, self.kernel_dim * heads)
        self.key_projection = nn.Linear(dim, self.kernel_dim * heads)
        self.value_projection = nn.Linear(dim, self.kernel_dim * heads)
        self.out_projection = nn.Linear(self.kernel_dim * heads, dim)

        self.feature_map = self._init_feature_map()

    def _init_feature_map(self):
        def feature_map(x):
            return torch.exp(-0.5 * torch.sum(x ** 2, dim=-1, keepdim=True))

        return feature_map

    def forward(self, x):
        batch_size, seq_length, dim = x.shape
        assert dim == self.dim, "Input dimension must match layer dimension"

        q = self.query_projection(x).view(batch_size, seq_length, self.heads, self.kernel_dim)
        k = self.key_projection(x).view(batch_size, seq_length, self.heads, self.kernel_dim)
        v = self.value_projection(x).view(batch_size, seq_length, self.heads, self.kernel_dim)

        q = self.feature_map(q)
        k = self.feature_map(k)

        kv = torch.einsum('bhld,bhle->bhde', k, v)
        qk = torch.einsum('bhld,bhld->bhl', q, k)
        qk = qk.unsqueeze(-1)

        output = torch.einsum('bhld,bhde->bhle', q, kv) / (qk + 1e-6)
        output = output.contiguous().view(batch_size, seq_length, self.heads * self.kernel_dim)
        return self.out_projection(output)


# 3. 定义CNN-LSTM模型，集成PerformerAttention
class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_channels, kernel_size, dropout):
        super(CNN_LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU()
        )
        self.performer_attention = PerformerAttention(num_channels)
        self.lstm = nn.LSTM(num_channels, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)  # Transpose for Conv1d input format
        x = self.cnn(x)
        x = x.transpose(1, 2)  # Transpose back to LSTM input format
        x = self.performer_attention(x)  # Apply Performer Attention
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take the last time step output
        out = self.fc(out)
        return out


input_size = features_tensor.shape[1]
hidden_size = 50  # 可以根据需求调整
num_layers = 2  # 可以根据需求调整#LSTM（长短时记忆网络）层的数量
output_size = 1
num_channels = 64#卷积层的输出通道数
kernel_size = 3
dropout = 0.2#防止过拟合


# 定义加权损失函数
class WeightedLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedLoss, self).__init__()
        self.weights = weights
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    def forward(self, output, target):
        loss_mse = self.mse(output, target)
        loss_mae = self.mae(output, target)
        return self.weights[0] * loss_mse + self.weights[1] * loss_mae


# 定义优化权重的函数
def optimize_weights(model, X_train, y_train, num_iterations=20):
    best_weights = None
    best_loss = float('inf')
    for _ in range(num_iterations):
        weights = [uniform(0, 1), uniform(0, 1)]
        weights = [w / sum(weights) for w in weights]  # 归一化权重
        criterion = WeightedLoss(weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 模型训练
        num_epochs = 20  # 为了快速优化，减少训练轮数
        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train)
            optimizer.zero_grad()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # 验证模型性能
        model.eval()
        outputs = model(X_train)
        loss = criterion(outputs, y_train).item()
        if loss < best_loss:
            best_loss = loss
            best_weights = weights

    return best_weights, best_loss


# 初始化模型
model = CNN_LSTM_Model(input_size, hidden_size, num_layers, output_size, num_channels, kernel_size, dropout)

# **记录训练开始时间**
start_time = time.time() # **Start time measurement**

# 优化权重
best_weights, best_loss = optimize_weights(model, X_train, y_train, num_iterations=20)
print(f'Best Weights: {best_weights}, Best Loss: {best_loss}')

# 使用最佳权重训练最终模型
criterion = WeightedLoss(best_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100  # 可以根据需求调整
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 验证集损失
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        test_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model loss')
plt.legend()
plt.show()

# 4. 结果预测和可视化
model.eval()
predicted_train = model(X_train).detach().numpy()
predicted_test = model(X_test).detach().numpy()

predicted_train = scaler_target.inverse_transform(predicted_train)
actual_train = scaler_target.inverse_transform(y_train.numpy())

predicted_test = scaler_target.inverse_transform(predicted_test)
actual_test = scaler_target.inverse_transform(y_test.numpy())

# 计算评价指标（训练集）
mse_train = mean_squared_error(actual_train, predicted_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(actual_train, predicted_train)
r2_train = r2_score(actual_train, predicted_train)

print(f'Training Set Metrics:')
print(f'MSE: {mse_train:.4f}')
print(f'RMSE: {rmse_train:.4f}')
print(f'MAE: {mae_train:.4f}')
print(f'R²: {r2_train:.4f}')

# 计算评价指标（测试集）
mse_test = mean_squared_error(actual_test, predicted_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(actual_test, predicted_test)
r2_test = r2_score(actual_test, predicted_test)

print(f'Test Set Metrics:')
print(f'MSE: {mse_test:.4f}')
print(f'RMSE: {rmse_test:.4f}')
print(f'MAE: {mae_test:.4f}')
print(f'R²: {r2_test:.4f}')

# 绘制预测结果与实际值对比图（训练集）
plt.figure(figsize=(12, 6))
plt.plot(predicted_train[-1500:], label='Predicted (Train)')
plt.plot(actual_train[-1500:], label='Actual (Train)')
plt.xlabel('Time')
plt.ylabel('Active Power')
plt.legend()
plt.title('RNN Model')
plt.show()

# 绘制预测结果与实际值对比图（测试集）
plt.figure(figsize=(12, 6))
plt.plot(predicted_test[-1500:], label='Predicted (Test)')
plt.plot(actual_test[-1500:], label='Actual (Test)')
plt.xlabel('Time')
plt.ylabel('Active Power')
plt.legend()
plt.title('RNN Model')
plt.show()

# **记录训练结束时间并计算总运行时间**
end_time = time.time()# **End time measurement**
runtime = end_time - start_time # **Calculate runtime**
print(f'Total training time: {runtime:.2f} seconds')