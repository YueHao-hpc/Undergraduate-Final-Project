import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import itertools

# 数据加载与预处理
df = pd.read_excel('S+P.xlsx', header=None)
df.fillna(0, inplace=True)
df = df.iloc[1:, :]  # 删除第一行

# 过滤非法标签（仅保留0-9）
label_series = df.iloc[:, 16].astype(str)
df = df[label_series.str.fullmatch(r'[0-9]')]

# 分离特征与标签
labels = df.iloc[:, 16].astype(int).values
features = df.iloc[:, :16].astype(float).values  # 16个特征列

# 标准化
scaler = MinMaxScaler(feature_range=(-1, 1))
features_normalized = scaler.fit_transform(features)

# 调整输入形状为 (batch_size, 1, 16) 适配CNN
X_train, X_val, y_train, y_val = train_test_split(
    features_normalized,
    labels,
    test_size=0.2,
    random_state=42
)

# 转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # 增加通道维度
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 数据集和加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 模型定义
class PerceptronAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, lstm_output):
        attention_weights = self.attention(lstm_output).transpose(1, 2)
        return torch.bmm(attention_weights, lstm_output).squeeze(1)

class CNNLSTM(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=100, num_classes=10, num_layers=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(32, hidden_dim, num_layers, batch_first=True)
        self.attention = PerceptronAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, 1, 16)
        x = self.cnn(x)  # 输出: (batch, 32, 4)
        x = x.permute(0, 2, 1)  # 调整为 (batch, seq=4, 32)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))  # 输出: (batch, 4, 100)
        out = self.attention(out)         # 输出: (batch, 100)
        return self.fc(out)

# 参数设置（与BiGRU一致）
input_dim = 16
hidden_dim = 100
num_classes = 10
num_layers = 1
model = CNNLSTM(input_dim, hidden_dim, num_classes, num_layers)

# 训练配置
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0)
num_epochs = 500

# 指标记录
metrics = {
    'train_acc': [], 'val_acc': [],
    'train_pre': [], 'val_pre': [],
    'train_rec': [], 'val_rec': [],
    'train_f1': [], 'val_f1': []
}

# 训练循环
for epoch in range(num_epochs):
    model.train()
    train_preds, train_true = [], []
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        train_preds.extend(torch.argmax(output, 1).cpu().numpy())
        train_true.extend(target.cpu().numpy())
    
    # 训练指标
    metrics['train_acc'].append(accuracy_score(train_true, train_preds))
    metrics['train_pre'].append(precision_score(train_true, train_preds, average='macro', zero_division=0))
    metrics['train_rec'].append(recall_score(train_true, train_preds, average='macro', zero_division=0))
    metrics['train_f1'].append(f1_score(train_true, train_preds, average='macro', zero_division=0))

    # 验证集评估
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data)
            val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    metrics['val_acc'].append(accuracy_score(val_true, val_preds))
    metrics['val_pre'].append(precision_score(val_true, val_preds, average='macro', zero_division=0))
    metrics['val_rec'].append(recall_score(val_true, val_preds, average='macro', zero_division=0))
    metrics['val_f1'].append(f1_score(val_true, val_preds, average='macro', zero_division=0))

    print(f'Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | '
          f"Val Acc: {metrics['val_acc'][-1]*100:.2f}%")

# 可视化与保存（与BiGRU一致）
plt.figure(figsize=(10,5))
for i, metric in enumerate(['acc', 'pre', 'rec', 'f1']):
    plt.subplot(2,2,i+1)
    plt.plot(metrics[f'train_{metric}'], label='Train')
    plt.plot(metrics[f'val_{metric}'], label='Val')
    plt.title(metric.capitalize())
    plt.legend()
plt.tight_layout()
plt.savefig('cnn_lstm_metrics.png')



# 可视化与保存（添加以下新内容）
# 训练指标曲线（保持原样）
plt.figure(figsize=(10,5))
for i, metric in enumerate(['acc', 'pre', 'rec', 'f1']):
    plt.subplot(2,2,i+1)
    plt.plot(metrics[f'train_{metric}'], label='Train')
    plt.plot(metrics[f'val_{metric}'], label='Val')
    plt.title(metric.capitalize())
    plt.legend()
plt.tight_layout()
plt.savefig('cnn_lstm_metrics.png')
plt.show()

# +++ 新增混淆矩阵 +++
conf_matrix = confusion_matrix(val_true, val_preds, labels=np.arange(10))
plt.figure(figsize=(5,5))
plt.imshow(conf_matrix, cmap=plt.cm.Blues, vmax=10)
plt.title('Confusion Matrix')
plt.colorbar()
classes = [f'Class {i}' for i in range(10)]
tick_marks = np.arange(10)
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted')
plt.ylabel('True')
for i, j in itertools.product(range(10), range(10)):
    plt.text(j, i, conf_matrix[i,j], 
             ha='center', 
             color='white' if conf_matrix[i,j]>5 else 'black')
plt.tight_layout()
plt.savefig('cnn_lstm_confusion_matrix.png')
plt.show()

# +++ 新增分类指标柱状图 +++
conf_matrix = np.array(conf_matrix)
num_classes = conf_matrix.shape[0]
precision = np.zeros(num_classes)
recall = np.zeros(num_classes)
f1_scores = np.zeros(num_classes)

for i in range(num_classes):
    TP = conf_matrix[i, i]
    FP = conf_matrix[:, i].sum() - TP
    FN = conf_matrix[i, :].sum() - TP
    
    precision[i] = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall[i] = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0

classes = [f'Class {i}' for i in range(num_classes)]
x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score')

ax.set_ylabel('Scores')
ax.set_title('Per-class Evaluation Metrics')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('cnn_lstm_class_scores.png')
plt.show()

# 保存模型（保持原样）
torch.save(model, 'model_cnn_lstm.pth')