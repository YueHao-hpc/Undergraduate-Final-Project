import pandas as pd
import torch
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

# 调整输入形状为 (batch_size, seq_len=1, input_dim=16)
X_train, X_val, y_train, y_val = train_test_split(
    features_normalized.reshape(-1, 1, 16),  # 增加序列维度
    labels,
    test_size=0.2,
    random_state=42
)

# 转换为张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# 数据集和加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# BiGRU模型定义
class BiGRUModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1):
        super(BiGRUModel, self).__init__()
        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)  # 双向输出拼接

    def forward(self, x):
        out, _ = self.gru(x)          # 输入形状: (batch, seq=1, input_dim=16)
        out = out[:, -1, :]            # 取最后时间步，形状: (batch, hidden_dim*2)
        out = self.fc(out)             # 输出形状: (batch, num_classes)
        return out

# 参数设置
input_dim = 16    # 输入特征维度
hidden_dim = 100  # 隐层维度
num_classes = 10  # 类别数
model = BiGRUModel(input_dim, hidden_dim, num_classes)

# 训练配置（与MLP一致）
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0, last_epoch=-1)
num_epochs = 500

# 指标记录
train_acc_list = []
val_acc_list = []
train_pre_list = []
val_pre_list = []
train_rec_list = []
val_rec_list = []
train_f1_list = []
val_f1_list = []

# 训练循环
for epoch in range(num_epochs):
    model.train()
    train_correct, train_total = 0, 0
    train_preds, train_true = [], []
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
        train_preds.extend(predicted.cpu().numpy())
        train_true.extend(target.cpu().numpy())
    
    # 计算训练指标
    train_acc = train_correct / train_total
    train_pre = precision_score(train_true, train_preds, average='macro', zero_division=0)
    train_rec = recall_score(train_true, train_preds, average='macro', zero_division=0)
    train_f1 = f1_score(train_true, train_preds, average='macro', zero_division=0)
    
    # 验证集评估
    model.eval()
    val_correct, val_total = 0, 0
    val_preds, val_true = [], []
    with torch.no_grad():
        for data, labels in val_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    
    val_acc = val_correct / val_total
    val_pre = precision_score(val_true, val_preds, average='macro', zero_division=0)
    val_rec = recall_score(val_true, val_preds, average='macro', zero_division=0)
    val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
    
    # 记录指标
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_pre_list.append(train_pre)
    val_pre_list.append(val_pre)
    train_rec_list.append(train_rec)
    val_rec_list.append(val_rec)
    train_f1_list.append(train_f1)
    val_f1_list.append(val_f1)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc*100:.2f}%')

# 绘制指标曲线（文件名改为bigru_xxx.png）
plt.figure(figsize=(10, 5))
plt.subplot(2,2,1)
plt.plot(train_acc_list, label='Train')
plt.plot(val_acc_list, label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(2,2,2)
plt.plot(train_pre_list, label='Train')
plt.plot(val_pre_list, label='Validation')
plt.title('Precision')
plt.legend()

plt.subplot(2,2,3)
plt.plot(train_rec_list, label='Train')
plt.plot(val_rec_list, label='Validation')
plt.title('Recall')
plt.legend()

plt.subplot(2,2,4)
plt.plot(train_f1_list, label='Train')
plt.plot(val_f1_list, label='Validation')
plt.title('F1 Score')
plt.legend()

plt.tight_layout()
plt.savefig('bigru_metrics.png')
plt.show()

# 混淆矩阵（文件名改为bigru_confusion_matrix.png）
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
    plt.text(j, i, conf_matrix[i,j], ha='center', color='white' if conf_matrix[i,j]>5 else 'black')
plt.tight_layout()
plt.savefig('bigru_confusion_matrix.png')
plt.show()
# 分类指标柱状图（文件名改为bigru_class_scores.png）
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
plt.savefig('bigru_class_scores.png')
plt.show()
# 保存模型
torch.save(model, 'model_bigru.pth')