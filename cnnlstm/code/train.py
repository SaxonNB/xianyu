#先导入需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
#读取数据
df = pd.read_csv('climate_total.csv')
 
#查看描述统计
df.describe().style.background_gradient(cmap = 'Oranges')
df.drop(['avg_slp', 'avg_stp', 'gust'], axis=1, inplace=True)#将日期转化为时间戳格式
df['DATE'] = pd.to_datetime(df['DATE'])
 
#从日期中提取特征
df['Day_of_Week'] = df['DATE'].dt.dayofweek
df['month'] = df['DATE'].dt.month
df['day'] = df['DATE'].dt.day
 
#添加一个新特征，每天温度的极差，即温度最大值减去最小值
df['min_max_diff'] = df['maxTemp'] - df['minTemp']
cols = [col for col in df.columns if col not in ['DATE', 'avg_temp', 'month', 'day', 'Day_of_Week']]
df_shifted = df[cols].shift(1)
df = pd.concat([df[['avg_temp', 'month', 'day', 'Day_of_Week', 'DATE']], df_shifted], axis=1)
df = df[1:]
from sklearn.preprocessing import MinMaxScaler
 
#选择特征列
features = ['avg_dewPoint', 'maxTemp', 'minTemp', 'mxspd', 'Prcp', 'visib', 'avg_windSp', 'min_max_diff', 'month', 'day', 'Day_of_Week', 'avg_temp']
#进行归一化处理
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features].values.astype('float32'))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
 
def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[:,:11][i:i+lookback]
        target = dataset[:, 11][i+lookback-1]
        X.append(feature)
        y.append(target)
    return X, y
 
def create_train_data(lookback):
    df = pd.read_csv("climate_total.csv")
    df.drop(['avg_slp', 'avg_stp', 'gust'], axis=1, inplace=True)
 
    df['DATE'] = pd.to_datetime(df['DATE'])
 
    df['Day_of_Week'] = df['DATE'].dt.dayofweek
    df['month'] = df['DATE'].dt.month
    df['day'] = df['DATE'].dt.day
 
    df['min_max_diff'] = df['maxTemp'] - df['minTemp']
 
    cols = [col for col in df.columns if col not in ['DATE', 'avg_temp', 'month', 'day', 'Day_of_Week']]
    df_shifted = df[cols].shift(1)
    df = pd.concat([df[['avg_temp', 'year', 'month', 'day', 'Day_of_Week', 'DATE']], df_shifted], axis=1)
    df = df[1:]
 
 
    features = ['avg_dewPoint', 'maxTemp', 'minTemp', 'mxspd', 'Prcp', 'visib', 'avg_windSp', 'min_max_diff', 'month', 'day', 'Day_of_Week', 'avg_temp']
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features].values.astype('float32'))
 
    X_lookback, y_lookback = create_dataset(df_scaled, lookback=lookback)
 
    idx_test = df.loc[df['DATE'] == '2023-01-01'].index[0]
    idx_val = df.loc[df['DATE'] == '2022-01-01'].index[0]
    X_train, y_train = torch.tensor(X_lookback[:(idx_test-lookback)]), torch.tensor(y_lookback[:(idx_test-lookback)]) #因为使用的时间窗口是lookback天，故实际上训练数据的第一个样本是从第lookback天开始的
    X_test, y_test = torch.tensor(X_lookback[(idx_test - lookback):]), torch.tensor(y_lookback[(idx_test - lookback):])
 
    X_val, y_val = torch.tensor(X_lookback[(idx_val - lookback):(idx_test-lookback)]), torch.tensor(y_lookback[(idx_val - lookback):(idx_test-lookback)])
 
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


#导入torch库
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
 
 
def train_model_no_attention(X_train, y_train, X_val, y_val):
    #定义早停机制
    class CustomEarlyStopping:
        def __init__(self, patience=10, delta=0, verbose=False):
            self.patience = patience
            self.delta = delta
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
 
        
        def __call__(self, val_loss):
            score = -val_loss
 
            if self.best_score is None:
                self.best_score = score
 
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}, score: {self.best_score}')
        
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
 
        
    #定义神经网络模型
 
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=11, out_channels=64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
            self.maxpool = nn.MaxPool1d(2)
            self.lstm1 = nn.LSTM(input_size=64, hidden_size=128,batch_first=True)
            self.dropout1 = nn.Dropout(0.2)
            self.bidirectional = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
            self.dropout2 = nn.Dropout(0.2)
            self.dense1 = nn.Linear(128 * 2, 64)  
            self.dense2 = nn.Linear(64, 8)
            self.dense3 = nn.Linear(8, 1)
 
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.maxpool(x.permute(0, 2, 1)) 
            x, _ = self.lstm1(x)
            x = self.dropout1(x)
            x, _ = self.bidirectional(x)
            x = x[:, -1, :]
            x = self.dropout2(x)
            x = F.sigmoid(self.dense1(x))
            x = self.dense2(x)
            x = self.dense3(x)
            return x
 
    #用来计算验证集的rmse
    def calculate_rmse(model, X, y, criterion):
        with torch.no_grad():
            y_pred = model(X.permute(0, 2, 1)).detach()
            rmse = np.sqrt(criterion(y_pred.cpu(), y.unsqueeze(1).detach()))
        return rmse
 
 
    # 创建模型实例
    Torchmodel = Net()
 
    # 定义损失函数和优化器，这类使用的是MSE损失函数和AdamW优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(Torchmodel.parameters(), lr=1e-3, weight_decay=1e-5)
 
    # 定义学习率调整器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
 
    # 将数据、模型和损失函数转移到GPU上
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Torchmodel = Torchmodel.to(device)
    criterion = criterion.to(device)
 
 
    # 模型定义完成后，开始训练模型
    # 定义训练参数
 
    loader = data.DataLoader(data.TensorDataset(X_train, y_train),
                            batch_size = 8, shuffle = True)
 
    X_val = X_val.to(device)
 
    #创建早停机制实例
    early_stopping = CustomEarlyStopping(patience=10, verbose=True)
 
    #用来存储训练和验证损失
    train_losses = []
    val_losses = []
 
    epochs = 300
    for epoch in range(epochs):
        Torchmodel.train()
        train_loss = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = Torchmodel(X_batch.permute(0, 2, 1))
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()  # 累加每个batch的损失
 
        train_loss /= len(loader)  # 计算平均训练损失
        train_losses.append(np.sqrt(train_loss))  # 保存训练损失
 
        Torchmodel.eval()
        
        val_rmse = calculate_rmse(Torchmodel, X_val, y_val, criterion)
        val_losses.append(val_rmse.item())  # 保存验证损失
 
        scheduler.step(val_rmse)
        
        early_stopping(val_rmse)
        
        # 应用早停机制，确定是否停止训练
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        if epoch % 10 == 0:
            print('*'*10, 'Epoch: ', epoch, '\ train RMSE: ', np.sqrt(train_loss), '\ val RMSE', val_rmse.item())
    
    return Torchmodel, train_losses, val_losses

X_train, y_train, X_val, y_val, X_test, y_test, scaler = create_train_data(lookback=14)
 
Torchmodel, train_losses, val_losses= train_model_no_attention(X_train, y_train, X_val, y_val)
 
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



Torchmodel.eval()
with torch.no_grad():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_test = X_test.to(device)
    predict = Torchmodel(X_test.permute(0, 2, 1)).cpu().detach()
 
 
#为了便于进行方向归一化处理，我们这里将预测值与测试集数据合并在一起
predict_sum = np.concatenate((X_test[:, -1, :].cpu().numpy(), predict), axis=1)
test_sum = np.concatenate((X_test[:, -1, :].cpu().numpy(), y_test.cpu().numpy().reshape(-1, 1)), axis=1)
 
predict_sum = scaler.inverse_transform(predict_sum)
test_sum = scaler.inverse_transform(test_sum)
 
r2 = r2_score(test_sum[:, -1], predict_sum[:, -1])
rmse = np.sqrt(mean_squared_error(test_sum[:, -1], predict_sum[:, -1]))
 
print("R² Score:", r2)
print("RMSE:", np.sqrt(mean_squared_error(test_sum[:, -1], predict_sum[:,-1])))
print("MAPE:", mean_absolute_percentage_error(test_sum[:, -1], predict_sum[:,-1]))