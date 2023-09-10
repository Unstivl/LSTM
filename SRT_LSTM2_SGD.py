import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


#gpu加速
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#导入数据，设置数据的超参数
data_seq=pd.read_csv("SRT\ConcatData.csv")
data_len = 40000
seq_len = 84
test_len = 1000

input_data = data_seq.iloc[0:data_len, 1:3]
input_data = torch.tensor(input_data.values, dtype=torch.float32)
input_set = torch.zeros(data_len-seq_len+1, seq_len, 2, dtype=torch.float32)
input_data=input_data.to(device)
input_set=input_set.to(device)

target_data = data_seq.iloc[seq_len:data_len+seq_len, 2]
target_data = torch.tensor(target_data.values, dtype=torch.float32)
target_set = torch.zeros(data_len-seq_len+1, seq_len, 1, dtype=torch.float32)
target_data=target_data.to(device)
target_set=target_set.to(device)

#滑窗拼接成合适的tensor用于训练
for i in range(data_len - seq_len + 1):
    input_seq = input_data[i:seq_len+i]
    input_set[i] = input_seq
    target_seq = target_data[i:seq_len+i]
    target_seq=target_seq.unsqueeze(-1)
    target_set[i] = target_seq


#测试集X,Y
testX_data = data_seq.iloc[data_len+510:data_len+510+test_len, 1:3]
testX_data = torch.tensor(testX_data.values, dtype=torch.float32)
testX_set=torch.zeros(test_len-seq_len+1, seq_len, 2, dtype=torch.float32)
testX_data=testX_data.to(device)
testX_set=testX_set.to(device)

testY_data = data_seq.iloc[data_len+510+seq_len:data_len+510+test_len+seq_len, 2]
testY_data = torch.tensor(testY_data.values, dtype=torch.float32)
testY_set=torch.zeros(test_len-seq_len+1, seq_len, 1, dtype=torch.float32)
testY_data=testY_data.to(device)
testY_set=testY_set.to(device)

for i in range(test_len - seq_len + 1):
    testX_seq = testX_data[i:seq_len+i]
    testX_set[i] = testX_seq
    testY_seq = testY_data[i:seq_len+i]
    testY_seq=testY_seq.unsqueeze(-1)
    testY_set[i] = testY_seq

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size,drop_prob=0.3): # ,weight_decay=0.01):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.weight_decay = weight_decay

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=drop_prob)
        
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

        # 调用参数初始化
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        # 将初始隐藏状态设置为零
        self.lstm.reset_parameters()
    
    def forward(self, x):
        # 初始化隐藏状态和记忆单元状态
        h0 = torch.normal(mean=0.0, std=0.1, size=(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c0 = torch.normal(mean=0.0, std=0.1, size=(self.num_layers, x.size(0), self.hidden_size)).to(device)
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        x=x.to(device)
        # 前向传播
        out, _ = self.lstm(x, (h0, c0))

        # 提取最后一个时间步的输出并应用全连接层
        out = self.fc(out[:, -1, :])
        return out
    


# 定义模型超参数
input_size = 2
hidden_size = 16
num_layers = 3
output_size = 1
batch_size = 32
num_epochs = 3
learning_rate = 0.01

# 实例化模型
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model.to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #,weight_decay=model.weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #,weight_decay=model.weight_decay)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate) #,weight_decay=model.weight_decay)
# optimizer = torch.optim.Adadelta(model.parameters())#, lr=learning_rate)#,weight_decay=model.weight_decay)
#第二个终端上使用adagrad训练，第三个终端上使用adadelta训练
train_loss_list = []
test_inacc_list = []
test_loss_list = []
# 模型训练
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for i in range(0, input_set.size(dim=0)- seq_len + 1):
        # 获取一个批次的数据和标签
        input_seq = input_set[i:i+batch_size]
        target_seq = target_set[i:i+batch_size]
        
        # 前向传播
        output_seq = model(input_seq)
        loss = criterion(output_seq, target_seq[:,0,:])
        
        # 反向传播并优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        # 查看最后几轮output_seq
        # if i>=input_set.size(dim=0)- seq_len + 1 - 100 and i<=input_set.size(dim=0)- seq_len + 1 - 90 :
        #     print("i=",i,"input_seq=",input_seq)
        #     print("output_seq=",output_seq)
        #     print("target_seq=",target_seq[:,0,:])
    
    train_loss_list.append(train_loss / (input_set.size(dim=0) / batch_size))

    #使用训练好的模型进行预测,计算平均正确率
    model.eval()
    test_loss = 0.0
    test_inacc = 0

    with torch.no_grad():    
        for i in range(0, testX_set.size(dim=0)- seq_len + 1):
            testX_seq = testX_set[i:i+batch_size]
            testY_seq = testY_set[i:i+batch_size]
            output_seq_test = model(testX_seq)
            
            # 看个别结果
            # if ((i>=15 and i<=20) or (i>=700 and i<=705)) and epoch==2:
                # print("i=",i,"testY_seq=",testY_seq[:,0,:])
                # print("output_seq_test=",output_seq_test)

            loss = criterion(output_seq_test, testY_seq[:,0,:])
            if torch.numel(testX_seq) != 0 and torch.numel(testY_seq) != 0 and testY_seq[:, 0, :].sum().item() != 0:
                test_loss += loss.item()
                inaccuracy = torch.abs(output_seq_test - testY_seq[:,0,:]) / testY_seq[:,0,:]
                test_inacc += inaccuracy.mean().item()
            # 看看个别数据如何   
        test_inacc=test_inacc/(testX_set.size(dim=0)- seq_len + 1)
        test_loss=test_loss/(testY_set.size(dim=0)- seq_len + 1)
        test_inacc_list.append(test_inacc)
        test_loss_list.append(test_loss)
        # test_inacc_list = list(filter(lambda x: x != float('inf'), test_inacc_list))
        print("Epoch: %d, Train loss: %.4f, Average Test loss: %.4f, Test inaccuracy: %.4f" % (epoch+1, train_loss / (input_set.size(dim=0) / batch_size), test_loss , test_inacc))

# 绘制不准确率曲线
fig, axs = plt.subplots(1, 3)

x1 = list(range(1, len(test_inacc_list) + 1))
axs[0].plot(x1, test_inacc_list)
axs[0].set_xlabel('epoch')
axs[0].set_ylabel('inaccuracy of power')
axs[0].set_title('Test Inaccuracy')

x2 = list(range(1, len(train_loss_list) + 1))
axs[1].plot(x2, train_loss_list)
axs[1].set_xlabel('epoch')
axs[1].set_ylabel('train loss')
axs[1].set_title('Train Loss')

x3 = list(range(1, len(test_loss_list) + 1))
axs[2].plot(x2, test_loss_list)
axs[2].set_xlabel('epoch')
axs[2].set_ylabel('test loss(total)')
axs[2].set_title('test Loss')

plt.tight_layout()
plt.savefig('SRT_LSTM2_SGD.svg', format='svg')

plt.show()



