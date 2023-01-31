# This code is a u2-net-based SOD model for de-noising imaged radar signals.
from function import *
from model import U2NET
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NGPU = torch.cuda.device_count()
multi_gpu("1,2")
random_seed(0)
batch_size = 32

#-------------------------DATA LOAD------------------------------
label = []
input = [] 
signal_number = 50 # select each data number
signal_name = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2' ,'P3', 'P4', 'T1', 'T2', 'T3', 'T4'] # signal_type
for i in range(-5,6): # all snr dataset
            tx = sio.loadmat('./dataset-CWD-500(add_tx)/dataset_tx'+str(i*2))
            rx = sio.loadmat('./dataset-CWD-500(add_tx)/dataset'+str(i*2))
            for j in range(len(signal_name)):
                label.extend(tx[signal_name[j]+'_tx'][0:signal_number])
                input.extend(rx[signal_name[j]][0:signal_number])
label = binaryzation(label) # Answer label data is binary data
input = data_reshape(input) # Data preprocessing for use as input to the model
label = data_reshape(label)

X_train, X_val, y_train, y_val = train_test_split(input,label,test_size=0.2,random_state=42) # split train,validation data
train_dataset = Train_data(X_train, y_train)
val_dataset= val_data(X_val,y_val)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


#-------------------------MODEL SETTING ------------------------------
net = U2NET(1,1) # model
if (DEVICE.type == 'cuda') and (torch.cuda.device_count() > 1): # for multi gpu code
    print('Multi GPU activate')
    net = nn.DataParallel(net, device_ids = list(range(NGPU)))
net.to(DEVICE)
optimizer = op.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) # optimizer
epoch_num = 1000 # epoch number
batch_size_train = 16
batch_size_val = 1
train_num = 0
val_num = 0
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
#-------------------------MODEL TRAIN ---------------------------------

print("----------------------Train start------------------------")
for epoch in range(0,epoch_num):
	net.train  # Train loop
	for data in trainloader:
		ite_num = ite_num + 1
		ite_num4val = ite_num4val + 1
		train_inputs, train_labels = data
		# print(train_inputs)
		train_inputs = train_inputs.to(DEVICE) 
		train_labels = train_labels.to(DEVICE) 
		optimizer.zero_grad() 
		# print(train_inputs.size())
		# print(train_labels.size())
		d0, d1, d2, d3, d4, d5, d6  = net(train_inputs) 
		# print(outputs.size())
		loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, train_labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.data.item()
		running_tar_loss += loss2.data.item()
		del d0, d1, d2, d3, d4, d5, d6, loss2, loss 

		print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
		if epoch % 10 == 0: # Save model every 10 epoch
			saveModel()
			print('savemodel')



