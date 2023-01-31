# This code was created to check the results of the u2-net. Select a random value from the dataset
# to store the model's input and output values and correct answer values.

from function  import * 
import torch
from random import *
from model import U2NET 
from PIL import Image
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NGPU = torch.cuda.device_count()
multi_gpu("1,2")
random_seed(0)
batch_size = 1
#-------------------------DATA LOAD------------------------------
label = []
input = [] 
signal_name = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2' ,'P3', 'P4', 'T1', 'T2', 'T3', 'T4'] # signal_type
for i in range(-5,6): # all snr dataset
            tx = sio.loadmat('./dataset-CWD-500(add_tx)/dataset_tx'+str(i*2))
            rx = sio.loadmat('./dataset-CWD-500(add_tx)/dataset'+str(i*2))
            for j in range(len(signal_name)):
                label.extend(tx[signal_name[j]+'_tx'])
                input.extend(rx[signal_name[j]])
label = binaryzation(label)
input = data_reshape(input) 
label = data_reshape(label)
test_dataset = Test_data(input)
label_dataset = label_data(label)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle =False)
labelloader = DataLoader(label_dataset, batch_size=batch_size, shuffle =False)
#-------------------------MODEL LOAD AND RUN ------------------------------
net = U2NET(1,1) # model load
if (DEVICE.type == 'cuda') and (torch.cuda.device_count() > 1):
    print('Multi GPU activate')
    net = nn.DataParallel(net, device_ids = list(range(NGPU)))
net.to(DEVICE)
path = "U2net.pth" 
net.load_state_dict(torch.load(path))
inputs_arr = []
outputs_arr = []
answer_arr = []
with torch.no_grad(): 
    for i, inputs in enumerate(testloader,0): 
        inputs = inputs.to(DEVICE)
        # print(inputs.size())
        d1,d2,d3,d4,d5,d6,d7 = net(inputs)
        pred = d1
        pred = normPRED(pred)
        inputs_image = inputs.cpu().numpy()
        outputs_image = pred.cpu().numpy()

        inputs_arr.extend(inputs_image)
        outputs_arr.extend(outputs_image)
        
    for i, labels in enumerate(labelloader,0): 
        labels = labels.to(DEVICE) 
        labels_image = labels.cpu().numpy()
        answer_arr.extend(labels_image)
#-------------------------RESULT------------------------------ # save input image, output image, label image

input_image= np.array(inputs_arr) 
input_image = np.squeeze(input_image,1)

output_image= np.array(outputs_arr)
output_image = np.squeeze(output_image,1)

answer_image = np.array(answer_arr)
answer_image = np.squeeze(answer_image)

print('----------------------finish----------------------')