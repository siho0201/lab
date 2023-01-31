import scipy.io as sio
import torch
import numpy as np
import random
import os
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from PIL import Image
def random_seed(seed): # Set the seed to the same value to maintain the dataset
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Using GPU : ', torch.cuda.is_available() , ' |  Seed : ', seed)

def multi_gpu(gpu_number): # This function is required to use multi gpu. Please enter the gpu number. example: multi_gpu("0,1,3")
    torch.multiprocessing.set_start_method('spawn')
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
    os.environ["CUDA_VISIBLE_DEVICES"]= gpu_number  
    NGPU = torch.cuda.device_count()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'9. Selected device: {DEVICE}')
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

def binaryzation(dataset): # This function binarizes the answer dataset.
    for i in range(len(dataset)):
        ori = dataset[i]
        ori = ori*1.2 / np.max(ori)
        ori = np.exp(ori)
        ori = np.uint8(255*ori /np.max(ori))
        th, dst = cv2.threshold(ori, -1, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        dataset[i] = dst
        return dataset

def data_reshape(dataset): # This function reshape your dataset To configure the data to be put into the model
    dataset = np.array(dataset)
    dataset = dataset.reshape(len(dataset),1,256,256)
    return dataset

def saveModel(net): # save the model
    path = "./U2net.pth"
    torch.save(net.state_dict(),path)

class Train_data(Dataset): # make train dataset
  def __init__(self, X_train,y_train):
    self.x_data = X_train
    self.y_data = y_train

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx].astype(np.float32)
    y = self.y_data[idx].astype(np.float32)
    x = x/255
    y = y/255

    x = torch.tensor(x,dtype = torch.float32)
    y = torch.tensor(y,dtype = torch.float32)

    return x, y

class val_data(Dataset): # make validation dataset
  def __init__(self,X_val, y_val):
    self.x_data = X_val
    self.y_data = y_val

  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx].astype(np.float32)
    y = self.y_data[idx].astype(np.float32)
    x = x/255
    y = y/255
    x = torch.tensor(x,dtype = torch.float32)
    y = torch.tensor(y,dtype = torch.float32)

    return x, y

class Test_data(Dataset): # make test dataset
  def __init__(self,sample):
    self.x_data = sample
 
  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx].astype(np.float32)
    x = x/255
    x = torch.tensor(x,dtype = torch.float32)
    return x
class label_data(Dataset): # make test dataset for test level
  def __init__(self,label):
    self.x_data = label
 
  def __len__(self): 
    return len(self.x_data)

  def __getitem__(self, idx): 
    x = self.x_data[idx].astype(np.float32)
    x = x/255
    x = torch.tensor(x,dtype = torch.float32)
    return x   

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v): # u2-net loss function
    bce_loss = nn.BCELoss(size_average=True)
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))
    return loss0, loss

def normPRED(d): # normalize output
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)

    return dn

def check_output_image(input_image, output_image,answer_image): # save output image
	for i in range(132):
		count =0
		input_sample = np.uint8(input_image[i]*255)
		output_sample = np.round(output_image[i]*255)
		output_sample[output_sample < 0] = 0
		output_sample[output_sample > 250] = 255
		output_sample = np.uint8(output_sample)
		answer_sample = np.uint8(answer_image[i]*255)
		input = Image.fromarray(input_sample)
		output = Image.fromarray(output_sample)
		answer = Image.fromarray(answer_sample)
		if i < 11:
			input.save('./u2net_img/LFM_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/LFM_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/LFM_'+str(count)+'_answer.jpg','JPEG')
		elif 11<= i < 22:
			input.save('./u2net_img/Costas_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/Costas_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/Costas_'+str(count)+'_answer.jpg','JPEG')
		elif 22<= i < 33:
			input.save('./u2net_img/Barker_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/Barker_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/Barker_'+str(count)+'_answer.jpg','JPEG')
		elif 33<= i < 44:
			input.save('./u2net_img/Frank_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/Frank_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/Frank_'+str(count)+'_answer.jpg','JPEG')
		elif 44<= i < 55:
			input.save('./u2net_img/T1_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/T1_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/T1_'+str(count)+'_answer.jpg','JPEG')
		elif 55<= i < 66:
			input.save('./u2net_img/T2_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/T2_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/T2_'+str(count)+'_answer.jpg','JPEG')
		elif 66<= i < 77:
			input.save('./u2net_img/T3_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/T3_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/T3_'+str(count)+'_answer.jpg','JPEG')
		elif 77<= i < 88:
			input.save('./u2net_img/T4_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/T4_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/T4_'+str(count)+'_answer.jpg','JPEG')
		elif 88<= i < 99:
			input.save('./u2net_img/P1_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/P1_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/P1_'+str(count)+'_answer.jpg','JPEG')
		elif 99<= i < 110:
			input.save('./u2net_img/P2_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/P2_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/P2_'+str(count)+'_answer.jpg','JPEG')
		elif 110<= i < 121:
			input.save('./u2net_img/P3_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/P3_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/P3_'+str(count)+'_answer.jpg','JPEG')
		if 121<= i < 132:
			input.save('./u2net_img/P4_'+str(count)+'_input.jpg','JPEG')
			output.save('./u2net_img/P4_'+str(count)+'_output.jpg','JPEG')
			answer.save('./u2net_img/P4_'+str(count)+'_answer.jpg','JPEG')
		count +=1 
		if count >10:
			count = 0
    

