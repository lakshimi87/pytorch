"""
Title: facial point detection
Description: facial point detection
Author: Aubrey Choi
Date: 2024-07-10
Version: 1.2
License: MIT License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import glob
import random
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

TrainImagePath = "images/train_images"
TestImagePath = "images/test_images"
SaveImagePath = "saves"
OutputDataPath = "training.csv"

isCuda = torch.cuda.is_available()
isMps = torch.backends.mps.is_available()
device = torch.device('cuda' if isCuda else 'mps' if isMps else 'cpu')
print(f'Current device is {device}')

class ImageDataset(Dataset):
	def __init__(self, dataframe, tfs):
		self.dataframe = dataframe
		self.transform = transforms.Compose(tfs)
	def __getitem__(self, index):
		v = self.dataframe.iloc[index]
		image = Image.open(v['image']).getchannel(0)
		image = self.transform(image)
		return (image, v[:-1].values.astype('float32'))
	def __len__(self):
		return len(self.dataframe)
	def getLabel(self, index):
		v = self.dataframe.iloc[index]
		return v['image']

def outputImage(file, v):
	image = Image.open(file)
	draw = ImageDraw.Draw(image)
	for i in range(0, len(v), 2):
		draw.rectangle(((v[i]-1, v[i+1]-1), (v[i]+1, v[i+1]+1)))
	return image

def showImage(image, label):
	plt.imshow(image)
	plt.title(label)
	plt.show()

def getFileList(root):
	return glob.glob(root+"/*.jpg")
			
df = pd.read_csv(OutputDataPath)
df['image']=np.array([f"{TrainImagePath}/{idx}.jpg" for idx in range(len(df))])
print(f"total data count : {len(df)}")
xTrain, xTest = train_test_split(df, test_size=0.2)
xTrain.reset_index(drop=True, inplace=True)
xTest.reset_index(drop=True, inplace=True)
print(xTrain.head())
print(xTest.head())

tfs = [
	transforms.ToTensor(), 
	transforms.Normalize((0.5,), (0.5,)),
]
trainData = ImageDataset(xTrain, tfs)
testData = ImageDataset(xTest, tfs)
x, y = trainData[0]
label = trainData.getLabel(0)
showImage(outputImage(label, y), label)
validationData = getFileList(TestImagePath)

BatchSize = 16
EpochNum = 100
LearningRate = 0.0001
ImageWidth, ImageHeight = x.shape[1], x.shape[2]
OutputSize = len(y)

print(f'number of training data: {len(trainData)}')
print(f'number of test data: {len(testData)}')
print(f'number of test data: {len(testData)}')
print(f"image size : {ImageWidth}x{ImageHeight}")
print(f'Output size : {OutputSize}')

trainLoader = DataLoader(dataset = trainData,
	batch_size = BatchSize, shuffle = True)
testLoader = DataLoader(dataset = testData,
	batch_size = BatchSize, shuffle = True)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		w, h = ImageWidth, ImageHeight
		self.conv111 = nn.Conv2d(1, 8, 5, 1, 2)
		self.conv112 = nn.Conv2d(8, 16, 3, 1, 1)
		self.conv12 = nn.Conv2d(1, 16, 1, 1, 0)
		w, h = w//2, h//2
		self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
		w, h = w//2, h//2
		self.dropout1 = nn.Dropout2d(0.1)
		self.dropout2 = nn.Dropout(0.2)
		self.fc1 = nn.Linear(w*h*32, 2048)
		self.fc2 = nn.Linear(2048, OutputSize)
		self.norm8 = nn.BatchNorm2d(8)
		self.norm16 = nn.BatchNorm2d(16)
		self.norm32 = nn.BatchNorm2d(32)
	def forward(self, x):
		a = self.conv111(x)
		a = self.norm8(a)
		a = F.relu(a)
		a = self.conv112(a)
		a = self.norm16(a)
		b = self.conv12(x)
		b = self.norm16(b)
		x = a*0.9+b*0.1
		x = F.relu(x)
		x = self.dropout1(x)
		x = F.max_pool2d(x, 2)
		x = self.conv2(x)
		x = self.norm32(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		return x

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = LearningRate)
#criterion = nn.HuberLoss()
#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
print(model)

for epoch in range(EpochNum):
	model.train()
	for image, target in trainLoader:
		image = image.to(device)
		target = target.to(device)
		optimizer.zero_grad()
		output = model(image)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

	model.eval()
	losses, count = 0, 0
	for image, target in testLoader:
		image = image.to(device)
		target = target.to(device)
		output = model(image)
		prediction = output.data.max(1)[1]
		loss = criterion(output, target)
		losses += loss.item()
		count += 1
	print(f"epoch : {epoch+1}/{EpochNum} loss : {losses/count:.3f}")

	gridImage = Image.new('RGB', size=(ImageWidth*4, ImageHeight*4))
	for i, file in enumerate(random.sample(validationData, 16)):
		transform = transforms.Compose(tfs)
		image = Image.open(file).getchannel(0)
		image = transform(image).reshape(1, 1, ImageHeight, ImageWidth)
		image = image.to(device)
		output = model(image).to('cpu')
		image = outputImage(file, output[0])
		gridImage.paste(image, box=((i%4)*ImageWidth, (i//4)*ImageHeight))
	gridImage.save(f"{SaveImagePath}/{epoch+1:03d}.jpg")
