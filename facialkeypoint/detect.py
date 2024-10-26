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

os.makedirs(SaveImagePath, exist_ok=True)

if torch.cuda.is_available():
    device='cuda'
    useBf16=True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
elif torch.backends.mps.is_available():
    device='mps'
    useBf16=False
    torch.mps.empty_cache()
else:
    device='cpu'
    useBf16=False
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
df = df[[
	'left_eye_center_x','left_eye_center_y',
	'right_eye_center_x','right_eye_center_y',
	'nose_tip_x','nose_tip_y',
	'mouth_left_corner_x','mouth_left_corner_y',
	'mouth_right_corner_x','mouth_right_corner_y',
	'mouth_center_top_lip_x','mouth_center_top_lip_y',
	'mouth_center_bottom_lip_x','mouth_center_bottom_lip_y',
]]
df['image']=np.array([f"{TrainImagePath}/{idx}.jpg" for idx in range(len(df))])
df = df.dropna()
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
print(f"image size : {ImageWidth}x{ImageHeight}")
print(f'Output size : {OutputSize}')

trainLoader = DataLoader(
	dataset = trainData,
	batch_size = BatchSize,
	shuffle = True,
)
testLoader = DataLoader(
	dataset = testData,
	batch_size = BatchSize, 
	shuffle = True
)

# ResidualBlock class
class ResidualBlock(nn.Module):
	def __init__(self, inFeatures):
		super().__init__()
		self.block = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(inFeatures, inFeatures, 3),
			nn.InstanceNorm2d(inFeatures),
			nn.LeakyReLU(0.2, inplace=True),
			nn.ReflectionPad2d(1),
			nn.Conv2d(inFeatures, inFeatures, 3),
			nn.InstanceNorm2d(inFeatures),
		)
	def forward(self, x):
		return x + self.block(x)

class ResNetKeypointModel(nn.Module):
	def __init__(self, outputSize):
		super().__init__()
		model = [
			*self.convBlock(1, 8, False),
			*self.convBlock(8, 16),
			*self.convBlock(16, 64),
		]
		for _ in range(2):
			model += [ResidualBlock(64)]
		model += [nn.Flatten()]
		model += [nn.Linear(ImageWidth*ImageHeight*64, outputSize)]
		self.model = nn.Sequential(*model)

	@staticmethod
	def convBlock(inFeatures, outFeatures, normalize=True):
		layers = [nn.Conv2d(inFeatures, outFeatures, 3, 1, 1)]
		if normalize: layers += [nn.InstanceNorm2d(outFeatures)]
		layers += [nn.LeakyReLU(0.2, inplace=True)]
		return layers

	def forward(self, x):
		return self.model(x)

# 모델 인스턴스화 및 손실 함수 설정
model = ResNetKeypointModel(OutputSize).to(device)
optimizer = optim.Adam(model.parameters(), lr=LearningRate)
#criterion = nn.HuberLoss()
criterion = nn.MSELoss()
#criterion = nn.SmoothL1Loss()
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
