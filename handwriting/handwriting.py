"""
Title: hand writing number detection
Description: hand writing number detection
Author: Aubrey Choi
Date: 2024-07-10
Version: 1.2
License: MIT License
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

isCuda = torch.cuda.is_available()
device = torch.device('cuda' if isCuda else 'cpu')

print(f'Current cuda device is {device}')

BatchSize = 50
EpochNum = 10
LearningRate = 0.0001

trainData = datasets.MNIST(root = './data', train = True, download = True,
	transform = transforms.ToTensor())
testData = datasets.MNIST(root = './data', train = False, download = True,
	transform = transforms.ToTensor())

print(f'number of training data: {len(trainData)}')
print(f'number of test data: {len(testData)}')

#image, label = trainData[0]
#plt.imshow(image.squeeze().numpy(), cmap = 'gray')
#plt.title(f'label : {label}')
#plt.show()

trainLoader = torch.utils.data.DataLoader(dataset = trainData,
	batch_size = BatchSize, shuffle = True)
testLoader = torch.utils.data.DataLoader(dataset = testData,
	batch_size = BatchSize, shuffle = True)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 8, 3, 1)
		self.conv2 = nn.Conv2d(8, 16, 3, 1)
		self.dropout = nn.Dropout(0.5)
		self.fc1 = nn.Linear(2304, 256)
		self.fc2 = nn.Linear(256, 10)
	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim = 1)
		return output
	def getFeatures(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		return x

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr = LearningRate)
criterion = nn.CrossEntropyLoss()
print(model)

model.train()
step = 0
for epoch in range(EpochNum):
	for data, target in trainLoader:
		data = data.to(device)
		target = target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		step += 1
		if step%1000 == 0: print(f'{step:6d} {loss.item():.3f}')

model.eval()
correct = 0
for data, target in testLoader:
	data = data.to(device)
	target = target.to(device)
	output = model(data)
	prediction = output.data.max(1)[1]
	correct += prediction.eq(target.data).sum()

print(f'Accuracy: {100*correct.item()/len(testLoader.dataset):.2f}%')

