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
import glob
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

TrainImagePath = "images/train_images"
TestImagePath = "images/test_images"
SaveImagePath = "saves"
CsvFilePath = "training.csv"

isCuda = torch.cuda.is_available()
isMps = torch.backends.mps.is_available()
device = torch.device('cuda' if isCuda else 'mps' if isMps else 'cpu')

class ImageDataset(Dataset):
    def __init__(self, dataframe, tfs):
        self.dataframe = dataframe
        self.transforms = transforms.Compose(tfs)
    def __getitem__(self, index):
        v = self.dataframe.iloc[index]
        image = Image.open(v['image']).getchannel(0)
        image = self.transforms(image)
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
        draw.ellipse(((v[i]-1, v[i+1]-1), (v[i]+1, v[i+1]+1)))
    return image

def showImage(image, label):
    plt.imshow(image)
    plt.title(label)
    plt.show()

def getFileList(root):
    return glob.glob(root+"/*.jpg")

# read csv file
df = pd.read_csv(CsvFilePath)
df = df.iloc[:, :4]
df['image'] = np.array([
    f"{TrainImagePath}/{idx}.jpg" for idx in range(len(df))])
print(f"Total data count {len(df)}")

#결손치 제거
df = df.dropna(axis = 0)

#학습데이터와 테스트데이터하고 분리
xTrain, xTest = train_test_split(df, test_size = 0.2)
xTrain = xTrain.reset_index(drop=True)
xTest.reset_index(drop=True, inplace=True)

#임시 출력
print(xTrain.head())
print(xTest.head(10))

# 트랜스품 정의(Tensor로 바꾸는 것, 정규화
tfs = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )),
]

# 데이터셋 만들기
trainData = ImageDataset(xTrain, tfs)
testData = ImageDataset(xTest, tfs)

#데이터셋 잘 동작하는지 테스트 코드
"""
x, y = trainData[0]                 
label = trainData.getLabel(0)
image = outputImage(label, y)                 
showImage(image, label)
"""

# 검증 데이터 리스트 얻어오기
validationData = getFileList(TestImagePath)

# 인공지능 파라미터들 설정
BatchSize = 16
EpochNum = 100
LearningRate = 0.0001

#이미지 정보
x, y = trainData[0]
ImageWidth, ImageHeight = x.shape[1], x.shape[2]
OutputSize = len(y)
print(f"Image width = {ImageWidth}, height = {ImageHeight}")
print(f"Ouput size = {OutputSize}")

# FacialNetwork 모델 클래스 정의
class FacialNetwork(nn.Module):                 
    def __init__(self):
        super().__init__()
        w, h = ImageWidth, ImageHeight
        self.conv1 = nn.Conv2d(1, 4, 5)
        w, h = w-4, h-4    # filter size에 의해서 줄어드는 것
        #w, h = w//2, h//2  # maxplooling kernel size에 의해서 줄어들는것
        self.conv2 = nn.Conv2d(4, 8, 3)
        w, h = w-2, h-2
        w, h = w//2, h//2

        self.fc1 = nn.Linear(w*h*8, 1024)
        self.fc2 = nn.Linear(1024, OutputSize)

        self.norm4 = nn.BatchNorm2d(4)
        self.norm8 = nn.BatchNorm2d(8)
        self.norm16 = nn.BatchNorm2d(16)
        self.norm32 = nn.BatchNorm2d(32)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.norm4(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        x = self.drop1(x)
        x = self.conv2(x)
        #x = self.norm8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x

# FacialNetwork 모델 생성
model = FacialNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=LearningRate)
#criterion = nn.MSELoss()
#criterion = nn.SmoothL1Loss()
criterion = nn.HuberLoss()
print(model)

# 데이터 로더 만들기
trainLoader = DataLoader(trainData, batch_size=BatchSize)
testLoader = DataLoader(testData, batch_size=BatchSize)

# 학습과 검증
for epoch in range(EpochNum):
    model.train()
    for image, target in trainLoader: #차원이 하나 증가한다.
        image = image.to(device)
        target = target.to(device)
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
        loss = criterion(output, target)
        losses += loss.item()
        count += 1
    print(f"{epoch+1}/{EpochNum} loss : {losses/count:.3f}")
    
    gridImage = Image.new('RGB', size=(ImageWidth*2, ImageHeight*2))
    for i, file in enumerate(random.sample(validationData, 4)):
        transform = transforms.Compose(tfs)
        image = Image.open(file).getchannel(0)
        image = transform(image).reshape(1, 1, ImageWidth, ImageHeight)
        image = image.to(device)
        output = model(image)
        image = outputImage(file, output[0].to(torch.device('cpu')))
        gridImage.paste(image, box=((i%2)*ImageWidth, (i//2)*ImageHeight))
    gridImage.save(f"{SaveImagePath}/{epoch+1:03d}.jpg")
        







                 
