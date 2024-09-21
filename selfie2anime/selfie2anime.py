"""
Title: selfie to anime
Description: selfie to anime with GAN
Author: Aubrey Choi
Date: 2024-07-10
Version: 1.2
License: MIT License
"""

import glob
import random
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import itertools
import datetime
import time
from torchvision.utils import save_image, make_grid
from torchvision import datasets

# H/W acceleration
if torch.cuda.is_available():
	device = torch.device('cuda')
	torch.cuda.empty_cache()
elif torch.backends.mps.is_available():
	device = torch.device('mps')
else:
	device = torch.device('cpu')

def loadImages(files):
	images = []
	for fileName in files:
		image = Image.open(fileName)
		if image.mode != 'RGB': image = image.convert('RGB')
		images.append(image.copy())
		image.close()
	return images

# image dataset class
class ImageDataset(Dataset):
	def __init__(self, root, trans, unaligned=False, mode='train'):
		self.transform = transforms.Compose(trans)
		self.unaligned = unaligned
		filesA = sorted(glob.glob(os.path.join(root, f'{mode}A')+'/*.*'))
		filesB = sorted(glob.glob(os.path.join(root, f'{mode}B')+'/*.*'))
		self.lenA = len(filesA)
		self.lenB = len(filesB)
		self.imagesA = loadImages(filesA)
		self.imagesB = loadImages(filesB)

	def __getitem__(self, index):
		idx = index%self.lenA
		imageA = self.imagesA[idx]
		idx = random.randrange(self.lenB) if self.unaligned else index%self.lenB
		imageB = self.imagesB[idx]
		itemA = self.transform(imageA)
		itemB = self.transform(imageB)
		return { 'A':itemA, 'B':itemB }

	def __len__(self):
		return self.lenA

# initialize weights by normal
def initWeightsNormal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
		if hasattr(m, 'bias') and m.bias != None:
			nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('BatchNorm2d') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0.0)

# ResidualBlock class
class ResidualBlock(nn.Module):
	def __init__(self, inFeatures):
		super(ResidualBlock, self).__init__()
		self.block = nn.Sequential(
			nn.ReflectionPad2d(1),
			nn.Conv2d(inFeatures, inFeatures, 3),
			nn.InstanceNorm2d(inFeatures),
			nn.ReLU(inplace=True),
			nn.ReflectionPad2d(1),
			nn.Conv2d(inFeatures, inFeatures, 3),
			nn.InstanceNorm2d(inFeatures),
		)
	def forward(self, x):
		return x + self.block(x.to(device))

# GeneratorResNet class
class GeneratorResNet(nn.Module):
	def __init__(self, inputShape, numResidualBlocks):
		super(GeneratorResNet, self).__init__()
		channels = inputShape[0]

		outFeatures = 64
		model = [
			nn.ReflectionPad2d(channels),
			nn.Conv2d(channels, outFeatures, 7),
			nn.InstanceNorm2d(outFeatures),
			nn.ReLU(inplace=True),
		]
		for _ in range(2):
			inFeatures, outFeatures = outFeatures, outFeatures*2
			model += [
				nn.Conv2d(inFeatures, outFeatures, 3, stride=2, padding=1),
				nn.InstanceNorm2d(outFeatures),
				nn.ReLU(inplace=True),
			]

		for _ in range(numResidualBlocks):
			model += [ResidualBlock(outFeatures)]

		for _ in range(2):
			inFeatures, outFeatures = outFeatures, outFeatures//2
			model += [
				nn.Upsample(scale_factor=2),
				nn.Conv2d(inFeatures, outFeatures, 3, stride=1, padding=1),
				nn.InstanceNorm2d(outFeatures),
				nn.ReLU(inplace=True),
			]

		model += [nn.ReflectionPad2d(channels), 
			nn.Conv2d(outFeatures, channels, 7),
			nn.Tanh()]
		self.model = nn.Sequential(*model)
	
	def forward(self, x):
		return self.model(x.to(device))

# Discriminator class
class Discriminator(nn.Module):
	def __init__(self, inputShape):
		super(Discriminator, self).__init__()
		channels, height, width = inputShape

		self.outputShape = (1, height//2**4, width//2**4)

		self.model = nn.Sequential(
			*Discriminator.discriminatorBlock(channels, 64, False),
			*Discriminator.discriminatorBlock(64, 128),
			*Discriminator.discriminatorBlock(128, 256),
			*Discriminator.discriminatorBlock(256, 512),
			nn.ZeroPad2d((1, 0, 1, 0)),
			nn.Conv2d(512, 1, 4, padding=1)
		)

	def discriminatorBlock(inFilters, outFilters, normalize=True):
		layers = [nn.Conv2d(inFilters, outFilters, 4, stride=2, padding=1)]
		if normalize: layers += [nn.InstanceNorm2d(outFilters)]
		layers += [nn.LeakyReLU(0.2, inplace=True)]
		return layers

	def forward(self, img):
		return self.model(img.to(device))

Channels = 3
ImgHeight, ImgWidth = 256, 256
ResidualBlocks = 9
LearningRate = 0.0002
BetaTuple = (0.5, 0.999)
Epochs = 300
InitEpoch = 0
DecayEpoch = 100
LambdaCyc = 10.0
LambdaID = 5.0
BatchSize = 2
SampleInterval = 1000
CheckPointInterval = 1

os.makedirs(f"images", exist_ok=True)
os.makedirs(f"saved_models", exist_ok=True)

criterionGAN = nn.MSELoss()
criterionCycle = nn.L1Loss()
criterionIdentity = nn.L1Loss()

inputShape = (Channels, ImgHeight, ImgWidth)
GAB = GeneratorResNet(inputShape, ResidualBlocks).to(device)
GBA = GeneratorResNet(inputShape, ResidualBlocks).to(device)
DA = Discriminator(inputShape).to(device)
DB = Discriminator(inputShape).to(device)

criterionGAN.to(device)
criterionCycle.to(device)
criterionIdentity.to(device)

GAB.apply(initWeightsNormal)
GBA.apply(initWeightsNormal)
DA.apply(initWeightsNormal)
DB.apply(initWeightsNormal)

optimizerG = torch.optim.Adam(
	itertools.chain(GAB.parameters(), GBA.parameters()), lr=LearningRate,
	betas=BetaTuple
)
optimizerDA = torch.optim.Adam(
	DA.parameters(), lr=LearningRate, betas=BetaTuple)
optimizerDB = torch.optim.Adam(
	DB.parameters(), lr=LearningRate, betas=BetaTuple)

class LambdaLR:
	def __init__(self, epochs, offset, decayStartEpoch):
		self.epochs = epochs
		self.offset = offset
		self.decayStartEpoch = decayStartEpoch
	def step(self, epoch):
		return 1.0 - max(0, epoch+self.offset-self.decayStartEpoch)/(self.epochs-self.decayStartEpoch)

for d in os.listdir("saved_models/"):
	if not d.startswith("GAB") or not d.endswith(".pth"): continue
	epoch = int(d[3:-4])
	if epoch >= InitEpoch: InitEpoch = epoch+1

if InitEpoch > 0:
	print(f"Load epoch {InitEpoch}....")
	path = "saved_models"
	GAB.load_state_dict(torch.load(f"{path}/GAB{InitEpoch-1}.pth", map_location=device, weights_only=True))
	GBA.load_state_dict(torch.load(f"{path}/GBA{InitEpoch-1}.pth", map_location=device, weights_only=True))
	DA.load_state_dict(torch.load(f"{path}/DA{InitEpoch-1}.pth", map_location=device, weights_only=True))
	DB.load_state_dict(torch.load(f"{path}/DB{InitEpoch-1}.pth", map_location=device, weights_only=True))

lrSchedulerG = torch.optim.lr_scheduler.LambdaLR(
	optimizerG, lr_lambda=LambdaLR(Epochs, InitEpoch, DecayEpoch).step
)
lrSchedulerDA = torch.optim.lr_scheduler.LambdaLR(
	optimizerDA, lr_lambda=LambdaLR(Epochs, InitEpoch, DecayEpoch).step
)
lrSchedulerDB = torch.optim.lr_scheduler.LambdaLR(
	optimizerDB, lr_lambda=LambdaLR(Epochs, InitEpoch, DecayEpoch).step
)

# ReplayBuffer class
class ReplayBuffer:
	def __init__(self, maxSize=50):
		self.maxSize = maxSize
		self.data = []
	def pushAndPop(self, data):
		toReturn = []
		for element in data.data:
			element = torch.unsqueeze(element, 0)
			if len(self.data) < self.maxSize:
				self.data.append(element)
				toReturn.append(element)
			elif random.uniform(0, 1) > 0.5:
				i = random.randrange(0, self.maxSize)
				toReturn.append(self.data[i].clone())
				self.data[i] = element
			else:
				toReturn.append(element)
		return torch.cat(toReturn)

fakeABuffer = ReplayBuffer()
fakeBBuffer = ReplayBuffer()

myTransforms = [
	transforms.Resize(int(ImgHeight*1.12), Image.Resampling.BICUBIC),
	transforms.RandomCrop((ImgHeight, ImgWidth)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataLoader = DataLoader(
	ImageDataset(f"dataset", myTransforms, unaligned=True),
	batch_size=BatchSize,
	shuffle=True,
)

valDataLoader = DataLoader(
	ImageDataset(f'dataset', 
		myTransforms, unaligned=True, mode='test'),
	batch_size=7,
	shuffle=True,
)

def sampleImages(batchesDone):
	imgs = next(iter(valDataLoader))
	GAB.eval()
	GBA.eval()
	realA = imgs['A'].to(device)
	fakeB = GAB(realA)
	realB = imgs['B'].to(device)
	fakeA = GBA(realB)

	realA = make_grid(realA, nrow=7, normalize=True)
	realB = make_grid(realB, nrow=7, normalize=True)
	fakeA = make_grid(fakeA, nrow=7, normalize=True)
	fakeB = make_grid(fakeB, nrow=7, normalize=True)

	imageGrid = torch.cat((realA, fakeB, realB, fakeA), 1)
	save_image(imageGrid, f"images/{batchesDone:06}.png", normalize=False)
	del(imgs)
	del(imageGrid)

timeStamp, epochSize = time.time(), len(dataLoader)*BatchSize
for epoch in range(InitEpoch, Epochs):
	for i, batch in enumerate(dataLoader):
		# get images from data set
		realA = batch['A'].to(device)
		realB = batch['B'].to(device)

		# make valid and fake target
		t = np.ones((realA.size(0), *DA.outputShape), dtype=np.float32)
		valid = torch.from_numpy(t).to(device)
		t = np.zeros((realA.size(0), *DA.outputShape), dtype=np.float32)
		fake = torch.from_numpy(t).to(device)

		GAB.train()
		GBA.train()

		optimizerG.zero_grad()

		lossIDA = criterionIdentity(GBA(realA), realA)
		lossIDB = criterionIdentity(GAB(realB), realB)
		lossIdentity = (lossIDA + lossIDB)/2

		fakeB = GAB(realA)
		fakeA = GBA(realB)
		lossGANAB = criterionGAN(DB(fakeB), valid)
		lossGANBA = criterionGAN(DA(fakeA), valid)
		lossGAN = (lossGANAB + lossGANBA)/2

		recovA = GBA(fakeB)
		recovB = GAB(fakeA)
		lossCycleA = criterionCycle(recovA, realA)
		lossCycleB = criterionCycle(recovB, realB)
		lossCycle = (lossCycleA + lossCycleB)/2

		lossG = lossGAN + LambdaCyc + lossCycle + LambdaID + lossIdentity

		lossG.backward()
		optimizerG.step()

		optimizerDA.zero_grad()

		lossReal = criterionGAN(DA(realA), valid)
		_fakeA = fakeABuffer.pushAndPop(fakeA.detach())
		lossFake = criterionGAN(DA(_fakeA), fake)

		lossDA = (lossReal + lossFake)/2
		lossDA.backward()
		optimizerDA.step()

		optimizerDB.zero_grad()

		lossReal = criterionGAN(DB(realB), valid)
		_fakeB = fakeBBuffer.pushAndPop(fakeB.detach())
		lossFake = criterionGAN(DB(_fakeB), fake)

		lossDB = (lossReal + lossFake)/2
		lossDB.backward()
		optimizerDB.step()

		lossD = (lossDA + lossDB)/2

		batchesDone = epoch*epochSize + (i+1)*BatchSize
		batchesLeft = Epochs*epochSize - batchesDone
		deltaTime = (time.time()-timeStamp)/((epoch-InitEpoch)*epochSize+(i+1)*BatchSize)
		timeLeft = datetime.timedelta(seconds=batchesLeft*deltaTime)

		print(f"\r[Epoch {epoch}/{Epochs}]",
			f"[Batch {i}/{len(dataLoader)}]",
			f"D loss:{lossD.item():.2f},",
			f"adv: {lossG.item():.2f},",
			f"cycle: {lossCycle.item():.2f},",
			f"identity: {lossIdentity.item():.2f}",
			f"ETA: {timeLeft}", flush=True)

		if batchesDone%SampleInterval == 0:
			sampleImages(batchesDone)

	lrSchedulerG.step()
	lrSchedulerDA.step()
	lrSchedulerDB.step()

	if CheckPointInterval != -1 and epoch%CheckPointInterval == 0:
		path = "saved_models"
		torch.save(GAB.state_dict(), f"{path}/GAB{epoch}.pth")
		torch.save(GBA.state_dict(), f"{path}/GBA{epoch}.pth")
		torch.save(DA.state_dict(), f"{path}/DA{epoch}.pth")
		torch.save(DB.state_dict(), f"{path}/DB{epoch}.pth")

