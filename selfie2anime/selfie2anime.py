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

def toRGB(image):
	rgbImage = Image.new('RGB', image.size)
	rgbImage.paste(image)
	return rgbImage

class ImageDataset(Dataset):
	def __init__(self, root, _transforms=None, unaligned=False, mode='train'):
		self.transform = transforms.Compose(_transforms)
		self.unaligned = unaligned
		if mode=='train':
			self.filesA = sorted(glob.glob(os.path.join(root, 'trainA')+'/*.*'))
			self.filesB = sorted(glob.glob(os.path.join(root, 'trainB')+'/*.*'))
		else:
			self.filesA = sorted(glob.glob(os.path.join(root, 'testA')+'/*.*'))
			self.filesB = sorted(glob.glob(os.path.join(root, 'testB')+'/*.*'))

	def __getitem__(self, index):
		imageA = Image.open(self.filesA[index%len(self.filesA)])
		if self.unaligned:
			imageB = Image.open(random.choice(self.filesB))
		else:
			imageB = Image.open(self.filesB[index%len(self.filesB)])

		if imageA.mode != 'RGB': imageA = toRGB(imageA)
		if imageB.mode != 'RGB': imageB = toRGB(imageA)

		itemA = self.transform(imageA)
		itemB = self.transform(imageB)
		return { 'A':itemA, 'B':itemB }
	def __len__(self):
		return max(len(self.filesA), len(self.filesB))

def initWeightsNormal(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
		if hasattr(m, 'bias') and m.bias != None:
			nn.init.constant_(m.bias.data, 0.0)
	elif classname.find('BatchNorm2d') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0.0)

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

DatasetName = 'selfie2anime'
Channels = 3
ImgHeight, ImgWidth = 256, 256
ResidualBlocks = 9
LearningRate = 0.0002
b1, b2 = 0.5, 0.999
Epochs = 200
InitEpoch = 0
DecayEpoch = 100
LambdaCyc = 10.0
LambdaID = 5.0
BatchSize = 1
SampleInterval = 10
CheckPointInterval = 1

os.makedirs(f"images/{DatasetName}", exist_ok=True)
os.makedirs(f"saved/{DatasetName}", exist_ok=True)

criterionGAN = nn.MSELoss()
criterionCycle = nn.L1Loss()
criterionIdentity = nn.L1Loss()

inputShape = (Channels, ImgHeight, ImgWidth)
GAB = GeneratorResNet(inputShape, ResidualBlocks)
GBA = GeneratorResNet(inputShape, ResidualBlocks)
DA = Discriminator(inputShape)
DB = Discriminator(inputShape)

cuda = torch.cuda.is_available()
mps = torch.backends.mps.is_available()
if cuda:
	device = torch.device('cuda')
	torch.cuda.empty_cache()
elif mps:
	device = torch.device('mps')
else:
	device = torch.device('cpu')
GAB = GAB.to(device)
GBA = GBA.to(device)
DA = DA.to(device)
DB = DB.to(device)
criterionGAN.to(device)
criterionCycle.to(device)
criterionIdentity.to(device)

GAB.apply(initWeightsNormal)
GBA.apply(initWeightsNormal)
DA.apply(initWeightsNormal)
DB.apply(initWeightsNormal)

optimizerG = torch.optim.Adam(
	itertools.chain(GAB.parameters(), GBA.parameters()), lr=LearningRate,
	betas=(b1, b2)
)
optimizerDA = torch.optim.Adam(DA.parameters(), lr=LearningRate, betas=(b1, b2))
optimizerDB = torch.optim.Adam(DB.parameters(), lr=LearningRate, betas=(b1, b2))

class LambdaLR:
	def __init__(self, epochs, offset, decayStartEpoch):
		self.epochs = epochs
		self.offset = offset
		self.decayStartEpoch = decayStartEpoch
	def step(self, epoch):
		return 1.0 - max(0, epoch+self.offset-self.decayStartEpoch)/(self.epochs-self.decayStartEpoch)

lrSchedulerG = torch.optim.lr_scheduler.LambdaLR(
	optimizerG, lr_lambda=LambdaLR(Epochs, InitEpoch, DecayEpoch).step
)
lrSchedulerDA = torch.optim.lr_scheduler.LambdaLR(
	optimizerDA, lr_lambda=LambdaLR(Epochs, InitEpoch, DecayEpoch).step
)
lrSchedulerDB = torch.optim.lr_scheduler.LambdaLR(
	optimizerDB, lr_lambda=LambdaLR(Epochs, InitEpoch, DecayEpoch).step
)

#Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

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

_transforms = [
	transforms.Resize(int(ImgHeight*1.12), Image.Resampling.BICUBIC),
	transforms.RandomCrop((ImgHeight, ImgWidth)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataLoader = DataLoader(
	ImageDataset(f"dataset/{DatasetName}", _transforms=_transforms, unaligned=True),
	batch_size=BatchSize,
	shuffle=True,
)
valDataLoader = DataLoader(
	ImageDataset(f'dataset/{DatasetName}', 
		_transforms=_transforms, unaligned=True, mode='test'),
	batch_size=5,
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

	realA = make_grid(realA, nrow=5, normalize=True)
	realB = make_grid(realB, nrow=5, normalize=True)
	fakeA = make_grid(fakeA, nrow=5, normalize=True)
	fakeB = make_grid(fakeB, nrow=5, normalize=True)

	imageGrid = torch.cat((realA, fakeB, realB, fakeA), 1)
	save_image(imageGrid, f"images/{DatasetName}/{batchesDone}.png", normalize=False)

prevTime = time.time()
for epoch in range(InitEpoch, Epochs):
	for i, batch in enumerate(dataLoader):
		# random learning
		if random.randrange(1000) < 700: continue

		realA = batch['A'].to(device)
		realB = batch['B'].to(device)

		t = np.ones((realA.size(0), *DA.outputShape), dtype=np.float32)
		valid = torch.from_numpy(t).to(device)
		t = np.ones((realA.size(0), *DA.outputShape), dtype=np.float32)
		fake = torch.from_numpy(t).to(device)

		GAB.train()
		GBA.train()
		optimizerG.zero_grad()

		lossIDA = criterionIdentity(GAB(realA), realA)
		lossIDB = criterionIdentity(GAB(realB), realB)
		lossIdentity = (lossIDA + lossIDB)/2

		fakeB = GAB(realA)
		lossGANAB = criterionGAN(DB(fakeB), valid)
		fakeA = GBA(realB)
		lossGANBA = criterionGAN(DA(fakeA), valid)
		lossGAN = (lossGANAB + lossGANBA)/2

		recovA = GAB(fakeB)
		lossCycleA = criterionCycle(recovA, realA)
		recovB = GAB(fakeA)
		lossCycleB = criterionCycle(recovB, realB)
		lossCycle = (lossCycleA + lossCycleB)/2

		lossG = lossGAN + LambdaCyc + lossCycle + LambdaID + lossIdentity

		lossG.backward()
		optimizerG.step()

		optimizerDA.zero_grad()

		lossReal = criterionGAN(DA(realA), valid)

		_fakeA = fakeABuffer.pushAndPop(fakeA)
		lossFake = criterionGAN(DA(_fakeA.detach()), fake)

		lossDA = (lossReal + lossFake)/2
		lossDA.backward()
		optimizerDA.step()

		optimizerDB.zero_grad()

		lossReal = criterionGAN(DB(realB), valid)

		_fakeB = fakeBBuffer.pushAndPop(fakeB)
		lossFake = criterionGAN(DB(_fakeB.detach()), fake)

		lossDB = (lossReal + lossFake)/2
		lossDB.backward()
		optimizerDB.step()
		lossD = (lossDA + lossDB)/2

		batchesDone = epoch*len(dataLoader) + i
		batchesLeft = Epochs * len(dataLoader) - batchesDone
		timeLeft = datetime.timedelta(seconds=batchesLeft*(time.time()-prevTime))
		prevTime = time.time()

		print(f"\r[Epoch {epoch}/{Epochs}]",
			f"[Batch {i}/{len(dataLoader)}]",
			f"D loss:{lossD.item():.2f},",
			f"adv: {lossG.item():.2f},",
			f"cycle: {lossCycle.item():.2f},",
			f"identity: {lossIdentity.item():.2f}",
			f"ETA: {timeLeft}")

		if batchesDone%SampleInterval == 0:
			sampleImages(batchesDone)

	lrSchedulerG.step()
	lrSchedulerDA.step()
	lrSchedulerDB.step()

	if CheckPointInterval != -1 and epoch%CheckPointInterval == 0:
		torch.save(GAB.state_dict(), f"saved_models/{DatasetName}/GAB{epoch}.pth")
		torch.save(GBA.state_dict(), f"saved_models/{DatasetName}/GBA{epoch}.pth")
		torch.save(DA.state_dict(), f"saved_models/{DatasetName}/DA{epoch}.pth")
		torch.save(DB.state_dict(), f"saved_models/{DatasetName}/DB{epoch}.pth")

