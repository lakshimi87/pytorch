import os
import numpy as np
import pandas as pd
import random
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from konlpy.tag import *

if torch.cuda.is_available():
	device = torch.device('cuda')
#elif torch.backends.mps.is_available():
#	device = torch.device('mps')
else:
	device = torch.device('cpu')
print(device)

TokenPad, TokenSOS, TokenEOS = 0, 1, 2

class WordVocab():
	def __init__(self):
		self.word2index = { '<PAD>': TokenPad, '<SOS>': TokenSOS, '<EOS>': TokenEOS, }
		self.word2count = {}
		self.index2word = { TokenPad: '<PAD>', TokenSOS: '<SOS>', TokenEOS: '<EOS>' }
		self.numWords = 3  # PAD, SOS, EOS 포함

	def addSentence(self, sentence):
		for word in sentence: self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.numWords
			self.word2count[word] = 1
			self.index2word[self.numWords] = word
			self.numWords += 1
		else:
			self.word2count[word] += 1

class TextDataset(Dataset):
	def __init__(self, csvPath, min_length=3, maxLength=32):
		super().__init__()

		self.tagger = Hannanum()
		self.maxLength = maxLength

		df = pd.read_csv(csvPath)

		self.srcs, self.targets = [], []

		# 단어 사전 생성
		self.wordvocab = WordVocab()

		for _, row in df.iterrows():
			src = row['Q']
			tgt = row['A']

			# 한글 전처리
			src = self.cleanText(src)
			tgt = self.cleanText(tgt)

			if len(src) > min_length and len(tgt) > min_length:
				# 최소 길이를 넘어가는 문장의 단어만 추가
				self.wordvocab.addSentence(src)
				self.wordvocab.addSentence(tgt)
				self.srcs.append(torch.tensor(self.convertToSequence(src)))
				self.targets.append(torch.tensor(self.convertToSequence(tgt)))

	def cleanText(self, sentence):
		sentence = re.sub(r'[^A-Za-z0-9가-힣]+', r' ', sentence)
		sentence = self.tagger.morphs(sentence)
		return sentence

	def convertToSequence(self, sentence):
		seqLen = min(len(sentence), self.maxLength-1)
		seq = [self.wordvocab.word2index[sentence[i]] for i in range(seqLen)]
		seq.append(TokenEOS)
		for i in range(seqLen+1, self.maxLength):
			seq.append(TokenPad)
		return seq

	def __getitem__(self, idx):
		return self.srcs[idx], self.targets[idx]

	def __len__(self):
		return len(self.srcs)

dataset = TextDataset('data/ChatbotData.csv')

# train and test dataset split
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

class Encoder(nn.Module):
	def __init__(self, numVocabs, hiddenSize, embeddingDim, numLayers=1, dropout=0.2):
		super().__init__()
		self.numVocabs = numVocabs
		# 임베딩 레이어 정의 (number of vocabs, embedding dimension)
		self.embedding = nn.Embedding(numVocabs, embeddingDim)
		self.dropout = nn.Dropout(dropout)
		# RNN (embedding dimension)
		self.rnn = nn.GRU(embeddingDim, hiddenSize, numLayers)
		
	def forward(self, x):
		x = self.embedding(x).permute(1, 0, 2)
		x = self.dropout(x)
		_, hidden = self.rnn(x)
		return hidden

class Decoder(nn.Module):
	def __init__(self, numVocabs, hiddenSize, embeddingDim, numLayers=1, dropout=0.2):
		super().__init__()
		# 단어사전 개수
		self.numVocabs = numVocabs
		self.embedding = nn.Embedding(numVocabs, embeddingDim)
		self.dropout = nn.Dropout(dropout)
		self.rnn = nn.GRU(embeddingDim, hiddenSize, numLayers)
		self.fc = nn.Linear(hiddenSize, numVocabs)

	def forward(self, x, hiddenState):
		x = x.unsqueeze(0) # (1, batchSize) 로 변환
		x = self.embedding(x)
		x = self.dropout(x)
		output, hidden = self.rnn(x, hiddenState)
		output = self.fc(output.squeeze(0))
		# (sequence_length, batchSize, hiddenSize(32) x bidirectional(1))
		return output, hidden

class Seq2Seq(nn.Module):
	def __init__(self, numVocabs, hiddenSize, embeddingDim, numLayers = 1, dropout=0.5):
		super().__init__()
		self.encoder = Encoder(numVocabs, hiddenSize, embeddingDim, numLayers).to(device)
		self.decoder = Decoder(numVocabs, hiddenSize, embeddingDim, numLayers).to(device)
		self.numVocabs = numVocabs

	def forward(self, inputs):
		# inputs : (batchSize, seqLength)
		batchSize, seqLength = inputs.shape

		# 리턴할 예측된 outputs를 저장할 임시 변수
		# (seqLength, batchSize, self.numVocabs)
		PredictedOutputs = torch.zeros(seqLength, batchSize, self.numVocabs).to(device)

		# 인코더에 입력 데이터 주입, encoder output은 버리고 hidden state 만 살립니다.
		# 여기서 hidden state가 디코더에 주입할 context vector 입니다.
		# (Bidirectional(1) x number of layers(1), batchSize, hiddenSize)
		hidden = self.encoder(inputs)

		# (batchSize) shape의 SOS TOKEN으로 채워진 디코더 입력 생성
		input = torch.full((batchSize, ), TokenSOS, device=device)
		#print("seq2seq input:", input.shape)

		# 순회하면서 출력 단어를 생성합니다.
		# 0번째는 TokenSOS 이 위치하므로, 1번째 인덱스부터 순회합니다.
		for t in range(0, seqLength):
			# input : 디코더 입력 (batchSize)
			# output: (batchSize, numVocabs)
			# hidden: (Bidirectional(1) x number of layers(1), batchSize, hiddenSize)
			#     context vector와 동일 shape
			output, hidden = self.decoder(input, hidden)

			PredictedOutputs[t] = output.reshape(batchSize, self.numVocabs)

			input = output.argmax(1)

		# return: (batchSize, sequence_length, numVocabs)로 변경
		return PredictedOutputs.permute(1, 0, 2)

NumVocabs = dataset.wordvocab.numWords
HiddenSize = 1024
EmbeddingDim = 512
NumLayers = 1
DropOutRate = 0.5
print(f'numVocabs: {NumVocabs}\n======================')

# Seq2Seq 생성
model = Seq2Seq(NumVocabs, HiddenSize, EmbeddingDim, NumLayers, DropOutRate)
print(model)

"""
# test
lossFn = nn.CrossEntropyLoss()
for _ in range(1):
	x, y = next(iter(train_loader))
	output = model(x.to(device))
	print(x.shape, y.shape, output.shape)
	output = output.reshape(-1, output.size(2))
	y = y.view(-1)
	print(x.shape, y.shape, output.shape)
	loss = lossFn(output, y.to(device))
	print(f"loss = {loss.item()}")
quit()
"""

class EarlyStopping:
	def __init__(self, patience=3, delta=0.0, mode='min', verbose=True):
		"""
		patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
		delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
		mode	 (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
		verbose (bool): 메시지 출력. default: True
		"""
		self.earlyStop = False
		self.patience = patience
		self.verbose = verbose
		self.counter = 0

		self.best_score = np.Inf if mode == 'min' else 0
		self.mode = mode
		self.delta = delta

	def __call__(self, score):
		if self.best_score is None:
			self.best_score = score
			self.counter = 0
		elif self.mode == 'min':
			if score < (self.best_score - self.delta):
				self.counter = 0
				self.best_score = score
				if self.verbose:
					print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
			else:
				self.counter += 1
				if self.verbose:
					print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
						  f'Best: {self.best_score:.5f}' \
						  f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')

		elif self.mode == 'max':
			if score > (self.best_score + self.delta):
				self.counter = 0
				self.best_score = score
				if self.verbose:
					print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
			else:
				self.counter += 1
				if self.verbose:
					print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
						  f'Best: {self.best_score:.5f}' \
						  f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')

		if self.counter >= self.patience:
			if self.verbose:
				print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
			# Early Stop
			self.earlyStop = True
		else:
			# Continue
			self.earlyStop = False

LR = 1e-2
optimizer = optim.Adam(model.parameters(), lr=LR)
lossFunction = nn.CrossEntropyLoss(ignore_index=TokenPad)

es = EarlyStopping(patience=10, delta=0.001, mode='min', verbose=True)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
												 mode='min', 
												 factor=0.8, 
												 patience=2,
												 threshold_mode='abs',
												 min_lr=1e-8, 
												 verbose=True)

def train(model, dataLoader, optimizer, lossFunction, device):
	model.train()
	runningLoss = 0
	for x, y in dataLoader:
		x, y = x.to(device), y.to(device)

		optimizer.zero_grad()
		
		# output: (batchSize, sequence_length, numVocabs)
		output = model(x)
		output_dim = output.size(2)
		
		# 1번 index 부터 슬라이싱한 이유는 0번 index가 SOS TOKEN 이기 때문
		# (batchSize*sequence_length, numVocabs) 로 변경
		output = output.reshape(-1, output_dim)
		
		# (batchSize*sequence_length) 로 변경
		y = y.view(-1)
		
		# Loss 계산
		loss = lossFunction(output, y)
		loss.backward()
		optimizer.step()
		
		runningLoss += loss.item() * x.size(0)
	return runningLoss / len(dataLoader)

def evaluate(model, dataLoader, lossFunction, device):
	model.eval()
	evalLoss = 0
	with torch.no_grad():
		for x, y in dataLoader:
			x, y = x.to(device), y.to(device)
			output = model(x)
			output_dim = output.size(2)
			output = output.reshape(-1, output_dim)
			y = y.view(-1)
			loss = lossFunction(output, y)
			evalLoss += loss.item() * x.size(0)
	return evalLoss / len(dataLoader)

def convertToSentence(sequences, index2word):
	outputs = []
	for p in sequences:
		word = index2word[p]
		if p not in [TokenSOS, TokenEOS, TokenPad]:
			outputs.append(word)
		if word == TokenEOS:
			break
	return ' '.join(outputs)

def random_evaluation(model, dataset, index2word, device, n=10):
	n_samples = len(dataset)
	indices = list(range(n_samples))
	np.random.shuffle(indices)	  # Shuffle
	sampled_indices = indices[:n]   # Sampling N indices
	
	# 샘플링한 데이터를 기반으로 DataLoader 생성
	sampler = SubsetRandomSampler(sampled_indices)
	sampledLoader = DataLoader(dataset, batch_size=10, sampler=sampler)
	
	model.eval()
	with torch.no_grad():
		for x, y in sampledLoader:
			x, y = x.to(device), y.to(device)		
			output = model(x)
			# output: (number of samples, sequence_length, numVocabs)
			
			preds = output.detach().cpu().numpy()
			x = x.detach().cpu().numpy()
			y = y.detach().cpu().numpy()
			
			for i in range(n):
				print(f'질문   : {convertToSentence(x[i], index2word)}')
				print(f'답변   : {convertToSentence(y[i], index2word)}')
				print(f'예측답변: {convertToSentence(preds[i].argmax(1), index2word)}')
				print('==='*10)

NumEpochs = 80
SavePath = 'models/seq2seq-chatbot-kor.pt'

bestLoss = np.inf
for epoch in range(NumEpochs):
	loss = train(model, train_loader, optimizer, lossFunction, device)
	evalLoss = evaluate(model, test_loader, lossFunction, device)
	
	if evalLoss < bestLoss:
		bestLoss = evalLoss
		torch.save(model.state_dict(), SavePath)
	
	print(f'epoch: {epoch+1}, loss: {loss:.4f}, evalLoss: {evalLoss:.4f}')
	
	# Early Stop
	es(loss)
	if es.earlyStop: break
	
	# Scheduler
	scheduler.step(evalLoss)
				   
model.load_state_dict(torch.load(SavePath))
torch.save(model.state_dict(), f'models/seq2seq-chatbot-kor-{bestLoss:.4f}.pt')

model.load_state_dict(torch.load(SavePath))
random_evaluation(model, test_dataset, dataset.wordvocab.index2word, device)

