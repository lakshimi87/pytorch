"""
Title: learn simple ANN
Description: reversi simple ANN learning
Author: Aubrey Choi
Date: 2024-07-10
Version: 1.2
License: MIT License
"""

import math	 # 수학 관련된 라이브러리 모듈
import sys
import random   # 난수 발생용 라이브러리 모듈
import pygame   # pygame을 위한 라이브러리 모듈
import time	 # 시간을 측정하기 위한 모듈
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game import ReversiGame, WhiteTurn, BlackTurn

Path = "./reversi-simple.pth"
EpsilonDecay = 0.99
SaveInterval = 20
TotalGames = 200
LearningRate = 0.1

class SimpleANN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(8*8, 1)
	def forward(self, x):
		x = self.fc1(x)
		x = F.tanh(x)
		return x

def convertToInput(board):
	x = [ 1.0 if c == 1 else -1.0 if c == 2 else 0.0 for c in board ]
	x = np.array(x).reshape(1, 64)
	return torch.FloatTensor(x)

# random하게 두기
def waitRandom(board, turn):
	hints = [ i for i in range(64) if board[i] == 0 ]
	idx = random.randrange(len(hints))
	return hints[idx]

# 컴퓨터 두기
def waitAI(board, turn):
	hints = [ i for i in range(64) if board[i] == 0 ]
	if len(hints) == 0: return -1
	if len(hints) == 1: return hints[0]
	if random.random() < epsilon: return random.choice(hints)
	model.eval()
	if turn == WhiteTurn:
		p, maxp = hints[0], -1
		for hint in hints:
			pboard = board[:]
			ReversiGame.prerun(pboard, hint, turn)
			out = model(convertToInput(board)).detach().squeeze().numpy()
			if out > maxp: p, maxp = hint, out
		return p
	else:
		p, minp = hints[0], -1
		for hint in hints:
			pboard = board[:]
			ReversiGame.prerun(pboard, hint, turn)
			out = model(convertToInput(board)).detach().squeeze().numpy()
			if out < minp: p, minp = hint, out
		return p

game = ReversiGame()
model = SimpleANN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# 게임 메인 루프(game main loop)
win = 0
epsilon = 1.0
for g in range(TotalGames):
	# 보드 생성
	aiTurn = random.randrange(1, 3)
	if aiTurn == 1: game.newGame(waitAI, waitRandom)
	else: game.newGame(waitRandom, waitAI)

	# 게임 실행하는 메인 모듈
	episode = []
	for board, turn, flips in game.next():
		out = model(convertToInput(board)).detach().squeeze().numpy()
		episode.append((board, turn, out))

	w, b = ReversiGame.getScores(board)
	if w > b and aiTurn == WhiteTurn: win += 1
	elif w < b and aiTurn == BlackTurn: win += 1
	print(f"Win rate = {win*100/(g+1):.1f}%")

	#   에피소드에 대해서 입력, 출력 데이터 준비 
	z = (w > b)*2.0 - 1.0
	x, y = [], []
	for board, turn, v in episode:
		x.append(convertToInput(board))
		# MDP 계산
		v += LearningRate*(z - v)
		y.append(v)
	x = torch.FloatTensor(np.array(x))
	y = torch.FloatTensor(np.array(y)).unsqueeze(1).unsqueeze(2)

	model.train()
	yy = model(x)
	loss = criterion(yy, y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	epsilon *= EpsilonDecay

	# save
	if (g+1)%SaveInterval == 0:
		torch.save(model.state_dict(), Path)
		print(f"Saving {Path} is succeded")

