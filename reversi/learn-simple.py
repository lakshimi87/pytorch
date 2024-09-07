"""
Title: learn simple ANN
Description: reversi simple ANN learning
Author: Aubrey Choi
Date: 2024-07-10
Version: 1.2
License: MIT License
"""
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game.reversiGame import ReversiGame, WhiteTurn, BlackTurn

Path = "models/reversi-simple.pth"
EpsilonDecay = 0.99
TotalGames = 1000
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
	return torch.tensor(x).view(1, -1)

# random하게 두기
def waitRandom(board, turn):
	hints = [ i for i in range(64) if board[i] == 0 ]
	return np.random.choice(hints)

# 컴퓨터 두기
def waitAI(board, turn):
	hints = [ i for i in range(64) if board[i] == 0 ]
	if len(hints) == 0: return -1
	if len(hints) == 1: return hints[0]
	if np.random.rand() < epsilon: return np.random.choice(hints)
	if turn == WhiteTurn:
		p, maxp = hints[0], -1
		for hint in hints:
			pboard = board[:]
			ReversiGame.prerun(pboard, hint, turn)
			out = model(convertToInput(board)).item()
			if out > maxp: p, maxp = hint, out
		return p
	else:
		p, minp = hints[0], -1
		for hint in hints:
			pboard = board[:]
			ReversiGame.prerun(pboard, hint, turn)
			out = model(convertToInput(board)).item()
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
	model.eval()
	# 보드 생성
	aiTurn = np.random.randint(1, 3)
	if aiTurn == 1: game.newGame(waitAI, waitRandom)
	else: game.newGame(waitRandom, waitAI)

	# 게임 실행하는 메인 모듈
	episode = []
	for board, turn, flips in game.next():
		out = model(convertToInput(board)).item()
		episode.append((board, out))

	w, b = ReversiGame.getScores(board)
	if (w > b and aiTurn == WhiteTurn) or (w < b and aiTurn == BlackTurn): win += 1
	print(f"Win rate = {win*100/(g+1):.1f}%", end='\r')

	#   에피소드에 대해서 입력, 출력 데이터 준비 
	z = (w > b)*2.0 - 1.0
	x, y = [], []
	for board, v in episode:
		x.append(convertToInput(board))
		v += LearningRate*(z - v)
		y.append(v)
	x = torch.stack(x)
	y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

	model.train()
	yy = model(x).squeeze(-1)
	loss = criterion(yy, y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	epsilon *= EpsilonDecay

# save
torch.save(model.state_dict(), Path)
print(f"\nSaving {Path} is succeded")

