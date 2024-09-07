"""
Title: play reversi game
Description: play reversi games with specified network
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
from game.reversiGame import ReversiGame, WhiteTurn, BlackTurn
from game.reversiDisplay import ReversiDisplay
import random
import pickle

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

# 컴퓨터 두기
def waitAI(board, turn):
	hints = [ i for i in range(64) if board[i] == 0 ]
	if len(hints) == 0: return -1
	if len(hints) == 1: return hints[0]
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

if __name__ == '__main__':
	path = "models/reversi-simple.pth"
	#	load model with weights
	if not os.path.exists(path):
		print(f"Error: {path} file not found.")
		quit()
	model = SimpleANN()
	model.load_state_dict(torch.load(path, weights_only=True))
	print(f"Loading {path} is succeeded")
	model.eval()

	display = ReversiDisplay()

	if not display.waitForStart(): quit()
	while True:
		if not display.run(None, waitAI): break
		if not display.showResult(): break
