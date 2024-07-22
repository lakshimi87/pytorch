"""
Title: ReversiGame class
Description: Game for reversi game using PyGame
Author: Aubrey Choi
Date: 2024-06-30
Version: 1.2
License: MIT License
"""

import random
import time

WhiteTurn, BlackTurn = 1, 2
EmptyTile, WhiteTile, BlackTile, HintTile = 0, 1, 2, 3

class ReversiGame:
	def newGame(self, whitePlayer, blackPlayer):
		board = [3] * 64
		board[27], board[28], board[35], board[36] = 1, 2, 2, 1
		board[20], board[29], board[34], board[43] = 0, 0, 0, 0
		self.board = board
		self.lastPlace = None
		self.turn = WhiteTurn
		self.wait = (whitePlayer, blackPlayer)

	def getFlipTiles(board, p, turn):
		dxy = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
		x, y = p % 8, p // 8
		flips = []
		for dx, dy in dxy:
			isFlip = False
			tf = []
			nx, ny = x + dx, y + dy
			while 0 <= nx < 8 and 0 <= ny < 8:
				np = nx + ny * 8
				if board[np] == turn:
					isFlip = True
					break
				if board[np] != (turn ^ 3): break
				tf.append(np)
				nx, ny = nx + dx, ny + dy
			if isFlip:
				flips += tf
		return flips

	def getHints(board, turn):
		hintCount = 0
		for i in range(64):
			if board[i] == 1 or board[i] == 2:
				continue
			ft = ReversiGame.getFlipTiles(board, i, turn)
			board[i] = 0 if len(ft) > 0 else 3
			hintCount += 1 if board[i] == 0 else 0
		return hintCount

	def getScores(board):
		wCount, bCount = 0, 0
		for i in range(64):
			if board[i] == 1:
				wCount += 1
			elif board[i] == 2:
				bCount += 1
		return wCount, bCount

	def prerun(board, p, turn):
		board[p] = turn
		ft = ReversiGame.getFlipTiles(board, p, turn)
		for t in ft: board[t] = turn

	def flipTiles(board, flips, turn):
		for t in flips:
			board[t] = turn

	def next(self):
		passCount = 0
		hintCount = 4
		while passCount < 2:
			if hintCount == 0:
				passCount += 1
				self.turn ^= 3
				hintCount = ReversiGame.getHints(self.board, self.turn)
				yield (self.board, self.turn, None)
				continue
			p = self.wait[self.turn-1](self.board, self.turn)
			if p == None or self.board[p] != 0: break
			flips = ReversiGame.getFlipTiles(self.board, p, self.turn)
			ReversiGame.flipTiles(self.board, flips, self.turn)
			self.board[p] = self.turn
			passCount = 0
			self.turn ^= 3
			hintCount = ReversiGame.getHints(self.board, self.turn)
			yield (self.board, self.turn, flips)

