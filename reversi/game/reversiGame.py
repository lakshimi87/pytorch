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

ScoreBoard = [0] * 64  # Placeholder values for ScoreBoard

class ReversiGame:
	def newGame(self, whitePlayer, blackPlayer):
		board = [3] * 64
		board[27], board[28], board[35], board[36] = 1, 2, 2, 1
		board[20], board[29], board[34], board[43] = 0, 0, 0, 0
		self.board = board
		self.hintCount = 4
		self.lastPlace = None
		self.turn = WhiteTurn
		self.wait = (whitePlayer, blackPlayer)

	def getFlipTiles(self, board, p, turn):
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

	def getHints(self, board, turn):
		hintCount = 0
		for i in range(64):
			if board[i] == 1 or board[i] == 2:
				continue
			ft = self.getFlipTiles(board, i, turn)
			board[i] = 0 if len(ft) > 0 else 3
			hintCount += 1 if board[i] == 0 else 0
		return hintCount

	def getScores(self, board):
		wCount, bCount = 0, 0
		for i in range(64):
			if board[i] == 1:
				wCount += 1
			elif board[i] == 2:
				bCount += 1
		return wCount, bCount

	def prerun(self, board, p, turn):
		if p < 0:
			return
		board[p] = turn
		ft = self.getFlipTiles(board, p, turn)
		for t in ft:
			board[t] = turn

	def getPrerunScore(self, board, p, turn):
		pboard = board[:]
		pboard[p] = turn
		ft = self.getFlipTiles(pboard, p, turn)
		for t in ft:
			pboard[t] = turn
		score = 0
		ref = (0, 1 if turn == 1 else -1, 1 if turn == 2 else -1, 0)
		for k in range(64):
			score += ScoreBoard[k] * ref[pboard[k]]
		return score

	def place(self, board, p, turn):
		board[p] = turn
		self.drawBoard(board)
		self.flipTiles(board, p, turn)
		turn ^= 3
		hintCount = self.getHints(board, turn)
		if hintCount > 0:
			return turn
		turn ^= 3
		hintCount = self.getHints(board, turn)
		if hintCount > 0:
			return turn
		return 0

	def waitForRandom(self, board, turn):
		hints = [i for i in range(64) if board[i] == 0]
		if len(hints) == 0:
			return -1
		return random.choice(hints)

	def drawBoard(self, board):
		# Placeholder for drawing the board, to be implemented with pygame
		pass

	def flipTiles(self, board, p, turn):
		# Placeholder for flipping tiles with animations, to be implemented with pygame
		flips = self.getFlipTiles(board, p, turn)
		for t in flips:
			board[t] = turn
		self.drawBoard(board)


