"""
Title: ReversiDisplay class
Description: Display for reversi game using PyGame
Author: Aubrey Choi
Date: 2024-06-30
Version: 1.2
License: MIT License
"""

import math	 # 수학 관련된 라이브러리 모듈
import sys
import random   # 난수 발생용 라이브러리 모듈
import pygame   # pygame을 위한 라이브러리 모듈
import time	 # 시간을 측정하기 위한 모듈
from pygame.locals import *
from .reversiGame import *

# 기본적인 컬러값
White = (255, 255, 255)
Black = (0, 0, 0)
GridColor = (0, 0, 150)
TextColor = (255, 255, 200)
TextOutlineColor = (0, 0, 0)
HintColor = (100, 100, 255)

class ReversiDisplay:
	def __init__(self, fps=30, spaceSize=80, margin=20, flipSpeed=32):
		self.game = ReversiGame()
		self.fps = fps
		self.spaceSize = spaceSize
		self.margin = margin
		self.flipSpeed = flipSpeed
		self.waitTime = 1000 // fps
		self.halfSpaceSize = spaceSize // 2
		self.windowSize = (spaceSize * 8 + margin * 2, spaceSize * 8 + margin + 40)

		# pygame 초기화
		pygame.init()

		# 디스플레이 서피스(display surface) 만들기
		self.displaySurf = pygame.display.set_mode(self.windowSize)
		pygame.display.set_caption("Reversi AI Game")

		# 텍스트를 위한 폰트
		self.normalFont = pygame.font.Font("freesansbold.ttf", 16)
		self.bigFont = pygame.font.Font("freesansbold.ttf", 64)

		# 백그라운드 이미지 설정
		self.boardImage = pygame.image.load('board.png')
		self.boardImage = pygame.transform.smoothscale(self.boardImage, (8 * spaceSize, 8 * spaceSize))
		self.boardImageRect = self.boardImage.get_rect()
		self.boardImageRect.topleft = (margin, margin)
		self.bgImage = pygame.image.load("bg.png")
		self.bgImage = pygame.transform.smoothscale(self.bgImage, self.windowSize)
		self.bgImage.blit(self.boardImage, self.boardImageRect)

	def __del__(self):
		pygame.quit()

	# 타일의 중앙값 찾기
	def getCenter(self, p):
		r, c = p // 8, p % 8	#  세로위치(r), 가로위치(c)
		return (
			self.margin + c * self.spaceSize + self.halfSpaceSize, 
			self.margin + r * self.spaceSize + self.halfSpaceSize
		)

	# 보드 그리기
	def drawBoard(self, board):
		# 배경을 먼저 그린다.
		self.displaySurf.blit(self.bgImage, self.bgImage.get_rect())

		# 보드의 줄을 친다.  8칸에 줄을 치기위해서 8+1개의 줄이 필요
		for i in range(8 + 1):
			t = i * self.spaceSize + self.margin
			pygame.draw.line(self.displaySurf, GridColor, (t, self.margin), (t, self.margin + 8 * self.spaceSize))
			pygame.draw.line(self.displaySurf, GridColor, (self.margin, t), (self.margin + 8 * self.spaceSize, t))

		# 돌을 그립니다.
		for i in range(8 * 8):
			cx, cy = self.getCenter(i)
			if board[i] == 1:
				pygame.draw.circle(self.displaySurf, White, (cx, cy), self.halfSpaceSize - 4)
			elif board[i] == 2:
				pygame.draw.circle(self.displaySurf, Black, (cx, cy), self.halfSpaceSize - 3)
			elif board[i] == 0:
				pygame.draw.rect(self.displaySurf, HintColor, (cx - 4, cy - 4, 8, 8))
	
	# 정보를 표시하기
	def drawInfo(self, board, turn):
		scores = ReversiGame.getScores(board)
		colors = ("", "White", "Black")
		# 표시할 문자열 만들기
		text = f"White : {scores[0]:2}  Black : {scores[1]:2}  Turn : {colors[self.game.turn]}"
		# 문자열을 이미지로 변환
		scoreSurf = self.normalFont.render(text, True, TextColor)
		scoreRect = scoreSurf.get_rect()
		scoreRect.bottomleft = (self.margin, self.windowSize[1] - 5)
		# 문자열 이미지를 디스플레이에 표시
		self.displaySurf.blit(scoreSurf, scoreRect)

	# 타일 뒤집기
	def flipTiles(self, board, turn, flips):
		self.drawBoard(board)
		# 타일 뒤집기 애니메이션
		for rgb in range(0, 255, self.flipSpeed):
			color = tuple([rgb] * 3) if turn == 1 else tuple([255 - rgb] * 3)
			for t in flips:
				cx, cy = self.getCenter(t)
				pygame.draw.circle(self.displaySurf, color, (cx, cy), self.halfSpaceSize - 3)
			pygame.display.update()
			pygame.time.wait(self.waitTime)
		self.drawInfo(board, turn)

	# 클릭위치값이 주어졌을때, 클릭된 보드의 셀번호 반환
	def getClickPosition(self, x, y):
		rx, ry = (x - self.margin) // self.spaceSize, (y - self.margin) // self.spaceSize
		return None if rx < 0 or rx >= 8 or ry < 0 or ry >= 8 else rx + ry * 8

	# 사용자 입력 기다리기
	def waitForInput(self, board, turn):
		while True:
			self.drawBoard(board)
			self.drawInfo(board, turn)
			pygame.display.update()  # display 업데이트
			# 이벤트 처리 (PyGame에서 마우스, 키보드 또는 윈도우의 버튼들)
			for event in pygame.event.get():
				if event.type == QUIT:
					return None
				if event.type == KEYUP and event.key == K_ESCAPE:
					return None
				if event.type == MOUSEBUTTONUP:
					p = self.getClickPosition(event.pos[0], event.pos[1])
					if p is not None and board[p] == 0:
						return p

	def draw(self, board, turn):
		for _ in range(10):
			self.drawBoard(board)
			self.drawInfo(board, turn)
			pygame.display.update()  # display 업데이트
			# 이벤트 처리 (PyGame에서 마우스, 키보드 또는 윈도우의 버튼들)
			for event in pygame.event.get():
				if event.type == QUIT:
					return False
				if event.type == KEYUP and event.key == K_ESCAPE:
					return False
		return True

	# 시작 버튼 기다리기
	def waitForStart(self):
		# start game button 생성
		startGameSurf = self.bigFont.render("Start Game", True, TextColor)
		startGameRect = startGameSurf.get_rect()
		startGameRect.center = (self.windowSize[0] // 2, self.windowSize[1] // 2)
		while True:
			self.displaySurf.blit(self.bgImage, self.bgImage.get_rect())
			# 사용자 게임 버튼 정보 표시
			self.drawOutlineText("Start Game", self.bigFont, 
				(self.windowSize[0] // 2, self.windowSize[1] // 2), TextColor, TextOutlineColor)
			pygame.display.update()  # display 업데이트
			# 이벤트 처리 (PyGame에서 마우스, 키보드 또는 윈도우의 버튼들)
			for event in pygame.event.get():
				if event.type == QUIT:
					return False
				if event.type == KEYUP and event.key == K_ESCAPE:
					return False
				if event.type == MOUSEBUTTONUP:
					if startGameRect.collidepoint(event.pos):
						return True
			pygame.time.wait(self.waitTime)

	def waitFrames(self, frames):
		for event in pygame.event.get():
			if event.type == QUIT: return False
			if event.type == KEYUP and event.key == K_ESCAPE: return False
		pygame.time.wait(self.waitTime*frames)
		return True

	# 결과 출력
	def showResult(self):
		scores = ReversiGame.getScores(self.game.board)
		result = (scores[0] > scores[1]) + (scores[0] < scores[1])*2
		msg = ( "Draw", "White Win!!", "Black Win!!" )
		for _ in range(self.fps*5):
			self.drawBoard(self.game.board)
			self.drawInfo(self.game.board, 0)
			self.drawOutlineText(msg[result], self.bigFont, 
				(self.windowSize[0]//2, self.windowSize[1]//2), TextColor, TextOutlineColor)
			pygame.display.update()  # display 업데이트
			for event in pygame.event.get():
				if event.type == QUIT: return False
				if event.type == KEYUP and event.key == K_ESCAPE: return False
				if event.type == MOUSEBUTTONUP: return True
			pygame.time.wait(self.waitTime)
		return True

	def drawOutlineText(self, text, font, pos, c, o):
		d = ( (-2, 0), (0, -2), (2, 0), (0, 2) )
		for x, y in d:
			ts = font.render(text, True, o)
			tr = ts.get_rect()
			tr.center = (pos[0]+x, pos[1]+y)
			self.displaySurf.blit(ts, tr)
		ts = font.render(text, True, c)
		tr = ts.get_rect()
		tr.center = pos
		self.displaySurf.blit(ts, tr)
	
	def run(self, waitWhitePlayer, waitBlackPlayer):
		# 보드 생성
		self.game.newGame(waitWhitePlayer, waitBlackPlayer)
		self.draw(self.game.board, WhiteTurn)

		# 게임 실행하는 메인 모듈
		for board, turn, flips in self.game.next():
			if flips != None: self.flipTiles(board, turn^3, flips)
			if not self.waitFrames(3): return False
			self.draw(board, turn)
		return True

