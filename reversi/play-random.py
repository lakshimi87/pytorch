from game import *
import random

def waitComputer(board, hints):
	idx = random.randomrange(len(hints))
	return hints[idx]

game = ReversiGame()
display = ReversiDisplay()

while True:
	if not display.waitForStart(): break
	game.newGame(waitComputer, waitComputer)
	display.drawBoard(game.board)
