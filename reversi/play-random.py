"""
Title: play random
Description: play reversi games with random computer choice
Author: Aubrey Choi
Date: 2024-07-10
Version: 1.2
License: MIT License
"""

from game import ReversiDisplay
import random

def waitRandom(board, turn):
	hints = [ i for i in range(64) if board[i] == 0 ]
	idx = random.randrange(len(hints))
	return hints[idx]

display = ReversiDisplay()

if not display.waitForStart(): quit()
while True:
	if not display.run(None, waitRandom): break
	if not display.showResult(): break
