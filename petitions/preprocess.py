import json
import pandas as pd

with open("archive/files") as f:
	files = [file.strip() for file in f.readlines()]
print(files[:10])

def readfile(petitions, file):
	with open("archive/"+file) as f:
		items = f.readlines()
	if len(petitions) == 0:
		parsed = json.loads(items[0])
		for key in parsed:
			petitions[key] = []
	if 'replies' in petitions: petitions.remove('replies')
	for item in items:
		parsed = json.loads(item)
		for key in petitions:
			petitions[key].append(parsed[key])

petitions = { }
for file in files:
	readfile(petitions, file)

df = pd.DataFrame.from_dict(petitions)
print(len(df))
print(df.head())
