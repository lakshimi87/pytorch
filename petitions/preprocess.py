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
for file in files[:2]:
	readfile(petitions, file)

df = pd.DataFrame.from_dict(petitions)
print(len(df))
print(df.head())


import re

def removeWhiteSpace(text):
	return re.sub('[\\t\\r\\n\\f\\v]', ' ', str(text))

def removeSpecialChar(text):
	return re.sub('[^ㄱ-ㅎ|가-힣0-9]+', ' ', str(text))

print(df.loc[2]['content'])

df.title = df.title.apply(removeWhiteSpace)
df.title = df.title.apply(removeSpecialChar)

df.content = df.content.apply(removeWhiteSpace)
df.content = df.content.apply(removeSpecialChar)

print(df.loc[2]['content'])


from konlpy.tag import Okt
okt = Okt()
df['titleToken'] = df.title.apply(okt.morphs)
df['contentToken'] = df.content.apply(okt.nouns)
df['finalToken'] = df.titleToken + df.contentToken
df['num_agree'] = df['num_agree'].apply(lambda x:int(x))
print(df.dtypes)

df['label'] = df['num_agree'].apply(lambda x: 'Yes' if x>=80 else 'No')
print(df[df.label == 'Yes'])

dfFinal = df[['finalToken', 'label']]

from gensim.models import Word2Vec
embeddingModel = Word2Vec(dfFinal['finalToken'], sg=1,
	vector_size=100, window=2, min_count=1, workers=4)
print(embeddingModel)

modelResult = embeddingModel.wv.most_similar('건강')
print(modelResult)

# save & load
from gensim.models import KeyedVectors
embeddingModel.wv.save_word2vec_format('data/petitions_tokens_w2v')
loadedModel = KeyedVectors.load_word2vec_format('data/petitions_tokens_w2v')
modelResult = embeddingModel.wv.most_similar('건강')
print(modelResult)

from numpy.random import RandomState
rs = RandomState()
train = dfFinal.sample(frac=0.8, random_state=rs)
test = dfFinal.loc[~dfFinal.index.isin(train.index)]
train.to_csv('data/train.csv', index=False, encoding='utf-8-sig')
test.to_csv('data/test.csv', index=False, encoding='utf-8-sig')
