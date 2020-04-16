# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 05:06:44 2020

@author: KIIT
"""
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.tokenize import regexp
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

df = open(r"C:\Users\KIIT\Documents\Python Scripts\frank.txt")
df.read

def tokenize_words(input):
    input = input.lower()
    tokenizer = regexp(r'w+')
    tokens = tokenizer.tokenize(input)
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    return ''.join(filtered)

pi = tokenize_words(df)

chars = sorted(list(set(pi)))
chartonum = dict((c,i) for i, c in enumerate(chars))

inpl = len(pi)
vocabl = len(chars)

print('total char', inpl)
print('total vocab', vocabl)

seql = 100
xdata = []
ydata = []

for i in range(0, inpl - seql, 1):
    inseq = pi[i:i + seql]
    outseq = pi[i + seql]
    xdata.append([chartonum[char] for char in inseq])
    ydata.append([chartonum[outseq])
    
np = len(xdata)
print('patterns', np)

x = np.reshape(xdata, (np, seql,1))
x = x/float(vocabl)

model = Sequential()
model.add(LSTM[254, input_shape=(x_shape[1], x.shape[2], return_seq=True)])
model.add(Dropout(0.2))
model.add(LSTM[254, return_seq=True)])
model.add(Dropout(0.2))
model.add(LSTM(328))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimize='slope')

filepath = 'model_weight_saved.hdf5'
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='min')
descb = [checkpoint]

model.fit(x, y, epochs=4, batch_size=256, callbacks=descb)

filename = 'model_weight_saved.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

numtochar = dict((c,i) for i, c in enumerate(chars))

start = np.random.randint(0, len(xdata) - 1)
pat = xdata[start]
print('rand speed')
print('\', ''.join([numtochar[val] for val in pat]), '\')

#text generation
for i in range(1000):
    x = np.reshape(xdata, (np, seql,1))
    x = x/float(vocabl)
    pred = model.predict(x, verbose=0)
    ind= np.argmax(pred)
    res = numtochar(ind)
    seqin = (numtochar[val] for val in pat)
    pat.append(ind)
    pat = pat[l:len(pat)]














