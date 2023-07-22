import pandas as pd
import numpy as np
import collections
import pickle
# import tensorflow as tf
import matplotlib.pyplot as plt
#import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import cv2
import pickle
# from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
# from keras.preprocessing import image
# from keras.models import Model, load_model
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.layers import Input, Dense, Dropout, Embedding, LSTM
#from keras.layers.merge import add
def readTextFile(path):
    with open(path) as f:
        captions = f.read()
    return captions
captions  = readTextFile("./Data/Flickr8k.token.txt")
captions = captions.split('\n')
print(len(captions))

descriptions = {}
for x in captions:
    first,second = x.split('\t')
    img_name = first.split(".")[0]
    if descriptions.get(img_name) is None:
        descriptions[img_name] = []
    descriptions[img_name].append(second)

def clean_text(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-z]+"," ",sentence)
    sentence = sentence.split()
    
    sentence  = [s for s in sentence if len(s)>1]
    sentence = " ".join(sentence)
    return sentence

for key,caption_list in descriptions.items():
    for i in range(len(caption_list)):
        caption_list[i] = clean_text(caption_list[i])

with open("descriptions_1.txt","w") as f:
    f.write(str(descriptions))

descriptions = None
with open("descriptions_1.txt",'r') as f:
    descriptions= f.read()
    
json_acceptable_string = descriptions.replace("'","\"")
descriptions = json.loads(json_acceptable_string)

vocab = set()
for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]

total_words = []
for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]

counter = collections.Counter(total_words)
freq_cnt = dict(counter)
print(len(freq_cnt.keys()))
# Sort this dictionary according to the freq count
sorted_freq_cnt = sorted(freq_cnt.items(),reverse=True,key=lambda x:x[1])
# Filter
threshold = 10
sorted_freq_cnt  = [x for x in sorted_freq_cnt if x[1]>threshold]
total_words = [x[0] for x in sorted_freq_cnt]
print(len(total_words))

word_to_idx = {}
idx_to_word = {}
for i,word in enumerate(total_words):
    word_to_idx[word] = i+1
    idx_to_word[i+1] = word

idx_to_word[1846] = 'startseq'
word_to_idx['startseq'] = 1846

idx_to_word[1847] = 'endseq'
word_to_idx['endseq'] = 1847

vocab_size = len(word_to_idx) + 1
print("Vocab Size",vocab_size)
max_len = 35
print(idx_to_word[1])

with open('saved_word_to_idx.pkl', 'wb') as f:
    pickle.dump(word_to_idx, f)
with open('saved_idx_to_wordy.pkl', 'wb') as f:
    pickle.dump(idx_to_word, f)
    
