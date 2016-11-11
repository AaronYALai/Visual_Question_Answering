# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-11 16:45:09
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-11 18:36:15

import pandas as pd
import numpy as np
import json
import csv

from collections import defaultdict
from utils import judge


base_dir = './rawdata/'
ques_file = 'train_questions.json'
imgid_file = 'train_images.json'
choice_file = 'train_choices.json'
cap_file = 'train_captions.json'
wordvec_file = 'glove_300d.csv'

with open(base_dir + ques_file) as file:
    ques_dict = json.load(file)

with open(base_dir + choice_file) as file:
    choices_dict = json.load(file)

with open(base_dir + cap_file) as file:
    caps_dict = json.load(file)

wordvec = pd.read_csv(base_dir + wordvec_file, sep=' ', header=None, index_col=0)
dictionary = wordvec.index.values

def clean_words(sentence, dictionary):
    result = []
    for word in sentence.split(' '):
        result += judge(word, dictionary)

    return result

# vectorize
ques_vec = {}
for key, val in ques_dict.items():
    word_bag = clean_words(val, dictionary)

    if len(word_bag) == 0:
        vec = np.zeros(wordvec.shape[1])
    else:
        vec = wordvec.loc[word_bag].fillna(0).values.mean(axis=0)

    ques_vec[key] = vec

choices_vec = defaultdict(list)
for key, choices in choices_dict.items():
    for val in choices:
        word_bag = clean_words(val, dictionary)

        if len(word_bag) == 0:
            vec = np.zeros(wordvec.shape[1])
        else:
            vec = wordvec.loc[word_bag].fillna(0).values.mean(axis=0)

        choices_vec[key].append(vec)

caps_vec = defaultdict(list)
for key, captions in caps_dict.items():
    for val in captions:
        word_bag = clean_words(val, dictionary)

        if len(word_bag) == 0:
            vec = np.zeros(wordvec.shape[1])
        else:
            vec = wordvec.loc[word_bag].fillna(0).values.mean(axis=0)

        vec = wordvec.loc[word_bag].fillna(0).values.mean(axis=0)
        caps_vec[key].append(vec)

# make array


# construct model


# train model


# predict and accuracy



import pdb;pdb.set_trace()






