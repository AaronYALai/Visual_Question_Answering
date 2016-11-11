# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-11 16:45:09
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-12 00:59:35

import pandas as pd
import numpy as np
import json
import csv
import re

from collections import defaultdict
from utils import load_train_data, clean_data
from datetime import datetime

base_dir = './rawdata/'
ques_file = 'train_questions'
choice_file = 'train_choices'
cap_file = 'train_captions.json'
wordvec_file = 'glove_300d.csv'

st = datetime.now()
train_data = load_train_data(ques_file, choice_file, cap_file, base_dir)
print('Done loading data. Using %s.' % str(datetime.now() - st))

train_data = clean_data(train_data, wordvec_file, base_dir)
print('Done transforming data. Using %s.' % str(datetime.now() - st))
# make array
wordvec = pd.read_csv(base_dir + wordvec_file, sep=' ', header=None, index_col=0)
#vectorize(word_bag, wordvec)

# construct model


# train model


# predict and accuracy



import pdb;pdb.set_trace()






