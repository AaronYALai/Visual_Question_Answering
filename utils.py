# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-11 18:10:00
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-12 12:09:33

import pandas as pd
import numpy as np
import json
import re

from collections import defaultdict


def judge(word, dictionary, cache=[]):
    """check if the word in the dictionary"""
    word = word.strip('?/\\",:\'().*#_!`')
    word = word.lower()

    if len(word) == 0:
        pass

    elif dictionary[word] == 1:
        cache.append(word)

    elif word[-2:] == "'s":
        if dictionary[word[:-2]] == 1:
            cache.append(word[:-2])
            cache.append("'s")

    elif word[-3:] == "n't":
        if dictionary[word[:-3]] == 1:
            cache.append(word[:-3])
            cache.append("n't")

    elif "/" in word:
        for vocab in word.split('/'):
            if len(vocab) != 0 and dictionary[vocab] == 1:
                cache.append(vocab)

    elif "-" in word:
        for vocab in word.split('-'):
            if len(vocab) != 0 and dictionary[vocab] == 1:
                cache.append(vocab)

    elif "," in word:
        for vocab in word.split(','):
            if len(vocab) != 0 and dictionary[vocab] == 1:
                cache.append(vocab)

    elif "(" in word:
        word = word[:word.index("(")]
        if dictionary[word] == 1:
            cache.append(word)

    elif word[-1] == "s":
        if dictionary[word[:-1]] == 1:
            cache.append(word[:-1])

    return cache


def clean_words(sentence, dictionary, result):
    """transfrom the sentence or words into a list of words"""
    for word in sentence.split(' '):
        result = judge(word, dictionary, result)

    return result


def load_train_data(ques_file, choice_file, cap_file, base_dir='./rawdata/'):
    """loading training data as a dictionary"""
    train_ques = pd.read_csv(base_dir + ques_file, sep='\t')
    train_choices = pd.read_csv(base_dir + choice_file, sep='\t')
    train_choices = train_choices.set_index('q_id')

    with open(base_dir + cap_file) as file:
        caps_dict = json.load(file)

    train_data = defaultdict(dict)
    for img_id, qid, ques in train_ques.values:
        train_data[qid]['question'] = ques
        train_data[qid]['caption'] = caps_dict[str(img_id)][0]

        choices = train_choices.loc[qid]['choices']
        clean_choices = []
        for choice in re.split('\(.?\)', choices):
            if len(choice.strip('" ')) > 0:
                clean_choices.append(choice.strip('" '))

        if len(clean_choices) != 5:
            print(clean_choices)
        train_data[qid]['choices'] = clean_choices

    return train_data


def clean_data(data, wordvec_file, base_dir='./rawdata'):
    """trainform data into words in dictionary"""
    wordvec = pd.read_csv(base_dir + wordvec_file, sep=' ', header=None,
                          index_col=0)
    dictionary = defaultdict(int)
    for w in wordvec.index.values:
        dictionary[w] = 1

    result_data = {}

    for qid, data_dict in data.items():
        result_dict = {}

        # vectorize question
        ques_vec = clean_words(data_dict['question'], dictionary, [])
        result_dict['question'] = ques_vec

        # vectorize choices
        choices_vec = []
        for choice in data_dict['choices']:
            choi_vec = clean_words(choice, dictionary, [])
            choices_vec.append(choi_vec)

        result_dict['choices'] = choices_vec

        # vectorize caption
        caption_vec = clean_words(data_dict['caption'], dictionary, [])
        result_dict['caption'] = caption_vec

        result_data[qid] = result_dict

    return result_data


def vectorize(word_bag, wordvec):
    """transfrom a text into a vector representation"""
    if len(word_bag) == 0:
        vec = np.zeros(wordvec.shape[1])
    elif len(word_bag) == 1:
        vec = wordvec.loc[word_bag[0]].values
    else:
        vec = wordvec.loc[word_bag].values.mean(axis=0)

    return vec
