# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-11 16:45:09
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-12 13:37:12

import pandas as pd
import numpy as np

from utils import load_train_data, clean_data, vectorize
from datetime import datetime

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD


def construct_model(n_input, n_output, n_hidden, neurons, lr=0.01,
                    momentum=0.95, dropout=0.2, activ='relu'):
    """Construct a deep neural network by Keras"""
    model = Sequential()
    model.add(Dense(input_dim=n_input, output_dim=neurons,
                    init="glorot_uniform"))
    model.add(Activation(activ))
    model.add(Dropout(dropout))

    for i in range(n_hidden - 1):
        model.add(Dense(input_dim=neurons, output_dim=neurons,
                        init="glorot_uniform"))
        model.add(Activation(activ))
        model.add(Dropout(dropout))

    model.add(Dense(input_dim=neurons, output_dim=n_output,
                    init="glorot_uniform"))
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=1e-6, momentum=momentum, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=["accuracy"])

    return model


def make_data(data, wordvec_file, ans_file=None, base_dir='./rawdata'):
    if ans_file:
        ans_data = pd.read_csv(base_dir+ans_file, sep='\t').set_index('q_id')
        answer_map = {
            'A': 0,
            'B': 1,
            'C': 2,
            'D': 3,
            'E': 4
        }
        ans_data['answer'] = ans_data['answer'].apply(lambda x: answer_map[x])

    wordvec = pd.read_csv(base_dir + wordvec_file, sep=' ', header=None,
                          index_col=0)

    vectors = {}
    for qid, data_dict in data.items():
        vectors[qid] = {}

        ques_vec = vectorize(data_dict['question'], wordvec)
        capt_vec = vectorize(data_dict['caption'], wordvec)
        context = np.concatenate((ques_vec, capt_vec))

        X = []
        for choice in data_dict['choices']:
            choi_vec = vectorize(choice, wordvec)
            x = np.concatenate((context, choi_vec))
            X.append(x)

        vectors[qid]['X'] = np.array(X)

        if ans_file:
            ans = ans_data.loc[qid]['answer']
            y = [([0, 1] if i == ans else [1, 0]) for i in range(5)]
            vectors[qid]['y'] = np.array(y)

    return vectors


def predict_accuracy(model, X, train_vec, questions):
    """calculate the accuracy of answering the questions"""
    predicts = model.predict(X)
    predicts = np.array_split(predicts, len(questions))

    hit = 0
    for ind, qid in enumerate(questions):
        ans = np.argmax(train_vec[qid]['y'][:, 1])
        pred = np.argmax(predicts[ind][:, 1])

        if ans == pred:
            hit += 1

    accuracy = hit / len(questions)

    return accuracy


def run_VQA(ques_file, choice_file, cap_file, ans_file, wordvec_file,
            n_hidden=2, neurons=128, dropout=0.2, lr=1e-3, batch_size=32,
            valid_ratio=0.2, nb_epoch=10, momentum=0.95, activ='relu',
            base_dir='./rawdata/', weight=1):
    """building a model to answer visual questions"""
    print('Start:')

    st = datetime.now()
    train_data = load_train_data(ques_file, choice_file, cap_file, base_dir)
    print('\tDone loading data. Using %s.' % str(datetime.now() - st))

    train_data = clean_data(train_data, wordvec_file, base_dir)
    print('\tDone transforming data. Using %s.' % str(datetime.now() - st))

    train_vec = make_data(train_data, wordvec_file, ans_file, base_dir)

    # split validation set
    questions = np.array(list(train_vec.keys()))
    n_ques = len(questions)
    rand_inds = np.random.permutation(n_ques)

    valid_ques = questions[rand_inds[:int(n_ques * valid_ratio)]]
    train_ques = questions[rand_inds[int(n_ques * valid_ratio):]]

    train_X = np.vstack(tuple([train_vec[qid]['X'] for qid in train_ques]))
    train_y = np.vstack(tuple([train_vec[qid]['y'] for qid in train_ques]))

    valid_X = np.vstack(tuple([train_vec[qid]['X'] for qid in valid_ques]))
    valid_y = np.vstack(tuple([train_vec[qid]['y'] for qid in valid_ques]))

    print('\tDone making data into vectors. Using %s.\n' %
          str(datetime.now() - st))

    instance = train_vec[questions[0]]
    model = construct_model(instance['X'].shape[1], instance['y'].shape[1],
                            n_hidden, neurons, lr, momentum, dropout, activ)

    print('\tDone constructing model. Start training ...')
    if int(weight) > 1:
        x_pos = train_X[train_y[:, 1] == 1]
        y_pos = train_y[train_y[:, 1] == 1]

        new_X = [train_X] + [x_pos for i in range(int(weight) - 1)]
        new_y = [train_y] + [y_pos for i in range(int(weight) - 1)]

        X = np.vstack(tuple(new_X))
        y = np.vstack(tuple(new_y))
    else:
        X, y = train_X, train_y

    model.fit(X, y, batch_size, nb_epoch, shuffle=True, verbose=1,
              validation_data=(valid_X, valid_y))

    print('\nStart predicting...')
    train_accu = predict_accuracy(model, train_X, train_vec, train_ques)
    valid_accu = predict_accuracy(model, valid_X, train_vec, valid_ques)

    print('\ttraining accuracy: %.2f %%; validation accuracy: %.2f %%' %
          (100*train_accu, 100*valid_accu))

    print('\nDone. Using %s.' % str(datetime.now() - st))


def main():
    run_VQA('train_questions', 'train_choices', 'train_captions.json',
            'train_ans_sol', 'glove_300d.csv', n_hidden=2, neurons=128,
            dropout=0.4, lr=1e-3, batch_size=28, valid_ratio=0.2,
            nb_epoch=20, weight=2)


if __name__ == '__main__':
    main()
