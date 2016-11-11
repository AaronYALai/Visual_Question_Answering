# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-11 18:10:00
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-11 18:31:06


def judge(word, dictionary):
    word = word.strip('?/\\",:\'().*#_!`')
    word = word.lower()
    result = []

    if len(word) == 0:
        pass

    elif word in dictionary:
        result.append(word)

    elif word[-2:] == "'s":
        if word[:-2] in dictionary:
            result.append(word[:-2])
            result.append("'s")

    elif word[-3:] == "n't":
        if word[:-3] in dictionary:
            result.append(word[:-3])
            result.append("n't")

    elif "/" in word:
        for vocab in word.split('/'):
            if len(vocab) != 0 and vocab in dictionary:
                result.append(vocab)

    elif "-" in word:
        for vocab in word.split('-'):
            if len(vocab) != 0 and vocab in dictionary:
                result.append(vocab)

    elif "," in word:
        for vocab in word.split(','):
            if len(vocab) != 0 and vocab in dictionary:
                result.append(vocab)

    elif "(" in word:
        word = word[:word.index("(")]
        if word in dictionary:
            result.append(word)

    elif word[-1] == "s":
        if word[:-1] in dictionary:
            result.append(word[:-1])

    return result
