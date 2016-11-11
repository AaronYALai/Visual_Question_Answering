# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-11-11 18:10:00
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-11 21:58:06


def judge(word, dictionary, cache=[]):
    word = word.strip('?/\\",:\'().*#_!`')
    word = word.lower()

    if len(word) == 0:
        pass

    elif word in dictionary:
        cache.append(word)

    elif word[-2:] == "'s":
        if word[:-2] in dictionary:
            cache.append(word[:-2])
            cache.append("'s")

    elif word[-3:] == "n't":
        if word[:-3] in dictionary:
            cache.append(word[:-3])
            cache.append("n't")

    elif "/" in word:
        for vocab in word.split('/'):
            if len(vocab) != 0 and vocab in dictionary:
                cache.append(vocab)

    elif "-" in word:
        for vocab in word.split('-'):
            if len(vocab) != 0 and vocab in dictionary:
                cache.append(vocab)

    elif "," in word:
        for vocab in word.split(','):
            if len(vocab) != 0 and vocab in dictionary:
                cache.append(vocab)

    elif "(" in word:
        word = word[:word.index("(")]
        if word in dictionary:
            cache.append(word)

    elif word[-1] == "s":
        if word[:-1] in dictionary:
            cache.append(word[:-1])

    return cache
