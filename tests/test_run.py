# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-15 01:00:07
# @Last Modified by:   AaronLai
# @Last Modified time: 2016-11-12 13:03:00

from unittest import TestCase
from run_VQA import run_VQA


class Test_running(TestCase):

    def test_VQA(self):
        run_VQA('train_questions', 'train_choices', 'train_captions.json',
                'train_ans_sol', 'glove_300d.csv', weight=2)
