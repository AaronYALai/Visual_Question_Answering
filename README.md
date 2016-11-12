Visual Question Answering
========

[![Build Status](https://travis-ci.org/AaronYALai/Visual_Question_Answering.svg?branch=master)](https://travis-ci.org/AaronYALai/Visual_Question_Answering)
[![Coverage Status](https://coveralls.io/repos/github/AaronYALai/Visual_Question_Answering/badge.svg?branch=master)](https://coveralls.io/github/AaronYALai/Visual_Question_Answering?branch=master)

About
--------

This is the final project of the course [**Machine Learning and having it deep and structured**](http://nol.ntu.edu.tw/nol/coursesearch/print_table.php?course_id=942%20U0590&class=&dpt_code=9210&ser_no=51785&semester=104-1&lang=EN) of National Taiwan University (offered by [**Hung-yi Lee**](http://speech.ee.ntu.edu.tw/~tlkagk/index.html)):

**Task**: Need to build an AI system to answer multiple choices question according to their corresponding images.

An example is shown below:

![overview](https://github.com/AaronYALai/Visual_Question_Answering/blob/master/img.png)

The answer is **(C)bananas**. This task includes solving a number of sub-problems in Natural Language Processing and Computer Vision. The system needs to localize the subject being referenced (something being sold), and needs to detect objects (bananas).

More information: [slides](https://docs.google.com/presentation/d/1xJkR75dLPlNs7RI3jtdNLEEyVETk6sZK3OyoN6yKx-M/edit#slide=id.gf3906c444_190_0)

Model Details
--------

Preprocessing:
- NLP part: For each question and its five choices, parse them into a list of lowercase words.
- ex. "What's this man selling?" -> ["what", "'s", "this", "man", "selling"]
- Visual part: find the caption for each image and parse the caption into a list of lowercase words.
- Image captions are from [**MS COCO**](http://mscoco.org/dataset/#download).

Transforming:
- Get the 300 dimensional vector representation of each word in the list.
- Word vector representations are from [**GloVe**](http://nlp.stanford.edu/projects/glove/).
- Question vector: average the word vectors of the question into a single 300 dimension vector.
- Choice vector: average the word vectors of each choice of questions into a single 300 dimension vector.
- Caption vector: average the word vectors of the caption of each image into a single 300 dimension vector.

Make data:
- For each question, make 5 training instances (x: 900 dim, y: 2 dim):
    - x = [ques, cap, choice1], y = [0, 1]...
    - y indicates [not the answer, is the answer]
- There are totally (5 * number of questions) training instances.

Training and predicting:
- Build a deep neural network (DNN) using [**Keras**](http://keras.io/) with input dim 900 and output dim 2.
- Using dropout and rectified linear unit (ReLU) as the activation function.
- Train the DNN by Nesterov Accelerated Gradient (NAG).
- Double or triple the training instances whose choice is right (y = [0, 1]) and train the model again.

This method originally trained on 146,962 questions (82,783 pictures), predicted on 72,802 questions (40,504 pictures) and reached about 82% accuracy.

#### kaggle page: [link](https://inclass.kaggle.com/c/104-1-mlds-final-project) (competed as *importeverything*)

#### Reference: [**GloVe**](http://nlp.stanford.edu/projects/glove/), [**Captions (COCO)**](http://mscoco.org/dataset/#download), [**Keras**](http://keras.io/)

Usage
--------
Clone the repo and use the [virtualenv](http://www.virtualenv.org/):

    git clone https://github.com/AaronYALai/Visual_Question_Answering

    cd Visual_Question_Answering

    virtualenv venv

    source venv/bin/activate

Install all dependencies and train agents:

    pip install -r requirements.txt

    python run_VQA.py
