#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan

Some env params may need to use.
"""


OOV = '_OOV_'
START = '_S_'
END = '_E_'
GOLD_TAG = 'GoldNER'
PRED_TAG = 'NER'
task = 'ner'
MAX_LEN = 175  # Max sequence length, because Weibo's input limitation

LogP_ZERO = float('-inf')
LogP_INF = float('inf')
LogP_ONE = 0.0
FloatX = 'float32'

BASE_DIR = r'/Users/heshenghuan/Projects/linear_chainCRF/'
MODEL_DIR = BASE_DIR + r'models/'
DATA_DIR = BASE_DIR + r'data/'
EMB_DIR = BASE_DIR + r'embeddings/'
OUTPUT_DIR = BASE_DIR + r'export/'
LOG_DIR = BASE_DIR + r'Summary/'
