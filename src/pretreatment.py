#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 08 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import jieba
import codecs as cs
import numpy as np
import tensorflow as tf
from collections import defaultdict
from parameters import OOV, MAX_LEN
from features import fields, pmi_fields, templates, pmi_templates
from features import feature_extractor

# keras API
pad_sequences = tf.contrib.keras.preprocessing.sequence.pad_sequences


def create_dicts(train, valid, test, threshold, mode, anno=None):
    """
    Returns three dicts, which are feature dictionary, word dictionary
    and label dictionary.

    ## Params
        train: train data
        valid: valid data
        test: test data
        threshold: threshold value of feature frequency
        mode: the type of embeddings(char/charpos)
        anno: whether the data is annotated
    ## Return
        returns 3 dicts
    """

    def get_label(stream, pos):
        return [[item[pos] for item in sentc] for sentc in stream]

    words = (get_label(train, 'r')  # 训练集字符集合
             + ([] if valid is None else get_label(valid, 'r'))  # 验证集字符集合
             + ([] if test is None else get_label(test, 'r')))  # 测试集字符集合
    labels = (get_label(train, 'y')
              + ([] if valid is None else get_label(valid, 'y'))
              + ([] if test is None else get_label(test, 'y')))
    featvs = (get_label(train, 'F')
              + ([] if valid is None else get_label(valid, 'F'))
              + ([] if test is None else get_label(test, 'F')))

    feature_to_freq = defaultdict(int)
    for sentc_feats in featvs:  # feats一句话的特征列表
        for feats in sentc_feats:  # ftv一句中某个字的特征集合
            for feat in feats:  # ft这句话中某个字的按模板生成的某个特征
                feature_to_freq[feat] += 1
    features_to_id = {OOV: 1}
    cur_idx = 2
    for sentc_feats in featvs:
        for feats in sentc_feats:
            for ft in feats:
                if (ft not in features_to_id):
                    if feature_to_freq[ft] > threshold:
                        # 只保留出现次数大于一定频次的特征
                        features_to_id[ft] = cur_idx
                        cur_idx += 1

    word_to_id = {}
    cur_idx = 1
    # print 'construct dict!!'
    for sentc in words:
        for t in sentc:
            if t not in word_to_id:
                word_to_id[t] = cur_idx
                cur_idx += 1

    label_to_id = {}
    cur_idx = 1
    for sent_lbs in labels:
        for l in sent_lbs:
            if l not in label_to_id:
                label_to_id[l] = cur_idx
                cur_idx += 1
    return features_to_id, word_to_id, label_to_id


def convdata_helper(sentc, fields, repre, anno):
    """
    According to the parameter 'repre', convert sentc to a given format.
    If: 'repre' == 'char', add key-value 'r' = char
        'repre' == 'charpos', add key-value 'r' = char+pos
    The 'pos' is the position of char in a Chinese word after segmentation.

    1.repre == char: sentc = [{'w': x, 'r': c, 'y': y, 'F': []} * n]
    2.repre == charpos: sentc = [{'w': x, 'r': c+pos, 'y': y, 'F': []} * n]
    """
    if repre == 'char' and anno is None:
        for item in sentc:
            item['r'] = item['w']
    else:
        sent = ''.join(item['w'] for item in sentc)
        token_tag_list = jieba.cut(sent)
        count = 0
        for token_str in token_tag_list:
            for i, char in enumerate(token_str):
                if repre == 'charpos':
                    r = char + str(i)
                else:
                    raise ValueError(
                        'representation cannot take value %s!\n' % repre)
                if anno is None:
                    # fields = (char, r, labels[count])
                    sentc[count]['r'] = r
                else:
                    raise ValueError(
                        'annotation cannot take value %s!\n' % anno)
                count += 1
    return sentc


def apply_feature_templates(sntc):
    """
    Apply feature templates, generate feats for each sentence.
    """
    if len(sntc[0]) == 4:
        # {'w': x, 'r':r, 'y': y, 'F': []}
        fields_template = 'w r y F'
    elif len(sntc[0]) == 5:
        # {'w': x, 'r': r, 'p': p, 'y': y, 'F': []}
        fields_template = 'w r p y F'
    # elif len(sntc[0]) == 5:
    #     fields_template = 'w r s p y'
    else:
        raise ValueError('Unknow representation!\n')
    if fields_template == 'w r y F':
        features = feature_extractor(
            # readiter(sntc, fields_template.split(' ')),
            sntc,
            templates=templates)
    elif fields_template == 'w r p y F':
        features = feature_extractor(
            # readiter(sntc, fields_template.split(' ')),
            sntc,
            templates=pmi_templates)
    else:
        raise ValueError('Unknow input fields!\n')
    return features


def conv_sentc(X, Y, word2idx, label2idx):
    sentc = [word2idx.get(w) for w in X]
    label = [label2idx.get(l) for l in Y]
    return sentc, label


def conv_feats(F, feat2idx, max_len=MAX_LEN):
    feat_num = len(F[0])
    sent_len = len(F)
    feats = []
    for feat in F:
        feats.append([feat2idx.get(f, 1) for f in feat])
    for i in xrange(max_len - sent_len):
        feats.append([0] * feat_num)
    return feats


def conv_corpus(sentcs, featvs, labels, word2idx, feat2idx, label2idx, max_len=MAX_LEN):
    """
    Converts the list of sentences and labelSeq. After conversion, it will
    returns a 2D tensor which contains word's numeric id sequences, and a 3D
    tensor which contains the one-hot label vectors. All of these tensors have
    been padded(post 0) by given MAX_LEN.

    The shape of returned 2D tensor is (len(sentcs), MAX_LEN), while the 3D
    tensor's is (len(sentcs), MAX_LEN, len(label2idx) + 1).

    # Parameters
        sentcs: the list of corpus' sentences.
        featvs: the list of feats for sentences.
        labels: the list of sentcs's label sequences.
        word2idx: the vocabulary of words, a map.
        word2idx: the vocabulary of labels, a map.
        max_len: the maximum length of input sentence.

    # Returns
        new_sentcs: 2D tensor of input corpus
        new_featvs: 3D tensor of input corpus' features
        new_labels: 2D tensor of input corpus' label sequences
    """
    assert len(sentcs) == len(
        labels), "The length of input sentences and labels not equal."
    assert len(sentcs) == len(
        featvs), "The length of input sentences and labels not equal."
    new_sentcs = []
    new_labels = []
    new_featvs = []
    for sentc, feats, label in zip(sentcs, featvs, labels):
        sentc, label = conv_sentc(sentc, label, word2idx, label2idx)
        new_sentcs.append(sentc)
        new_labels.append(label)
        new_featvs.append(conv_feats(feats, feat2idx, max_len))
    new_sentcs = pad_sequences(new_sentcs, maxlen=max_len, padding='post')
    new_labels = pad_sequences(new_labels, maxlen=max_len, padding='post')
    new_featvs = np.array(new_featvs)
    return new_sentcs, new_featvs, new_labels


def read_corpus(fn, mode, anno=None, has_label=True, fields=fields):
    """
    Reads corpus file, then returns the list of sentences and labelSeq.

    # Parameters
        mode: char/charpos
        anno: has to be None
        fields: the input fields

    # Returns
        corpus: the list of corpus' sentences, each sentence is a list of
                tuple '(char, lexical, label)'
        length: the length of each sentence in corpus
        max_len: the maximum length of sentences
    """
    if fn is None:
        return None
    with cs.open(fn, encoding='utf-8') as src:
        stream = src.read().strip().split('\n\n')
        corpus = []
        max_len = MAX_LEN
        for line in stream:
            line = line.strip().split('\n')
            max_len = max(max_len, len(line))
            sentc = []
            for e in line:
                token = e.split()
                assert len(token) == len(fields)
                # depends on fields, item could be like:
                # 1. {'w': x, 'y': y, 'F': []}
                # 2. {'w': x, 'p': p, 'y':y, 'F': []}
                item = {'F': []}  # F field reserved for feats
                for i in range(len(fields)):
                    item[fields[i]] = token[i]
                sentc.append(item)
            X = convdata_helper(sentc, fields, mode, anno)
            features = apply_feature_templates(X)
            corpus.append(features)
        length = [len(sent) for sent in corpus]
        length = np.asarray(length, dtype=np.int32)
        return corpus, length, max_len


def unfold_corpus(corpus):
    """
    Unfolds a corpus, converts it's sentences from a list of
    '(char, lexical, label)' into 3 independent lists, the lexical words list
    teh labels list and the features list.

    # Return
        sentcs: a list of sentences, each sentence is a list of lexcial words
        featvs: a list of features list,shape(sent, word, feats)
        labels: a list of labels' sequences
    """
    sentcs = []
    labels = []
    featvs = []
    for sent in corpus:
        sentcs.append([item['r'] for item in sent])
        labels.append([item['y'] for item in sent])
        featvs.append([item['F'] for item in sent])

    return sentcs, featvs, labels


def pretreatment(train_fn, valid_fn, test_fn, threshold=0, emb_type='char',
                 anno=None, test_label=True, fields=fields):
    """
    """
    print "###################################################################"
    print "# Pretreatment process."
    print "###################################################################"
    # Step 1: Read the train, valid and test file
    train_corpus, train_lens, train_max_len = read_corpus(
        train_fn, emb_type, anno, test_label, fields=fields)
    valid_corpus, valid_lens, valid_max_len = read_corpus(
        valid_fn, emb_type, anno, test_label, fields=fields)
    test_corpus, test_lens, test_max_len = read_corpus(
        test_fn, emb_type, anno, test_label, fields=fields)
    # Get maximum length of sentence
    max_len = max(train_max_len, valid_max_len, test_max_len)
    # Step 2: Generate dicts from corpus
    dict_feat, dict_lex, dict_y = create_dicts(
        train_corpus, valid_corpus, test_corpus, threshold, emb_type, anno)
    dic = {'words2idx': dict_lex, 'label2idx': dict_y, 'feats2idx': dict_feat}
    train = (train_corpus, train_lens)
    valid = (valid_corpus, valid_lens)
    test = (test_corpus, test_lens)
    return train, valid, test, dic, max_len
