#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import random
import numpy as np
import codecs as cs
import tensorflow as tf
from model import embedding_CRF
from src.parameters import MAX_LEN
from src.features import Template
from src.utils import eval_ner, read_emb_from_file
from src.parameters import MODEL_DIR, DATA_DIR, OUTPUT_DIR, LOG_DIR, EMB_DIR
from src.pretreatment import pretreatment, unfold_corpus, conv_corpus


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_data', DATA_DIR + r'weiboNER_2nd_conll.train', 'Training data file')
tf.app.flags.DEFINE_string(
    'test_data', DATA_DIR + r'weiboNER_2nd_conll.test', 'Test data file')
tf.app.flags.DEFINE_string(
    'valid_data', DATA_DIR + r'weiboNER_2nd_conll.dev', 'Validation data file')
tf.app.flags.DEFINE_string('log_dir', LOG_DIR, 'The log dir')
tf.app.flags.DEFINE_string('model_dir', MODEL_DIR, 'Models dir')
tf.app.flags.DEFINE_string('restore_model', 'None',
                           'Path of the model to restored')
# tf.app.flags.DEFINE_string("emb_dir", EMBEDDING_DIR, "Embeddings dir")
tf.app.flags.DEFINE_string("emb_type", "char", "Embeddings type: char/charpos")
tf.app.flags.DEFINE_string(
    "emb_file", EMB_DIR + "/weibo_charpos_vectors", "Embeddings file")
tf.app.flags.DEFINE_integer("emb_dim", 100, "embedding size")
tf.app.flags.DEFINE_string("output_dir", OUTPUT_DIR, "Output dir")
tf.app.flags.DEFINE_integer(
    "feat_thresh", 0, "Only keep feats which occurs more than 'thresh' times.")
# tf.app.flags.DEFINE_boolean('only_test', False, 'Only do the test')
tf.app.flags.DEFINE_float("lr", 0.002, "learning rate")
tf.app.flags.DEFINE_boolean(
    'fine_tuning', False, 'Whether fine-tuning the embeddings')
tf.app.flags.DEFINE_boolean(
    'eval_test', True, 'Whether evaluate the test data.')
tf.app.flags.DEFINE_boolean(
    'test_anno', True, 'Whether the test data is labeled.')
tf.app.flags.DEFINE_integer("max_len", MAX_LEN,
                            "max num of tokens per query")
tf.app.flags.DEFINE_integer("nb_classes", 17, "Tagset size")
# tf.app.flags.DEFINE_integer("hidden_dim", 100, "hidden unit number")
tf.app.flags.DEFINE_integer("batch_size", 200, "num example per mini batch")
tf.app.flags.DEFINE_integer("train_steps", 50, "trainning steps")
tf.app.flags.DEFINE_integer("display_step", 1, "number of test display step")
tf.app.flags.DEFINE_float("l2_reg", 0.0001, "L2 regularization weight")
tf.app.flags.DEFINE_boolean(
    'log', True, 'Whether to record the TensorBoard log.')
tf.app.flags.DEFINE_string("template", r"template", "Feature templates")


def convert_id_to_word(corpus, idx2label):
    return [[idx2label.get(word, 'O') for word in sentence]
            for sentence in corpus]


def evaluate(predictions, groundtruth=None):
    if groundtruth is None:
        return None, predictions
    # conlleval(predictions, groundtruth,
    results = eval_ner(predictions, groundtruth)
    #          folder + '/current.valid.txt', folder)
    # error_analysis(words, predictions, groundtruth, idx2word)
    return results, predictions


def write_prediction(filename, lex_test, pred_test):
    with cs.open(filename, 'w', encoding='utf-8') as outf:
        for sent_w, sent_l in zip(lex_test, pred_test):
            assert len(sent_w) == len(sent_l)
            for w, l in zip(sent_w, sent_l):
                outf.write(w + '\t' + l + '\n')
            outf.write('\n')


def test_evaluate(sess, unary_score, test_sequence_length, transMatrix, inp,
                  tX, tY):
    batchSize = FLAGS.batch_size
    totalLen = tX.shape[0]
    numBatch = int((tX.shape[0] - 1) / batchSize) + 1
    correct_labels = 0
    total_labels = 0
    pred_labels = []
    for i in range(numBatch):
        endOff = (i + 1) * batchSize
        if endOff > totalLen:
            endOff = totalLen
        y = tY[i * batchSize:endOff]
        feed_dict = {inp: tX[i * batchSize:endOff]}
        unary_score_val, test_sequence_length_val = sess.run(
            [unary_score, test_sequence_length], feed_dict)
        for tf_unary_scores_, y_, sequence_length_ in zip(
                unary_score_val, y, test_sequence_length_val):
            # print("seg len:%d" % (sequence_length_))
            tf_unary_scores_ = tf_unary_scores_[:sequence_length_]
            y_ = y_[:sequence_length_]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                tf_unary_scores_, transMatrix)
            # Evaluate word-level accuracy.
            pred_labels.append(viterbi_sequence)
            correct_labels += np.sum(np.equal(viterbi_sequence, y_))
            total_labels += sequence_length_
    accuracy = 100.0 * correct_labels / float(total_labels)
    print("Accuracy: %.2f%%" % accuracy)
    return pred_labels


def train(total_loss):
    return tf.train.AdamOptimizer(FLAGS.lr).minimize(total_loss)


def main(_):
    np.random.seed(1337)
    random.seed(1337)

    print "#" * 67
    print "# Loading data from:"
    print "#" * 67
    print "Train:", FLAGS.train_data
    print "Valid:", FLAGS.valid_data
    print "Test: ", FLAGS.test_data
    print "Feature threshold:", FLAGS.feat_thresh

    # Choose fields templates & features templates
    template = Template(FLAGS.template)
    # pretreatment process: read, split and create vocabularies
    train_set, valid_set, test_set, dicts, max_len = pretreatment(
        FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data,
        threshold=FLAGS.feat_thresh, emb_type='char',
        test_label=FLAGS.test_anno, fields=FLAGS.fields)

    # Reset the maximum sentence's length
    # max_len = max(MAX_LEN, max_len)
    FLAGS.max_len = max_len

    # unfold these corpus
    train_corpus, train_lens = train_set
    valid_corpus, valid_lens = valid_set
    test_corpus, test_lens = test_set
    train_sentcs, train_featvs, train_labels = unfold_corpus(train_corpus)
    valid_sentcs, valid_featvs, valid_labels = unfold_corpus(valid_corpus)
    test_sentcs, test_featvs, test_labels = unfold_corpus(test_corpus)

    # vocabularies
    feats2idx = dicts['feats2idx']
    words2idx = dicts['words2idx']
    label2idx = dicts['label2idx']
    FLAGS.label2idx = label2idx
    FLAGS.words2idx = words2idx
    FLAGS.feats2idx = feats2idx
    FLAGS.feat_size = len(feats2idx)

    print "Lexical word size:     %d" % len(words2idx)
    print "Label size:            %d" % len(label2idx)
    print "Features size:         %d" % len(feats2idx)
    print "-------------------------------------------------------------------"
    print "Training data size:    %d" % len(train_corpus)
    print "Validation data size:  %d" % len(valid_corpus)
    print "Test data size:        %d" % len(test_corpus)
    print "Maximum sentence len:  %d" % FLAGS.max_len

    del train_corpus
    del valid_corpus
    del test_corpus

    # neural network's output_dim
    nb_classes = len(label2idx) + 1
    FLAGS.nb_classes = max(nb_classes, FLAGS.nb_classes)

    # Embedding layer's input_dim
    nb_words = len(words2idx)
    FLAGS.nb_words = nb_words
    FLAGS.in_dim = FLAGS.nb_words + 1

    # load embeddings from file
    print "#" * 67
    print "# Reading embeddings from file: %s" % (FLAGS.emb_file)
    emb_mat, idx_map = read_emb_from_file(FLAGS.emb_file, words2idx)
    FLAGS.emb_dim = max(emb_mat.shape[1], FLAGS.emb_dim)
    print "embeddings' size:", emb_mat.shape
    if FLAGS.fine_tuning:
        print "The embeddings will be fine-tuned!"

    idx2label = dict((k, v) for v, k in FLAGS.label2idx.iteritems())
    # idx2words = dict((k, v) for v, k in FLAGS.words2idx.iteritems())

    # convert corpus from string seq to numeric id seq with post padding 0
    print "Preparing training, validate and testing data."
    train_X, train_F, train_Y = conv_corpus(
        train_sentcs, train_featvs, train_labels,
        words2idx, feats2idx, label2idx, max_len=max_len)
    valid_X, valid_F, valid_Y = conv_corpus(
        valid_sentcs, valid_featvs, valid_labels,
        words2idx, feats2idx, label2idx, max_len=max_len)
    test_X, test_F, test_Y = conv_corpus(
        test_sentcs, test_featvs, test_labels,
        words2idx, feats2idx, label2idx, max_len=max_len)

    del train_sentcs, train_featvs, train_labels
    del valid_sentcs, valid_featvs, valid_labels
    # del test_sentcs, test_featvs, test_labels

    print "#" * 67
    print "Training arguments"
    print "#" * 67
    print "L2 regular:    %f" % FLAGS.l2_reg
    print "nb_classes:    %d" % FLAGS.nb_classes
    print "Batch size:    %d" % FLAGS.batch_size
    print "Embedding dim: %d" % FLAGS.emb_dim
    # print "Hidden layer:  %d" % FLAGS.hidden_dim
    print "Train epochs:  %d" % FLAGS.train_steps
    print "Learning rate: %f" % FLAGS.lr

    print "#" * 67
    print "Training process start."
    print "#" * 67

    model = embedding_CRF(
        FLAGS.nb_words, FLAGS.emb_dim, emb_mat, FLAGS.feat_size,
        FLAGS.nb_classes, FLAGS.max_len, FLAGS.fine_tuning,
        FLAGS.batch_size, len(template.template), FLAGS.l2_reg)

    pred_test, test_loss, test_acc = model.run(
        train_X, train_F, train_Y, train_lens,
        valid_X, valid_F, valid_Y, valid_lens,
        test_X, test_F, test_Y, test_lens,
        FLAGS)

    print "Test loss: %f, accuracy: %f" % (test_loss, test_acc)
    pred_test = [pred_test[i][:test_lens[i]] for i in xrange(len(pred_test))]
    pred_test_label = convert_id_to_word(pred_test, idx2label)
    if FLAGS.eval_test:
        res_test, pred_test_label = evaluate(pred_test_label, test_labels)
        print "Test F1: %f, P: %f, R: %f" % (res_test['f1'], res_test['p'], res_test['r'])
    # original_text = [[item['w'] for item in sent] for sent in test_corpus]
    write_prediction(FLAGS.output_dir + 'prediction.utf8',
                     test_sentcs, pred_test_label)


if __name__ == "__main__":
    tf.app.run()
