#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 2017

@author: Heshenghuan (heshenghuan@sina.com)
http://github.com/heshenghuan
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf


def init_variable(shape, name=None):
    initial = tf.random_uniform(shape, -0.01, 0.01)
    return tf.Variable(initial, name=name)


def batch_index(length, batch_size, n_iter=100, shuffle=True):
    index = range(length)
    for j in xrange(n_iter):
        if shuffle:
            np.random.shuffle(index)
        for i in xrange(int(length / batch_size) + 1):
            yield index[i * batch_size: (i + 1) * batch_size]


class linear_chain_CRF():

    def __init__(self, nb_words, emb_dim, emb_matrix, feat_size, nb_classes,
                 time_steps, batch_size=None, templates=None, l2_reg=0.,
                 fine_tuning=False):
        self.nb_words = nb_words
        self.emb_dim = emb_dim
        self.feat_size = feat_size
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.l2_reg = l2_reg
        self.fine_tuning = fine_tuning

        if self.fine_tuning:
            self.emb_matrix = tf.Variable(
                emb_matrix, dtype=tf.float32, name="embeddings")
        else:
            self.emb_matrix = tf.constant(
                emb_matrix, dtype=tf.float32, name="embeddings")

        with tf.name_scope('inputs'):
            self.X = tf.placeholder(
                tf.int32, shape=[None, self.time_steps, len(templates)],
                name='X_placeholder')
            self.Y = tf.placeholder(
                tf.int32, shape=[None, self.time_steps],
                name='Y_placeholder')
            self.X_len = tf.placeholder(
                tf.int32, shape=[None, ], name='X_len_placeholder')
            self.keep_prob = tf.placeholder(tf.float32, name='output_dropout')
        self.build()
        return

    def build(self):
        with tf.name_scope('weights'):
            self.W = tf.get_variable(
                shape=[self.feat_size + 1, self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='weights'
                # regularizer=tf.contrib.layers.l2_regularizer(0.001)
            )

        with tf.name_scope('biases'):
            self.b = tf.Variable(tf.zeros([self.nb_classes], name="bias"))
        return

    def inference(self, X, X_len, reuse=None):
        with tf.name_scope('score'):
            # The weight matrix is treated as an embedding matrix
            # Using lookup & reduce_sum to complete calculation of unary score
            features = tf.nn.embedding_lookup(self.W, X)
            feat_vec = tf.reduce_sum(features, axis=2)
            feat_vec = tf.reshape(feat_vec, [-1, self.nb_classes])
            scores = feat_vec + self.b
            # scores = tf.nn.softmax(scores)
            scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
        return scores

    def get_batch_data(self, x, y, l, batch_size, shuffle=True):
        for index in batch_index(len(y), batch_size, 1, shuffle):
            feed_dict = {
                self.X: x[index],
                self.Y: y[index],
                self.X_len: l[index],
            }
            yield feed_dict, len(index)

    def test_unary_score(self):
        return self.inference(self.X, reuse=True)

    def loss(self, pred):
        with tf.name_scope('loss'):
            log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
                pred, self.Y, self.X_len)
            cost = tf.reduce_mean(-log_likelihood)
            reg = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.b)
            if self.fine_tuning:
                reg += tf.nn.l2_loss(self.emb_matrix)
            cost += reg * self.l2_reg
            return cost

    def accuracy(self, num, pred, y, y_lens, trans_matrix):
        """
        Given predicted unary_scores, using viterbi_decode find the best tags
        sequence. Then count the correct labels and total labels.
        """
        correct_labels = 0
        total_labels = 0
        for i in xrange(num):
            p_len = y_lens[i]
            unary_scores = pred[i][:p_len]
            gold = y[i][:p_len]
            tags_seq, _ = tf.contrib.crf.viterbi_decode(
                unary_scores, trans_matrix)
            correct_labels += np.sum(np.equal(tags_seq, gold))
            total_labels += p_len
        return (correct_labels, total_labels)

    def run(
        self,
        train_x, train_y, train_lens,
        valid_x, valid_y, valid_lens,
        test_x, test_y, test_lens,
        FLAGS=None
    ):
        if FLAGS is None:
            print "FLAGS ERROR"
            sys.exit(0)

        self.lr = FLAGS.lr
        self.training_iter = FLAGS.train_steps
        self.train_file_path = FLAGS.train_data
        self.test_file_path = FLAGS.valid_data
        self.display_step = FLAGS.display_step

        # predication & cost-calculation
        pred = self.inference(self.X, self.X_len)
        cost = self.loss(pred)

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(cost, global_step=global_step)

        # with tf.name_scope('summary'):
        #     localtime = time.strftime("%X %Y-%m-%d", time.localtime())
        #     Summary_dir = FLAGS.log_dir + localtime

        #     info = 'batch-{}, lr-{}, l2_reg-{}'.format(
        #         self.batch_size, self.lr, self.l2_reg)
        #     info = info + '\n' + self.train_file_path + '\n' + \
        #         self.test_file_path + '\n' + 'Method: linear-chain CRF'
        #     train_acc = tf.placeholder(tf.float32)
        #     train_loss = tf.placeholder(tf.float32)
        #     summary_acc = tf.scalar_summary('ACC ' + info, train_acc)
        #     summary_loss = tf.scalar_summary('LOSS ' + info, train_loss)
        #     summary_op = tf.merge_summary([summary_loss, summary_acc])

        #     test_acc = tf.placeholder(tf.float32)
        #     test_loss = tf.placeholder(tf.float32)
        #     summary_test_acc = tf.scalar_summary('ACC ' + info, test_acc)
        #     summary_test_loss = tf.scalar_summary('LOSS ' + info, test_loss)
        #     summary_test = tf.merge_summary(
        #         [summary_test_loss, summary_test_acc])

        #     train_summary_writer = tf.train.SummaryWriter(
        #         Summary_dir + '/train')
        #     test_summary_writer = tf.train.SummaryWriter(Summary_dir + '/test')

        with tf.name_scope('saveModel'):
            localtime = time.strftime("%X %Y-%m-%d", time.localtime())
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            save_dir = FLAGS.model_dir + localtime + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            max_acc, bestIter = 0., 0

            if self.training_iter == 0:
                saver.restore(sess, FLAGS.restore_model)

            for epoch in xrange(self.training_iter):

                for train, num in self.get_batch_data(train_x, train_y, train_lens, self.batch_size):
                    _, step, trans_matrix, loss, predication = sess.run(
                        [optimizer, global_step, self.transition, cost, pred],
                        feed_dict=train)
                    correct, total = self.accuracy(
                        num, predication, train[self.Y],
                        train[self.X_len], trans_matrix)
                    acc = float(correct) / total
                    # summary = sess.run(summary_op, feed_dict={
                    #                    train_loss: loss, train_acc: acc})
                    # train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)
                saver.save(sess, save_dir, global_step=step)

                if epoch % self.display_step == 0:
                    rd, loss, correct, total = 0, 0., 0, 0
                    for valid, num in self.get_batch_data(valid_x, valid_y, valid_lens, self.batch_size):
                        trans_matrix, _loss, predication = sess.run(
                            [self.transition, cost, pred], feed_dict=valid)
                        loss += _loss
                        tmp = self.accuracy(
                            num, predication, valid[self.Y],
                            valid[self.X_len], trans_matrix)
                        correct += tmp[0]
                        total += tmp[1]
                        rd += 1
                    loss /= rd
                    acc = float(correct) / total
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step
                    # summary = sess.run(summary_test, feed_dict={
                    #                    test_loss: loss, test_acc: acc})
                    # test_summary_writer.add_summary(summary, step)
                    print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    print 'Iter {}: valid loss(avg)={:.6f}, valid acc(avg)={:.6f}'.format(step, loss, acc)
                    print 'round {}: max_acc={} BestIter={}\n'.format(epoch, max_acc, bestIter)
            print 'Optimization Finished!'

            # test process
            pred_test_y = []
            acc, loss, rd = 0., 0., 0
            correct_labels, total_labels = 0, 0
            for test, num in self.get_batch_data(test_x, test_y, test_lens, self.batch_size, shuffle=False):
                trans_matrix, _loss, predication = sess.run(
                    [self.transition, cost, pred], feed_dict=test)
                loss += _loss
                rd += 1
                for i in xrange(num):
                    p_len = test[self.X_len][i]
                    unary_scores = predication[i][:p_len]
                    gold = test[self.Y][i][:p_len]
                    tags_seq, _ = tf.contrib.crf.viterbi_decode(
                        unary_scores, trans_matrix)
                    correct_labels += np.sum(np.equal(tags_seq, gold))
                    total_labels += p_len
                    pred_test_y.append(tags_seq)
            acc = float(correct_labels) / total_labels
            loss /= rd
            return pred_test_y, loss, acc
