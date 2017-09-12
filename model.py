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

    def __init__(self, feat_size, nb_classes, time_steps,
                 batch_size=None, templates=1, l2_reg=0.):
        self.feat_size = feat_size
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.l2_reg = l2_reg

        with tf.name_scope('inputs'):
            self.X = tf.placeholder(
                tf.int32, shape=[None, self.time_steps, templates],
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
            self.b = tf.get_variable(
                shape=[self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='bias'
            )
            # self.b = tf.Variable(tf.zeros([self.nb_classes], name="bias"))
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
            # if self.fine_tuning:
            #     reg += tf.nn.l2_loss(self.emb_matrix)
            cost += reg * self.l2_reg
            return cost

    def seq_score(self, pred):
        with tf.name_scope('seq_score'):
            seq_score = tf.contrib.crf.crf_sequence_score(
                pred, self.Y, self.X_len, self.transition)
            return seq_score

    def viterbi_decode(self, num, pred, y_lens, trans_matrix):
        """
        Given predicted unary_scores, using viterbi_decode find the best tags
        sequence.
        """
        labels = []
        scores = []
        for i in xrange(num):
            p_len = y_lens[i]
            unary_scores = pred[i][:p_len]
            tags_seq, tags_score = tf.contrib.crf.viterbi_decode(
                unary_scores, trans_matrix)
            labels.append(tags_seq)
            scores.append(tags_score)
        return (labels, scores)

    def accuracy(self, num, labels, y, y_lens):
        """
        Count the correct labels num and total labels num.
        """
        correct_labels = 0
        total_labels = 0
        for i in xrange(num):
            p_len = y_lens[i]
            gold = y[i][:p_len]
            tags_seq = labels[i]
            correct_labels += np.sum(np.equal(tags_seq, gold))
            total_labels += p_len
        return (correct_labels, total_labels)

    def margin_loss(self, num, labels, scores, y, y_lens, y_scores):
        """
        Calculate margin loss value.
        """
        value = 0.
        for i in xrange(num):
            p_len = y_lens[i]
            delta = np.sum(np.not_equal(labels[i], y[i][:p_len]))
            value += scores[i] + delta - y_scores[i]
        return value / num

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

        # golden tag sequences' seqscore
        y_scores = self.seq_score(pred)

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(cost, global_step=global_step)

        with tf.name_scope('summary'):
            if FLAGS.log:
                localtime = time.strftime("%Y%m%d-%X", time.localtime())
                Summary_dir = FLAGS.log_dir + localtime

                info = 'batch{}, lr{}, l2_reg{}'.format(
                    self.batch_size, self.lr, self.l2_reg)
                info += ';' + self.train_file_path + ';' + \
                    self.test_file_path + ';' + 'Method:linear-chain CRF'
                train_acc = tf.placeholder(tf.float32)
                train_loss = tf.placeholder(tf.float32)
                summary_acc = tf.summary.scalar('ACC ' + info, train_acc)
                summary_loss = tf.summary.scalar('LOSS ' + info, train_loss)
                summary_op = tf.summary.merge([summary_loss, summary_acc])

                valid_acc = tf.placeholder(tf.float32)
                valid_loss = tf.placeholder(tf.float32)
                summary_valid_acc = tf.summary.scalar('ACC ' + info, valid_acc)
                summary_valid_loss = tf.summary.scalar(
                    'LOSS ' + info, valid_loss)
                summary_valid = tf.summary.merge(
                    [summary_valid_loss, summary_valid_acc])

                train_summary_writer = tf.summary.FileWriter(
                    Summary_dir + '/train')
                valid_summary_writer = tf.summary.FileWriter(
                    Summary_dir + '/valid')

        with tf.name_scope('saveModel'):
            localtime = time.strftime("%X-%Y-%m-%d", time.localtime())
            saver = tf.train.Saver()
            save_dir = FLAGS.model_dir + localtime + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        with tf.Session() as sess:
            max_acc, bestIter = 0., 0

            if self.training_iter == 0:
                saver.restore(sess, FLAGS.restore_model)
                print "[+] Model restored from %s" % FLAGS.restore_model
            else:
                sess.run(tf.initialize_all_variables())

            for epoch in xrange(self.training_iter):

                for train, num in self.get_batch_data(train_x, train_y, train_lens, self.batch_size):
                    _, step, trans_matrix, loss, predication, gold_scores = sess.run(
                        [optimizer, global_step, self.transition, cost, pred, y_scores],
                        feed_dict=train)
                    tags_seqs, tags_scores = self.viterbi_decode(
                        num, predication, train[self.X_len], trans_matrix)
                    correct, total = self.accuracy(
                        num, tags_seqs, train[self.Y], train[self.X_len])
                    acc = float(correct) / total
                    m_loss = self.margin_loss(
                        num, tags_seqs, tags_scores, train[self.Y],
                        train[self.X_len], gold_scores)
                    if FLAGS.log:
                        summary = sess.run(summary_op, feed_dict={
                            train_loss: loss, train_acc: acc})
                        train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}, mloss={:.6f}'.format(step, loss, acc, m_loss)
                save_path = saver.save(sess, save_dir, global_step=step)
                print "[+] Model saved in file: %s" % save_path

                if epoch % self.display_step == 0:
                    rd, loss, correct, total, m_loss = 0, 0., 0, 0, 0.
                    for valid, num in self.get_batch_data(valid_x, valid_y, valid_lens, self.batch_size):
                        trans_matrix, _loss, predication, gold_scores = sess.run(
                            [self.transition, cost, pred, y_scores], feed_dict=valid)
                        loss += _loss
                        tags_seqs, tags_scores = self.viterbi_decode(
                            num, predication, valid[self.X_len], trans_matrix)
                        tmp = self.accuracy(
                            num, tags_seqs, valid[self.Y], valid[self.X_len])
                        m_loss += self.margin_loss(
                            num, tags_seqs, tags_scores, valid[self.Y],
                            valid[self.X_len], gold_scores)
                        correct += tmp[0]
                        total += tmp[1]
                        rd += 1
                    loss /= rd
                    acc = float(correct) / total
                    m_loss /= rd
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step
                    if FLAGS.log:
                        summary = sess.run(summary_valid, feed_dict={
                            valid_loss: loss, valid_acc: acc})
                        valid_summary_writer.add_summary(summary, step)
                    print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    print 'Iter {}: valid loss(avg)={:.6f}, acc(avg)={:.6f}, mloss={:.6f}'.format(step, loss, acc, m_loss)
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
                tags_seqs, tags_scores = self.viterbi_decode(
                    num, predication, test[self.X_len], trans_matrix)
                tmp = self.accuracy(
                    num, tags_seqs, test[self.Y], test[self.X_len])
                correct_labels += tmp[0]
                total_labels += tmp[1]
                pred_test_y.extend(tags_seqs)
            acc = float(correct_labels) / total_labels
            loss /= rd
            return pred_test_y, loss, acc


class embedding_CRF(linear_chain_CRF):

    def __init__(self, nb_words, emb_dim, emb_matrix, feat_size,
                 nb_classes, time_steps, fine_tuning=False,
                 batch_size=None, templates=1, l2_reg=0.):
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
            self.F = tf.placeholder(
                tf.int32, shape=[None, self.time_steps, templates],
                name='F_placeholder')
            self.X = tf.placeholder(
                tf.int32, shape=[None, self.time_steps],
                name='X_placeholder'
            )
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
                name='feat_weights'
                # regularizer=tf.contrib.layers.l2_regularizer(0.001)
            )

            self.T = tf.get_variable(
                shape=[self.emb_dim, self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='emb_weights'
            )

        with tf.name_scope('biases'):
            self.b = tf.get_variable(
                shape=[self.nb_classes],
                initializer=tf.truncated_normal_initializer(stddev=0.01),
                name='bias'
            )
            # self.b = tf.Variable(tf.zeros([self.nb_classes], name="bias"))
        return

    def inference(self, X, F, X_len, reuse=None):
        with tf.name_scope('score'):
            # The weight matrix is treated as an embedding matrix
            # Using lookup & reduce_sum to complete calculation of unary score
            features = tf.nn.embedding_lookup(self.W, F)
            feat_vec = tf.reduce_sum(features, axis=2)
            feat_vec = tf.reshape(feat_vec, [-1, self.nb_classes])

            # embedding features
            word_vec = tf.nn.embedding_lookup(self.emb_matrix, X)
            word_vec = tf.reshape(word_vec, [-1, self.emb_dim])
            scores = feat_vec + tf.matmul(word_vec, self.T) + self.b
            # scores = tf.nn.softmax(scores)
            scores = tf.reshape(scores, [-1, self.time_steps, self.nb_classes])
        return scores

    def loss(self, pred):
        '''
        Cost function.
        '''
        with tf.name_scope('loss'):
            log_likelihood, self.transition = tf.contrib.crf.crf_log_likelihood(
                pred, self.Y, self.X_len)
            cost = tf.reduce_mean(-log_likelihood)
            reg = tf.nn.l2_loss(self.W) + tf.nn.l2_loss(self.T)
            reg += tf.nn.l2_loss(self.b)
            # if self.fine_tuning:
            #     reg += tf.nn.l2_loss(self.emb_matrix)
            cost += reg * self.l2_reg
            return cost

    def get_batch_data(self, x, f, y, l, batch_size, shuffle=True):
        for index in batch_index(len(y), batch_size, 1, shuffle):
            feed_dict = {
                self.X: x[index],
                self.Y: y[index],
                self.F: f[index],
                self.X_len: l[index],
            }
            yield feed_dict, len(index)

    def run(
        self,
        train_x, train_f, train_y, train_lens,
        valid_x, valid_f, valid_y, valid_lens,
        test_x, test_f, test_y, test_lens,
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
        pred = self.inference(self.X, self.F, self.X_len)
        cost = self.loss(pred)

        with tf.name_scope('train'):
            global_step = tf.Variable(
                0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(cost, global_step=global_step)

        with tf.name_scope('saveModel'):
            localtime = time.strftime("%X %Y-%m-%d", time.localtime())
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            save_dir = FLAGS.model_dir + localtime + '/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        with tf.name_scope('summary'):
            if FLAGS.log:
                localtime = time.strftime("%Y%m%d-%X", time.localtime())
                Summary_dir = FLAGS.log_dir + localtime

                info = 'batch{}, lr{}, l2_reg{}'.format(
                    self.batch_size, self.lr, self.l2_reg)
                info += ';' + self.train_file_path + ';' + \
                    self.test_file_path + ';' + 'Method:.emb-enhance CRF'
                train_acc = tf.placeholder(tf.float32)
                train_loss = tf.placeholder(tf.float32)
                summary_acc = tf.summary.scalar('ACC ' + info, train_acc)
                summary_loss = tf.summary.scalar('LOSS ' + info, train_loss)
                summary_op = tf.summary.merge([summary_loss, summary_acc])

                valid_acc = tf.placeholder(tf.float32)
                valid_loss = tf.placeholder(tf.float32)
                summary_valid_acc = tf.summary.scalar('ACC ' + info, valid_acc)
                summary_valid_loss = tf.summary.scalar(
                    'LOSS ' + info, valid_loss)
                summary_valid = tf.summary.merge(
                    [summary_valid_loss, summary_valid_acc])

                train_summary_writer = tf.summary.FileWriter(
                    Summary_dir + '/train')
                valid_summary_writer = tf.summary.FileWriter(
                    Summary_dir + '/valid')

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            max_acc, bestIter = 0., 0

            if self.training_iter == 0:
                saver.restore(sess, FLAGS.restore_model)

            for epoch in xrange(self.training_iter):

                for train, num in self.get_batch_data(train_x, train_f, train_y, train_lens, self.batch_size):
                    _, step, trans_matrix, loss, predication = sess.run(
                        [optimizer, global_step, self.transition, cost, pred],
                        feed_dict=train)
                    tags_seqs, _ = self.viterbi_decode(
                        num, predication, train[self.X_len], trans_matrix)
                    correct, total = self.accuracy(
                        num, tags_seqs, train[self.Y], train[self.X_len])
                    acc = float(correct) / total
                    if FLAGS.log:
                        summary = sess.run(summary_op, feed_dict={
                            train_loss: loss, train_acc: acc})
                        train_summary_writer.add_summary(summary, step)
                    print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)
                saver.save(sess, save_dir, global_step=step)

                if epoch % self.display_step == 0:
                    rd, loss, correct, total = 0, 0., 0, 0
                    for valid, num in self.get_batch_data(valid_x, valid_f, valid_y, valid_lens, self.batch_size):
                        trans_matrix, _loss, predication = sess.run(
                            [self.transition, cost, pred], feed_dict=valid)
                        loss += _loss
                        tags_seqs, _ = self.viterbi_decode(
                            num, predication, valid[self.X_len], trans_matrix)
                        tmp = self.accuracy(
                            num, tags_seqs, valid[self.Y], valid[self.X_len])
                        correct += tmp[0]
                        total += tmp[1]
                        rd += 1
                    loss /= rd
                    acc = float(correct) / total
                    if acc > max_acc:
                        max_acc = acc
                        bestIter = step
                    if FLAGS.log:
                        summary = sess.run(summary_valid, feed_dict={
                            valid_loss: loss, valid_acc: acc})
                        valid_summary_writer.add_summary(summary, step)
                    print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                    print 'Iter {}: valid loss(avg)={:.6f}, acc(avg)={:.6f}'.format(step, loss, acc)
                    print 'round {}: max_acc={} BestIter={}\n'.format(epoch, max_acc, bestIter)
            print 'Optimization Finished!'

            # test process
            pred_test_y = []
            acc, loss, rd = 0., 0., 0
            correct_labels, total_labels = 0, 0
            for test, num in self.get_batch_data(test_x, test_f, test_y, test_lens, self.batch_size, shuffle=False):
                trans_matrix, _loss, predication = sess.run(
                    [self.transition, cost, pred], feed_dict=test)
                loss += _loss
                rd += 1
                tags_seqs, tags_scores = self.viterbi_decode(
                    num, predication, test[self.X_len], trans_matrix)
                tmp = self.accuracy(
                    num, tags_seqs, test[self.Y], test[self.X_len])
                correct_labels += tmp[0]
                total_labels += tmp[1]
                pred_test_y.extend(tags_seqs)
            acc = float(correct_labels) / total_labels
            loss /= rd
            return pred_test_y, loss, acc
