# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np

from .ops import *
from .utils import *


class LR(object):
    model_name = "LR"  # name for checkpoint

    def __init__(self, sess, epoch, batch_size, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.data_X, self.data_y, self.scaler, self.flist, self.old_flist, \
        self.test_X, self.test_Y, self.test_head, \
        self.dev_X, self.dev_Y, self.dev_head, self.train_head = load_common(dataset_name)
        # parameters
        self.input_dim = self.data_X.shape[1]
        self.y_dim = self.data_y.shape[1]
        # train
        self.lamd = 0.4
        self.learning_rate = 0.0001
        self.beta1 = 0.9

        # test
        self.sample_num = 1

        # get number of batches for a single epoch
        self.num_batches = len(self.data_X) // self.batch_size
        self.loss_result = check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_loss'

    def classifier(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)5c2s-(128)5c2s_BL-FC1024_BL-FC128_BL-FC12S’
        # All layers except the last two layers are shared by discriminator
        # Number of nodes in the last layer is reduced by half. It gives better results.
        with tf.variable_scope("classifier", reuse=reuse):
            # net = lrelu(linear(x, 64, scope='c_fc1', is_training=is_training, drop_out_keep=1.0),
            #             is_training=is_training, scope='c_bn1')
            # net = lrelu(bn(linear(net, 32, scope='c_fc2', is_training=is_training, drop_out_keep=1.0),
            #                is_training=is_training, scope='c_bn2'))
            out_logit = linear(x, self.y_dim, scope='c_fc1', is_training=is_training)
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit

    def build_model(self):
        # some parameters
        dims = [self.input_dim]

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [None] + dims, name='train_data')
        self.test_inputs = tf.placeholder(tf.float32, [None] + dims, name='test_data')

        # labels
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        """ Loss Function """

        # Loss
        code_fake, code_logit_fake = self.classifier(self.inputs, is_training=True, reuse=False)

        # discrete code : categorical
        disc_code_est = code_logit_fake[:, :self.y_dim]
        disc_code_tg = self.y[:, :self.y_dim]
        c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_code_est, labels=disc_code_tg))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        c_vars = [var for var in t_vars if 'c_' in var.name]
        l2_loss_holder = [tf.nn.l2_loss(v) for v in c_vars]
        self.c_loss = c_loss + tf.reduce_mean(l2_loss_holder) * self.lamd

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.c_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.c_loss, var_list=c_vars)

        """" test """
        # for test
        self.test_sigmoid, test_logit = self.classifier(self.test_inputs, is_training=False, reuse=True)

        """ Summary """
        c_loss_sum = tf.summary.scalar("c_loss", self.c_loss)

        # final summary operations
        self.c_sum = tf.summary.merge([c_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        df_loss = pd.DataFrame()
        c_loss_res = []
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            seed = np.random.randint(1, 999)
            np.random.seed(seed)
            np.random.shuffle(self.data_X)
            np.random.seed(seed)
            np.random.shuffle(self.data_y)
            for idx in range(start_batch_id, self.num_batches):
                batch_data = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                # generate code
                batch_labels = self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]

                # update c network
                _, summary_str_c, c_loss = self.sess.run(
                    [self.c_optim, self.c_sum, self.c_loss],
                    feed_dict={self.inputs: batch_data, self.y: batch_labels})
                self.writer.add_summary(summary_str_c, counter)

                counter += 1

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, c_loss: %.4f" \
                      % (epoch + 1, idx + 1, self.num_batches, time.time() - start_time, c_loss))
                c_loss_res.append(c_loss)

                # save training results for every 300 steps
                if np.mod(counter, 300) == 0:
                    pass
                    # samples = self.sess.run(self.fake_images,
                    #                         feed_dict={self.z: self.sample_z, self.y: self.test_codes})
                    # tot_num_samples = min(self.sample_num, self.batch_size)
                    # manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    # manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    # save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                    #             './' + check_folder(
                    #                 self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                    #                 epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)


        df_loss['c_loss'] = c_loss_res
        df_loss.to_csv(self.loss_result, index=None, sep='\t')
        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch, test_size=2048):
        # train
        train_prob = []
        for idx in range(0, self.data_X.shape[0] // test_size + 1):
            batch_data = self.data_X[idx * test_size:(idx + 1) * test_size]
            batch_prob = self.sess.run(self.test_sigmoid, feed_dict={self.test_inputs: batch_data})
            train_prob = train_prob + list(batch_prob[:, 1])
        df_train = self.train_head.copy()
        df_train['prob'] = train_prob
        df_train['label'] = self.data_y[:, 1]
        df_train.to_csv(
            check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d_train' % (
                    epoch + 1), index=None, sep='\t')
        # valid
        dev_prob = []
        for idx in range(0, self.dev_X.shape[0] // test_size + 1):
            batch_data = self.dev_X[idx * test_size:(idx + 1) * test_size]
            batch_prob = self.sess.run(self.test_sigmoid, feed_dict={self.test_inputs: batch_data})
            dev_prob = dev_prob + list(batch_prob[:, 1])
        df_dev = self.dev_head.copy()
        df_dev['prob'] = dev_prob
        df_dev.to_csv(
            check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d_dev' % (
                    epoch + 1), index=None, sep='\t')
        # test
        test_prob = []
        for idx in range(0, self.test_X.shape[0] // test_size + 1):
            batch_data = self.test_X[idx * test_size:(idx + 1) * test_size]
            batch_prob = self.sess.run(self.test_sigmoid, feed_dict={self.test_inputs: batch_data})
            test_prob = test_prob + list(batch_prob[:, 1])
        df_test = self.test_head.copy()
        df_test['prob'] = test_prob
        df_test.to_csv(
            check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d_test' % (
                    epoch + 1), index=None, sep='\t')

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
