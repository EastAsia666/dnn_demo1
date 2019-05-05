# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from .data_reader_new import *
from sklearn.metrics import roc_auc_score
from .adabound import AdaBoundOptimizer

from .ops import *
from .utils import *


class DNN(object):
    model_name = "DNN"  # name for checkpoint

    def __init__(self, sess, all_cnt, epoch, batch_size, dataset_name, checkpoint_dir, result_dir, log_dir):
        self.data_dir = os.path.join("./data", dataset_name)
        self.sess = sess
        self.all_cnt = all_cnt
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        # parameters
        self.input_dim = 4011
        self.y_dim = 2
        # train
        self.optimizer_name = 'sgd'
        self.learning_rate = 0.01
        self.beta1 = 0.9
        #self.lamda = 0.005
        self.lamda = 0.05
        self.weight_decay = 0.01
        self.eval_train = check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_eval'
        # test
        self.sample_num = 1

        # get number of batches for a single epoch
        self.num_batches = self.all_cnt // self.batch_size
        self.loss_result = check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_loss'

    def classifier(self, x, is_training=True, reuse=False):
        # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
        # Architecture : (64)5c2s-(128)5c2s_BL-FC1024_BL-FC128_BL-FC12S’
        # All layers except the last two layers are shared by discriminator
        # Number of nodes in the last layer is reduced by half. It gives better results.
        with tf.variable_scope("classifier", reuse=reuse):
            net = lrelu(linear(x, 512, scope='c_fc1', is_training=is_training, drop_out_keep=0.5))
            net = lrelu(bn(linear(net, 256, scope='c_fc2', is_training=is_training, drop_out_keep=0.5),
                           is_training=is_training, scope='c_bn2'))
            net = lrelu(bn(linear(net, 128, scope='c_fc3', is_training=is_training, drop_out_keep=0.5),
                           is_training=is_training, scope='c_bn3'))
            net = lrelu(bn(linear(net, 64, scope='c_fc4', is_training=is_training, drop_out_keep=0.5),
                           is_training=is_training, scope='c_bn4'))
            out_logit = linear(net, self.y_dim, scope='c_fc5', is_training=is_training)
            out = tf.nn.sigmoid(out_logit)

            return out, out_logit
    def prepare_optimizer(self, optimizer_name, _global_step):
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.learning_rate,
            global_step=_global_step,
            decay_steps=1000,
            decay_rate=.95,
            staircase=True,
        )
        if optimizer_name == "adabound":
            return AdaBoundOptimizer(learning_rate=learning_rate)
        elif optimizer_name == "amsbound":
            return AdaBoundOptimizer(learning_rate=learning_rate, amsbound=True)
        elif optimizer_name == "adam":
            return tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_name == "adagrad":
            return tf.train.AdagradOptimizer(learning_rate=learning_rate)
        elif optimizer_name == "momentum":
            return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=1e-6, use_nesterov=True)
        else:
            raise NotImplementedError("[-] Unsupported Optimizer %s" % optimizer_name)

    def build_model(self):
        self.train_dataset = create_dataset(self.data_dir + '/train', batch_size=self.batch_size, shuffle=True,
                                            repeat_num=self.epoch)
        self.valid_dataset = create_dataset(self.data_dir + '/valid', batch_size=self.batch_size, shuffle=False,
                                            repeat_num=0)
        self.test_dataset = create_dataset(self.data_dir + '/test', batch_size=self.batch_size, shuffle=False,
                                           repeat_num=0)
        self.train_dataset_final = create_dataset(self.data_dir + '/train', batch_size=self.batch_size, shuffle=False,
                                                  repeat_num=0)

        self.handle = tf.placeholder(tf.string, [])
        feed_iterator = tf.data.Iterator.from_string_handle(self.handle, self.train_dataset.output_types,
                                                            self.train_dataset.output_shapes)
        # self.feas, self.labels = feed_iterator.get_next()
        self.feas, self.labels, self.heads = feed_iterator.get_next()
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
        self.c_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_code_est, labels=disc_code_tg))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        c_vars = [var for var in t_vars if 'c_' in var.name]
        l2_loss = tf.reduce_mean([tf.nn.l2_loss(v) for v in c_vars]) * self.lamda
        self.c_loss += l2_loss
        # # SGD
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     self.c_optim = tf.train.GradientDescentOptimizer \
        #         (learning_rate=self.learning_rate).minimize(self.c_loss, var_list=c_vars)
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     self.c_optim = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate).minimize(self.c_loss,
        #                                                                                          var_list=c_vars)
        global_step = tf.train.get_or_create_global_step()
        gradients = tf.gradients(self.c_loss, c_vars)
        optimizer = self.prepare_optimizer(self.optimizer_name, global_step)
        self.train_op = optimizer.apply_gradients(zip(gradients, c_vars), global_step=global_step)
        # # weight decay
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     self.c_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
        #         .minimize(self.c_loss, var_list=c_vars)
        # with tf.control_dependencies([train_op]):
        #     l2_loss = self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in c_vars])
        #     sgd = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        #     self.c_optim = sgd.minimize(l2_loss, var_list=c_vars)
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
        # tf.local_variables_initializer().run()

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
        train_iterator = self.train_dataset.make_one_shot_iterator()
        train_handle = self.sess.run(train_iterator.string_handle())
        df_loss = pd.DataFrame()
        c_loss_res = []
        start_time = time.time()
        with open(self.eval_train, 'w') as f:
            f.write(
                '\t'.join(map(str, ['epoch',  'auc_train',  'auc_valid',  'auc_test'])))
            f.write('\n')
        for epoch in range(start_epoch, self.epoch):
            for idx in range(start_batch_id, self.num_batches):
                f, y, h = self.sess.run([self.feas, self.labels, self.heads], feed_dict={self.handle: train_handle})
                # f, y, h = self.sess.run([self.feas, self.labels], feed_dict={self.handle: train_handle})
                # update c network
                _, summary_str_c, c_loss = self.sess.run(
                    [self.train_op, self.c_sum, self.c_loss],
                    feed_dict={self.inputs: f, self.y: y})
                self.writer.add_summary(summary_str_c, counter)

                counter += 1

                # display training status
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, c_loss: %.4f" \
                      % (epoch + 1, idx + 1, self.num_batches, time.time() - start_time, c_loss))
                c_loss_res.append(c_loss)
            if epoch % 5 == 0:
                self.visualize_results(epoch, on_train=True, training=True)
            else:
                #self.visualize_results(epoch, on_train=False, training=True)
		        pass
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

        df_loss['c_loss'] = c_loss_res
        df_loss.to_csv(self.loss_result, index=None, sep='\t')
        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def tools_vis(self, dataset, writer, training=False):
        res = []
        prob_all = []
        label_all = []
        iterator = dataset.make_initializable_iterator()
        handle = self.sess.run(iterator.string_handle())
        self.sess.run(iterator.initializer)
        while True:
            try:
                f, y, h = self.sess.run([self.feas, self.labels, self.heads], feed_dict={self.handle: handle})
                # f, y, h = self.sess.run([self.feas, self.labels], feed_dict={self.handle: handle})
                batch_prob = self.sess.run(self.test_sigmoid, feed_dict={self.test_inputs: f})
                prob = list(batch_prob[:, 1])
                label = list(y[:, 1])
                prob_all.extend(prob)
                label_all.extend(label)
                for i, v in enumerate(h):
                    res.append(v + '\t' + '\t'.join(map(str, [label[i], prob[i]])))
            except Exception as e:
                break
        if len(prob_all) > 0 and len(label_all) > 0:
            auc = roc_auc_score(label_all, prob_all)
        else:
            auc = ''

        if not training:
            with open(writer, 'w') as f:
                # f.write('\t'.join(['name', 'idcard', 'phone', 'loan_dt', 'create_tm', 'label', 'prob']))
                f.write('\t'.join(['label', 'prob']))
                f.write('\n')
                for i, v in enumerate(res):
                    f.write(v)
                    f.write('\n')
        return  round(auc, 3)

    def visualize_results(self, epoch, on_train=False, training=False):
        if on_train == True:
            # train
            auc_train = self.tools_vis(self.train_dataset_final,
                                                 check_folder(self.result_dir + '/' + self.model_dir)
                                                 + '/' + self.model_name + '_epoch%03d_train' % (epoch + 1), training)
        else:
            auc_train = ''

        # valid
        auc_valid = self.tools_vis(self.valid_dataset, check_folder(self.result_dir + '/' + self.model_dir)
                                             + '/' + self.model_name + '_epoch%03d_valid' % (epoch + 1), training)
        # test
        auc_test = self.tools_vis(self.test_dataset, check_folder(self.result_dir + '/' + self.model_dir)
                                           + '/' + self.model_name + '_epoch%03d_test' % (epoch + 1), training)
        if training:
            with open(self.eval_train, 'a+') as f:
                f.write('\t'.join(map(str, [epoch + 1,  auc_train,  auc_valid,  auc_test])))
                f.write('\n')

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

