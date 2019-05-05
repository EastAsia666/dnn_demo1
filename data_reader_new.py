# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
import numpy as np
import sys


def parse_tfrecords(example, num_class=2):
    key_to_features = {
        # "head": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
        "feature": tf.FixedLenFeature([4011], tf.float32)
    }
    parsed = tf.parse_single_example(example, key_to_features)
    label = tf.one_hot(parsed['label'], num_class)
    label = tf.cast(label, tf.int8)
    fea = parsed['feature'][:3969]
    fea = tf.cast(fea, tf.float32)
    head = parsed['label']
    head = parsed['label']
#     return tf.reshape(fea, [63, 63, 1]), label, head
    return tf.reshape(fea, [63, 63, 1]), label, head


def create_dataset(f_names, batch_size=64, shuffle=False, repeat_num=0):
    dataset = tf.data.TFRecordDataset(f_names)
    if repeat_num > 0:
        dataset = dataset.repeat(repeat_num)
    dataset = dataset.map(parse_tfrecords)
    if shuffle:
        dataset = dataset.shuffle(81920 + 3 * batch_size)
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == "__main__":
    train_dataset = create_dataset(["/home/users/congziqi/spark/pyspark/toTF/fq_dnn_train.tfrecord"],
                                   batch_size=64, shuffle=True, repeat_num=2)
    val_dataset = create_dataset(["/home/users/congziqi/spark/pyspark/toTF/fq_dnn_valid.tfrecord"],
                                 batch_size=1024)
    handle = tf.placeholder(tf.string, [])
    feed_iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types,
                                                        train_dataset.output_shapes)
    feas, labels, heads = feed_iterator.get_next()
    train_iterator = train_dataset.make_one_shot_iterator()
    val_iterator = val_dataset.make_initializable_iterator()

    with tf.Session() as sess:
        train_handle = sess.run(train_iterator.string_handle())
        while 1:
            f, y, h = sess.run([feas, labels, heads], feed_dict={handle: train_handle})
            print(f)
            print(y)
            print(h)

