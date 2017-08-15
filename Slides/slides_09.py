#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# FileName:slides_09
# Created by zhuchao at 2017/8/14 上午10:16
# purpose: Slides_09 TensorFlow Input Pipeline


import tensorflow as tf
import threading
from PIL import  Image
import numpy as np


## Agenda
## Data Readers Revisited
## TFRecord
## Variable initializer
## Graph Collection
## Style Transfer

## Queues
# tf.Session objects are designed to multithreaded. -> can run ops in parallel

## tf.Coordinator and tf.train.QueueRunner

## QueueRunner
# create a number of threads cooperating to enqueue tensors in the same queue

## Coordinator
# help multiple threads stop together and report exceptions to a program that waits for them to stop

## different queues in tf
## tf.FIFOQueue
## tf.RandomShuffleQueue
## tf.PaddingFIFOQueue
## tf.PriorityQueue

BATCH_SIZE = 2
N_FEATURES = 9

q = tf.FIFOQueue(3, "float")
init = q.enqueue_many(([0., 0., 0.],))

x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])
with tf.Session() as sess:
    init.run()
    q_inc.run()
    q_inc.run()
    q_inc.run()
    q_inc.run()

## Create a queue
## tf.FIFOQueue(capacity,min_after_dequeue,dtypes,shapes=None,names=None)

## tf.Coordinator
## Can be used to manage the threads you created without queues

## Three ways to read in data
## Through tf.constant(make everything a constant) -> NO
## Feed dict Slow when client and workers are on different machines -> Maybe
## Data readers Readers allow us to load data directly into the worker process

## Different Readers for different file types
## tf.TextLineReader Outputs the lines of a file delimited by newlines
## tf.FixedLengthRecordReader
## tf.WholeFileReader
## tf.TFRecordReader
## tf.ReaderBase


## Read in files from queues
# filename_queue = tf.train.string_input_producer(["file0.csv","file1.csv"])
# reader = tf.TextLineReader()
# key,value = reader.read(filename_queue)

filename_queue = tf.train.string_input_producer(["../data/heart.csv"])
reader = tf.TextLineReader(skip_header_lines=1)

key, value = reader.read(filename_queue)
print(key, value)
# with tf.Session() as sess:
#     ## tf.train.string_input_producer creates a FIFOQueue under the hood, so to run the queue, we’ll need tf.Coordinator and tf.QueueRunner.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for _ in range(10):
#         key, value = reader.read(filename_queue)
#         key1, value1 = sess.run([key, value])
#         print(key1, value1)
#     coord.request_stop()
#     coord.join(threads)
#
with tf.Session() as sess:
    ## tf.train.string_input_producer creates a FIFOQueue under the hood, so to run the queue, we’ll need tf.Coordinator and tf.QueueRunner.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for _ in range(1):  # generate 1 example
        key, value = sess.run([key, value])  ## 计算得到的 key value 都是byte
        ## the value returned is just a string tensor,We need to use decoder to tansfer it to be our features.
        ## we could set its record_defaults in case some spaces are empty
        record_defaults = [[1.0] for _ in range(N_FEATURES)]
        record_defaults[4] = [""]  # make the fifth feature string
        record_defaults.append([1])
        content = tf.decode_csv(value, record_defaults=record_defaults)  # content is a list

        # we need convert content into tensor as features
        content[4] = tf.cond(tf.equal(content[4], tf.constant("Present")), lambda: tf.constant(1.0),
                             lambda: tf.constant(0.0))
        features = tf.stack(content[:N_FEATURES])
        label = content[-1]
        features_result = sess.run(features)
        print(features_result)
        print(features, label)
        # print(type(content), content)
        # print(type(key), type(value))
        # print(value)
        # print(key)
    coord.request_stop()
    coord.join(threads)

## TFReacord
## TensorFlow's binary file format a serialized tf.train.Example protobuf object

## Why binary?
## make better use of disk cache
## faster to move around
## can store data of different types(so you can put both images and labels in one place)

## Convert normal files to TFRecord
## Super easy

## Loss functions revisited
## Content loss
## To measure the content loss between the feature map in the content layer of the generated image and the content image
## Style loss
## To measure the style loss between the feature maps in the style layers of the generated image and the style image

## In practice you rarely use a queue by itself,but always with string_input_producer in more details in a little bit

## you can use tf.Coordinator to manage threads created by python

# def my_loop(coord):
#     x = 0
#     while not coord.should_stop():
#         print("still looping",x)
#         x = x + 1
#         if x == 1000:
#             coord.request_stop()
#
# coord = tf.train.Coordinator()
#
# threads = [threading.Thread(target=my_loop,args=(coord,)) for _ in range(10)]
#
# for t in threads:
#     t.start()
# coord.join(threads)

# minimum number elements in the queue after a dequeue,use to ensure
# that the samples are sufficiently mixed
# I think 10 times the BATCH_SIZE is sufficient
min_after_dequeue = 10 * BATCH_SIZE

# the maximum number of elements in the queue
capacity = 20 * BATCH_SIZE

# shuffle the data to generate BATCH_SIZE sample pairs
data_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE, capacity=capacity,
                                                 min_after_dequeue=min_after_dequeue)

## TFRecord
## help overcome irrational fear of binary files
## They make better use of cache.They are faster to move around.They can store data of different types.
## Like many machine learning frameworks, TensorFlow has its own binary data format which is called TFRecord.

def get_image_binary(filename):
    image = Image.open(filename)
    image = np.asarray(image,np.uint8)
    shape = np.asarray(image.shape,np.int32)
    return shape.tobytes(),image.tobytes()

def write_to_tfrecord(balel, shape,binary_image,tfrecord_file):
    '''
    This example is to write a sample to TFRecord file If you want to write more samples,just use a loop
    '''
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    example = tf.train.Example(features = tf.train.Feature(features={"label":tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                                                                     "shape":tf.train.Feature(bytes_list=tf.train.BytesList(value=[shape])),
                                                                     "image":tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_image]))
                                                                     }))
    writer.write(example.SerializeToString())
    writer.close()