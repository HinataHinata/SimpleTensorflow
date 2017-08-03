""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt # 帮助画图
import tensorflow as tf
import xlrd # xlrd 帮助操作Excel

import utils

DATA_FILE = '../data/fire_theft.xls'

# 函数要定义成 operation 的方式 -- 看起来似乎是 operation 可以使用 + - * ／ 操作符
def huber_loss(labels,predictions,delta=1.0):
	residual = tf.abs(predictions - labels)
	conditon = tf.less(residual,delta)
	small_res = 0.5 * tf.square(residual)
	large_res = delta * residual - 0.5 * tf.square(delta)
	# return tf.select(conditon,small_res,large_res) # select 被 where 替换
	return tf.where(conditon,small_res,large_res)


# Phase 1: Assemble the graph
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
print('data\r\n',data);
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
# Both have the type float32
X = tf.placeholder(dtype=tf.float32,name='X')
Y = tf.placeholder(dtype=tf.float32,name='Y')

# Step 3: create weight and bias, initialized to 0
# name your variables w and b
w = tf.Variable(0.0,trainable=True,name='weight')
b = tf.Variable(0.0,trainable=True,name='bias')

# Step 4: predict Y (number of theft) from the number of fire
# name your variable Y_predicted
Y_predicted = X * w + b
# Y_predicted = tf.add(tf.multiply(w,X),b,name='Y_predicted')

# Step 5: use the square error as the loss function
# name your variable loss
# loss = tf.square(Y - Y_predicted,name='loss')
loss = huber_loss(Y,Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
trainOp = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss=loss)

# Phase 2: Train our model
with tf.Session() as sess:

	# Step 7: initialize the necessary variables, in this case, w and b
	# TO - DO
	sess.run(tf.global_variables_initializer())
	tf.summary.FileWriter('./graphs/theft', graph=sess.graph)

	# Step 8: train the model
	for i in range(50): # run 100 epochs
		total_loss = 0
		for x, y in data:
			# Session runs optimizer to minimize loss and fetch the value of loss. Name the received value as l
			# TO DO: write sess.run()
			_,l = sess.run([trainOp,loss],feed_dict={X:x,Y:y})
			total_loss += l
		print("Epoch {0}: {1}".format(i, total_loss/n_samples))
		w1 = w.eval()
		b1 = b.eval()


# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w1 + b1, 'r', label='Predicted data')
plt.legend()
plt.show()