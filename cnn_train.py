# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

"""
	deal with 1-D data of chi2/DM curve from 0 to 1500
	100-D data
"""

# restore data from the 100d-chi2.fzy

from sklearn.externals import joblib
X, Y = joblib.load('100d-chi2.fzy')

X = X.reshape([X.shape[0], 1, 100, 1])
Y = Y.reshape([Y.shape[0], 1])


from sklearn.model_selection import train_test_split	
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=0)


def generate_batch(X, Y, n_examples, batch_size):
	for batch_i in range(n_examples // batch_size):
		# [batch_i * batch_size, batch_i * batch_size + batch_size]
		start = batch_i*batch_size	
		end = start + batch_size
		batch_xs = X[start:end]
		batch_ys = Y[start:end]
		yield batch_xs, batch_ys # 生成每一个batch

# define data sizes
BATCH_SIZE = 32
#  input: [batch, in_height, in_width, in_channels]
INPUT_X_SHAPE = [BATCH_SIZE, 1, 100, 1]
INPUT_Y_SHAPE = [BATCH_SIZE, 1]

# filter: [filter_height, filter_width, in_channels, out_channels]
W1_SHAPE = [1, 3, 1, 3]
B1_SHAPE = [1, 1, 100, 3]

STRIDE1_SHAPE = [1, 1, 1, 1]

POOL1_KSIZE = [1, 1, 5, 1]
POOL1_STRIDE = [1, 1, 2, 1]

W2_SHAPE = [1, 3, 3, 1]
B2_SHAPE = [1, 1, 50, 1]

STRIDE2_SHAPE = [1, 1, 1, 1]
POOL2_KSIZE = [1, 1, 3, 1]
POOL2_STRIDE = [1, 1, 2, 1]

FC_W_SHAPE = [25, 10]
FC_B_SHAPE = [10]
OUT_W_SHAPE = [10, 2]
OUT_B_SHAPE = [2]

# 输入层
tf_X = tf.placeholder(tf.float32, INPUT_X_SHAPE)
tf_Y = tf.placeholder(tf.float32, INPUT_Y_SHAPE)


# 卷积层+激活层
filter_w1 = tf.Variable(tf.random_normal(W1_SHAPE))
filter_b1 = tf.Variable(tf.random_normal(B1_SHAPE))

# (None, BATCH, 100, 1) ---> (None, 1, 100, 3)
conv1 = tf.nn.conv2d(tf_X, filter_w1, strides=STRIDE1_SHAPE, padding='SAME')

feature_map1 = tf.nn.relu(conv1 + filter_b1)

# 池化层
# (None, 1, 100, 3)--->(None, 1, 50, 3)
pool1 = tf.nn.max_pool(feature_map1, ksize=POOL1_KSIZE, strides=POOL1_STRIDE, padding='SAME')


# 卷积层+激活层
# (None, 3, 3, 1)
filter_w2 = tf.Variable(tf.random_normal(W2_SHAPE))
filter_b2 =  tf.Variable(tf.random_normal(B2_SHAPE))

# (None, 1, 50, 3) --->(None,1 ,50, 1)
conv = tf.nn.conv2d(pool1, filter_w2, strides=STRIDE2_SHAPE, padding='SAME')

feature_map2 = tf.nn.relu(conv + filter_b2)

# 池化层
# (None,1 ,50, 1) ---> (None, 1, 25, 1)
pool2 = tf.nn.max_pool(feature_map2, ksize=POOL2_KSIZE, strides=POOL2_STRIDE, padding='SAME')


# 将特征图进行展开
# (1, 25)
max_pool2_flat = tf.reshape(pool2, [-1, 5*5])


# 全连接层
fc_w1 = tf.Variable(tf.random_normal(FC_W_SHAPE))
fc_b1 =  tf.Variable(tf.random_normal(FC_B_SHAPE))
fc_out1 = tf.nn.relu(tf.matmul(max_pool2_flat, fc_w1) + fc_b1)

# 输出层
out_w1 = tf.Variable(tf.random_normal(OUT_W_SHAPE))
out_b1 = tf.Variable(tf.random_normal(OUT_B_SHAPE))
pred = tf.nn.softmax(tf.matmul(fc_out1, out_w1) + out_b1)



#开始训练

loss = -tf.reduce_mean(tf_Y * tf.log(tf.clip_by_value(pred, 1e-11, 1.0)))

# Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
# 相比于基础SGD算法，1.不容易陷于局部优点。2.速度更快
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

y_pred = tf.argmax(pred, 1)
bool_pred = tf.equal(tf.argmax(tf_Y, 1), y_pred)
accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32))


"""
X = np.random.rand(BATCH_SIZE, 1, 100, 1)
Y = np.random.rand(BATCH_SIZE, 1)
"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        for batch_xs, batch_ys in generate_batch(X_train, y_train, y_train.shape[0], BATCH_SIZE):
            sess.run(train_step, feed_dict={tf_X:batch_xs, tf_Y:batch_ys})
            print(epoch, loss)
        if(epoch % 1 == 0):
            randidx = np.random.randint(X_test.shape[0], size=32)
            test_x = X_test[randidx, :]
            test_y = y_test[randidx, :]
            res = sess.run(accuracy, feed_dict={tf_X:test_x,tf_Y:test_y})
            loss_res = sess.run(loss, feed_dict={tf_X: batch_xs, tf_Y:batch_ys})
            #feat_map1 = sess.run(feature_map1, feed_dict={tf_X: batch_xs, tf_Y:batch_ys})
            print(epoch, 'test accuracy', res)
            print(epoch, 'training loss', loss_res)

# saving models
