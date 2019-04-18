import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MINST_data', one_hot=True)


with tf.Graph().as_default() as g:
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W) + b)

    y_actual = tf.placeholder(tf.float32, shape=[None, 10])
    cross_entropy = -tf.reduce_sum(y_actual*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys =  mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_actual:batch_ys})

        correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_actual, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print (sess.run(accuracy, feed_dict={x:mnist.test.images, y_actual:mnist.test.labels}))