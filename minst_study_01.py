
#导入MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./MINST_data',one_hot=True)

#创建session链接图类
import tensorflow as tf
sess = tf.InteractiveSession()

#创建占位符
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#变量
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#变量初始化
sess.run(tf.initialize_all_variables())

#回归模型
y = tf.nn.softmax(tf.matmul(x, W) + b)

#交叉熵表示损失函数
cross_entrop = -tf.reduce_sum(y_*tf.log(y))

#训练模型
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entrop)
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0], y_:batch[1]})

#评估模型
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    accuracy_acc = accuracy.eval(feed_dict = {x:mnist.test.images, y_:mnist.test.labels})
    #if i % 50 == 0:
    #print("After %d train ,accuration = %g" % (i, accuracy_acc))


