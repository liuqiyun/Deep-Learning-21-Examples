import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

if not os.path.exists('dataset'):
    os.mkdir('dataset')

mnist = input_data.read_data_sets('dataset/', one_hot=True)
trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels


def do_train(device_type):
    if device_type == 'gpu':
        device_type = '/gpu:0'
    else:
        device_type = '/cpu:0'

    with tf.device(device_type):  # <= This is optional
        n_input = 784
        n_output = 10
        weights = {
            'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
            'wd1': tf.Variable(tf.random_normal([14 * 14 * 64, n_output], stddev=0.1))
        }
        biases = {
            'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
            'bd1': tf.Variable(tf.random_normal([n_output], stddev=0.1))
        }
        def conv_simple(_input, _w, _b):
            # Reshape input
            _input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])
            # Convolution
            _conv1 = tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
            # Add-bias
            _conv2 = tf.nn.bias_add(_conv1, _b['bc1'])
            # Pass ReLu
            _conv3 = tf.nn.relu(_conv2)
            # Max-pooling
            _pool = tf.nn.max_pool(_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # Vectorize
            _dense = tf.reshape(_pool, [-1, _w['wd1'].get_shape().as_list()[0]])
            # Fully-connected layer
            _out = tf.add(tf.matmul(_dense, _w['wd1']), _b['bd1'])
            # Return everything
            out = {
                'input_r': _input_r, 'conv1': _conv1, 'conv2': _conv2, 'conv3': _conv3
                , 'pool': _pool, 'dense': _dense, 'out': _out
            }
            return out
    print("CNN ready with {}".format(device_type))

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_output])

    learning_rate = 0.001
    training_epochs = 10
    batch_size = 100
    display_step = 1

    with tf.device(device_type):
        _pred = conv_simple(x, weights, biases)['out']
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=_pred, logits=y))
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred, y))
        optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        _corr = tf.equal(tf.argmax(_pred, 1), tf.argmax(y, 1))  # Count corrects
        accr = tf.reduce_mean(tf.cast(_corr, tf.float32))  # Accuracy
        init = tf.initialize_all_variables()
    # Saver
    save_step = 1
    savedir = "nets/"
    saver = tf.train.Saver(max_to_keep=3)
    if not os.path.exists('nets'):
        os.mkdir('nets')
    print("Network Ready to Go!")

    do_train = 1
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)

    start_time = time.time()
    if do_train == 1:
       for epoch in range(training_epochs):
           avg_cost = 0.
           total_batch = int(mnist.train.num_examples / batch_size)
           # Loop over all batches
           for i in range(total_batch):
               batch_xs, batch_ys = mnist.train.next_batch(batch_size)
               # Fit training using batch data
               sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
               # Compute average loss
               avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

           # Display logs per epoch step
           if epoch % display_step == 0:
               print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
               train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys})
               print(" Training accuracy: %.3f" % (train_acc))
               test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel})
               print(" Test accuracy: %.3f" % (test_acc))

           # Save Net
           if epoch % save_step == 0:
               saver.save(sess, "nets/cnn_mnist_simple.ckpt-" + str(epoch))
       print("Optimization Finished.")
       print("Device : {}, Execution Time : {} sec".format(device_type, time.time() - start_time))

do_train('gpu')
# do_train('cpu')
