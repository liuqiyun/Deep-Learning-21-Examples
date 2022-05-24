# coding: utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from tensorflow import saved_model

# force using CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# gpus = tf.config.experimental.list_physical_devices('GPU')##获取可用GPU
# for gpu in (gpus):
#  tf.config.experimental.set_memory_growth(gpu, True)##设置显存使用方式

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0 ##数据预处理归一化

x_train = x_train[..., tf.newaxis] ##增加一个通道维数
x_test = x_test[..., tf.newaxis]

train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1).batch(1000)##切分数据集为BatchDataset，混淆数据集
test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)
# train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)##切分数据集为BatchDataset，混淆数据集
# test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(Model):##cnn网络模型定义
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    # self.d1 = Dense(128, activation='relu')
    # self.d2 = Dense(10, activation='softmax')
    self.d1 = Dense(1, activation='relu')
    self.d2 = Dense(1, activation='softmax')
  @tf.function
  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# mynetwork = tf.keras.models.Sequential([    ##一般模型
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])

mynetwork = MyModel()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()##损失函数定义

optimizer = tf.keras.optimizers.Adam()##优化器定义

train_loss = tf.keras.metrics.Mean(name='train_loss')##损失值
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')##准确率

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function ##训练
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = mynetwork(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, mynetwork.trainable_variables)
    optimizer.apply_gradients(zip(gradients, mynetwork.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function ##测试
def test_step(images, labels):
    predictions = mynetwork(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


for epoch in range(1):
# for epoch in range(5):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_set:
        train_step(images, labels)

    for test_images, test_labels in test_set:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))

print("train_accuracy.result(): %g" % train_accuracy.result())
tf.saved_model.save(mynetwork, 'saved_model')##保存模型，表明文件夹即可

