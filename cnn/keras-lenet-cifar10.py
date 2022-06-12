"""
LeNet implementation in Keras, on Cifar10 datasets

Tensorflow version: 2.8.0
Keras version: 2.8.0
h5py version: 2.10.0
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras import layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
# 归一化
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# LeNet-5
model = models.Sequential()
model.add(layers.Conv2D(6, (5, 5), activation='sigmoid', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (5, 5), activation='sigmoid'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='sigmoid'))
model.add(layers.Dense(84, activation='sigmoid'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("test_acc:",test_acc)

# 将整个模型保存为 HDF5 文件。
# '.h5' 扩展名指示应将模型保存到 HDF5。
model.save('LeNet_classify_model.h5')

# 加载创建完全相同的模型，包括其权重和优化程序
loaded_model = tf.keras.models.load_model('LeNet_classify_model.h5')

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    img = (np.expand_dims(test_images[i],0))
    pred_arr = loaded_model.predict(img)
    predicted_label = np.argmax(pred_arr[0])
    true_label = test_labels[i][0]
    # 由于 CIFAR 的标签是 array，
    # 因此您需要额外的索引（index）。
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{}({})".format(class_names[predicted_label],class_names[true_label]),color=color)
plt.show()
