# Load pickled data
import pickle
import random
import cv2
import numpy as np

# TODO: Fill this in based on where you saved the training and testing data

# training_file = 'train.p'
training_file ='/home/oem/Documents/GitHub/CarND-Traffic-Sign-Classifier-Project/train.p'
testing_file = '/home/oem/Documents/GitHub/CarND-Traffic-Sign-Classifier-Project/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, Y_train = train['features'], train['labels']
X_test, Y_test = test['features'], test['labels']


### Replace each question mark with the appropriate value.

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:4]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(Y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# split data

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

X_train, Y_train = shuffle(X_train, Y_train)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=0.4, random_state=200)


# color conversion
x_train = []
x_test = []
x_val  = []
for image in X_train:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x_train.append(image_gray)

for image in X_val:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x_val.append(image_gray)

for image in X_test:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x_test.append(image_gray)

x_train     = np.array(x_train)
x_val       = np.array(x_val)
x_test      = np.array(x_test)

y_val       = Y_val
y_test      = Y_test
y_train     = Y_train

# print(x_train.shape)
# keep grayscale image size the same as color image
x_train      = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_val        = x_val.reshape(x_val.shape[0],x_val.shape[1],x_val.shape[2],1)
x_test       = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)


### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
n_train = x_train.shape[0]
ImgInd = random.sample(range(0, n_train), 6)

# example plots
i=1
plt.figure(figsize=(2,3))
for indx in ImgInd:
    image = x_train[indx].squeeze()
    plt.subplot(2,3,i)
    plt.imshow(image, cmap = 'gray')
    i+=1

import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Deep Neural Network structure
    # Con1 - relu - Pooling - Con2 - relu - Pooling -
    # Fc3 - relu - Fc4 - relue - Fc5 -
    # softmax  -cross entropy - cost
    # Hyperparameters
    mu = 0
    sigma = 0.1
    channel_color_x = x.get_shape().as_list()[3]

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    filter_size_width = 5
    filter_size_height = 5
    filter_depth = channel_color_x

    k_output = 6
    padding = 'VALID'
    strides = [1, 1, 1, 1]
    Fw1 = tf.Variable(tf.truncated_normal(shape = (filter_size_width,
          filter_size_height, filter_depth, k_output), mean = mu,
          stddev = sigma))
    Fb1 = tf.Variable(tf.zeros(k_output))
    hidden_layer1 = tf.nn.conv2d(x, Fw1, strides, padding) + Fb1

    # Activation.
    hidden_layer1 = tf.nn.relu(hidden_layer1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    hidden_layer1 = tf.nn.max_pool(hidden_layer1, ksize, strides, padding)
    keep_prob = 0.5
    hidden_layer1 = tf.nn.dropout(hidden_layer1, keep_prob)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    filter_size_width = 5
    filter_size_height = 5
    filter_depth = k_output
    k_output = 16
    padding = 'VALID'
    strides = [1, 1, 1, 1]
    Fw2 = tf.Variable(tf.truncated_normal( shape = (filter_size_width,
          filter_size_height, filter_depth, k_output), mean = mu, stddev = sigma))
    Fb2 = tf.Variable(tf.zeros(k_output))
    hidden_layer2 = tf.nn.conv2d (hidden_layer1, Fw2, strides, padding) + Fb2

    # Activation.
    hidden_layer2 = tf.nn.relu (hidden_layer2)


    # Pooling. Input = 10x10x16. Output = 5x5x16.
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = 'VALID'
    hidden_layer2 = tf.nn.max_pool(hidden_layer2, ksize, strides, padding)
    keep_prob = 0.5
    # hidden_layer2 = tf.nn.dropout(hidden_layer2, keep_prob)


    # TODO: Flatten. Input = 5x5x16. Output = 400.
    hidden_layer2 = flatten(hidden_layer2)
    keep_prob = 0.5
    # hidden_layer2 = tf.nn.dropout(hidden_layer2, keep_prob)

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    filter_size_len = hidden_layer2.get_shape().as_list()[1]
    k_output = 120
    Fw3 = tf.Variable(tf.truncated_normal(shape = (filter_size_len, k_output), mean = mu, stddev = sigma))
    Fb3 = tf.Variable(tf.zeros(k_output))
    hidden_layer3 = tf.matmul(hidden_layer2, Fw3) + Fb3
    # TODO: Activation.
    hidden_layer3 = tf.nn.relu(hidden_layer3)
    keep_prob = 0.5
    # hidden_layer3 = tf.nn.dropout(hidden_layer3, keep_prob)



    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    filter_size_len = k_output
    k_output  = 84
    Fw4 = tf.Variable(tf.truncated_normal(shape = (filter_size_len, k_output), mean = mu, stddev = sigma))
    Fb4 = tf.Variable(tf.zeros(k_output))
    hidden_layer4 = tf.matmul(hidden_layer3, Fw4) + Fb4


    # TODO: Activation.
    hidden_layer4 = tf.nn.relu(hidden_layer4)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    filter_size_len = k_output
    k_output  = n_classes
    Fw5 = tf.Variable(tf.truncated_normal(shape = (filter_size_len, k_output), mean = mu, stddev = sigma))
    Fb5 = tf.Variable(tf.zeros(k_output))
    hidden_layer5 = tf.matmul(hidden_layer4, Fw5) + Fb5

    logits = hidden_layer5

    return logits


EPOCHS = 500
BATCH_SIZE = 128


# define training batch tensor size
x = tf.placeholder(tf.float32, (None, x_train.shape[1], x_train.shape[2], x_train.shape[3]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(x_val, y_val)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, 'lenet')
    print("Model saved")

# test set
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(x_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# load new data
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random

AllPics = os.listdir("NewPic/")

i=1
x_data_new = []
for pic in AllPics:
    image = mpimg.imread('NewPic/'+pic)
    resized = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA)
    # plt.subplot(3,3,i)
    # plt.imshow(resized, cmap = 'gray')
    i+=1
    x_data_new.append(resized)

x_data_new = np.array(x_data_new)
y_data_new = np.array([8, 1, 28, 9, 5, 14, 25, 3])

nb_new = len(y_data_new)
plt.figure(figsize=(nb_new,2))
x_train, y_train = shuffle(x_train, y_train)

# compare the new labled image and the train image to make sure the new image
# is correclty labeled.

n=1;
for i in range(nb_new):
    Ni = random.randint(0, len(x_train)-500)
    for j in range(Ni, len(x_train)):
        if y_data_new[i] == y_train[j]:
            img1 = x_train[j].squeeze()
            # print(img1.shape)
            img2 = x_data_new [i]
            plt.subplot(nb_new ,2,n)
            plt.imshow(img1, cmap = 'gray')
            plt.subplot(nb_new,2,n+1)
            plt.imshow(img2)
            n = n+2
            break

xnew = []
for image in x_data_new:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    xnew.append(image_gray)

xnew = np.array(xnew)
x_data_newG = xnew.reshape(xnew.shape[0], xnew.shape[1], xnew.shape[2],1)

BATCH_SIZE = len(x_data_newG)
ktop = 3
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(x_data_newG, y_data_new)
    prediction = sess.run(correct_prediction, feed_dict={x: x_data_newG, y: y_data_new})
    logitsV = sess.run(tf.nn.softmax(logits), feed_dict={x: x_data_newG, y: y_data_new})
    OneHotY = sess.run(one_hot_y, feed_dict={y: y_data_new})
    TopKv = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k=ktop), feed_dict={x: x_data_newG, y: y_data_new})
    print("Accuracy of New Image = {:.3f}".format(test_accuracy))


IsInTop =np.zeros([nb_new,1])

plt.figure(figsize=(4,2))

for i in range(x_data_newG.shape[0]):
    performance =logitsV[i,:]
    TrueValue = OneHotY[i,:]
    plt.subplot(4,2,i+1)
    axes = plt.bar(range(0,n_classes), performance, 0.2, align='center', alpha=0.5, color = 'blue')
    axes = plt.bar(np.array(range(0,n_classes)), TrueValue, 0.3, align='center', alpha=0.5, color = 'red')
    if y_data_new[i] in TopKv.indices[i,:]:
        IsInTop[i]=1
print("Does the actual sign fall into the top: {0} prediction, 1 - yes, 0 -no \n".format(ktop), IsInTop)
plt.show()
