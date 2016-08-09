from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import tflearn

from data import load_train_data, load_test_data
from data import img_rows, img_cols

img_rows = 64
img_cols = 80
n_features = img_cols * img_rows
smooth = 1.0

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, shape=(-1,n_features))
    y_pred_f = tf.reshape(y_pred, shape=(-1,n_features))
    y_true_s = tf.reduce_sum(y_true_f, reduction_indices=(1))
    y_pred_s = tf.reduce_sum(y_pred_f, reduction_indices=(1))
    intersection = tf.reduce_sum(tf.mul(y_true_f,y_pred_f), reduction_indices=(1))
    return tf.reduce_mean(tf.div(tf.add(tf.mul(2.0,intersection),smooth),tf.add(tf.add(y_true_s,y_pred_s),smooth)), reduction_indices=(0))

def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)

def get_net(): 
    input_data = tflearn.input_data(shape=[None, img_rows, img_cols, 1]) #64 by 80
    conv1 = tflearn.conv_2d(input_data, nb_filter = 32, filter_size = 3, activation='relu')
    conv1 = tflearn.conv_2d(conv1, nb_filter = 32, filter_size = 3, activation='relu')
    pool1 = tflearn.max_pool_2d(conv1, kernel_size = [2,2]) #32 by 40

    conv2 = tflearn.conv_2d(pool1, nb_filter = 64, filter_size = 3, activation='relu')
    conv2 = tflearn.conv_2d(conv2, nb_filter = 64, filter_size = 3, activation='relu')
    pool2 = tflearn.max_pool_2d(conv2, kernel_size = [2,2]) #16 by 20

    conv3 = tflearn.conv_2d(pool2, nb_filter = 128, filter_size = 3, activation='relu')
    conv3 = tflearn.conv_2d(conv3, nb_filter = 128, filter_size = 3, activation='relu')
    pool3 = tflearn.max_pool_2d(conv3, kernel_size = [2,2]) #8 by 10

    conv4 = tflearn.conv_2d(pool3, nb_filter = 256, filter_size = 3, activation='relu')
    conv4 = tflearn.conv_2d(conv4, nb_filter = 256, filter_size = 3, activation='relu')
    pool4 = tflearn.max_pool_2d(conv4, kernel_size = [2,2]) #4 by 5

    conv5 = tflearn.conv_2d(pool4, nb_filter = 512, filter_size = 3, activation='relu')
    conv5 = tflearn.conv_2d(conv5, nb_filter = 512, filter_size = 3, activation='relu')

    up6   = tflearn.merge([tflearn.upsample_2d(conv5, kernel_size = [2,2]),conv4], mode='concat', axis=3) #8 by 10
    conv6 = tflearn.conv_2d(up6, nb_filter = 256, filter_size = 3, activation='relu')
    conv6 = tflearn.conv_2d(conv6, nb_filter = 256, filter_size = 3, activation='relu')

    up7   = tflearn.merge([tflearn.upsample_2d(conv6, kernel_size = [2,2]),conv3], mode='concat', axis=3) #16 by 20
    conv7 = tflearn.conv_2d(up7, nb_filter = 128, filter_size = 3, activation='relu')
    conv7 = tflearn.conv_2d(conv7, nb_filter = 128, filter_size = 3, activation='relu')

    up8   = tflearn.merge([tflearn.upsample_2d(conv7, kernel_size = [2,2]),conv2], mode='concat', axis=3) #32 by 40
    conv8 = tflearn.conv_2d(up8, nb_filter = 64, filter_size = 3, activation='relu')
    conv8 = tflearn.conv_2d(conv8, nb_filter = 64, filter_size = 3, activation='relu')

    up9   = tflearn.merge([tflearn.upsample_2d(conv8, kernel_size = [2,2]),conv1], mode='concat', axis=3) #64 by 80
    conv9 = tflearn.conv_2d(up9, nb_filter = 32, filter_size = 3, activation='relu')
    conv9 = tflearn.conv_2d(conv9, nb_filter = 32, filter_size = 3, activation='relu')

    conv10 = tflearn.conv_2d(conv9, nb_filter = 1, filter_size = 1, activation='sigmoid')

    adam = tflearn.Adam(learning_rate=1e5)
    net = tflearn.regression(conv10, optimizer='adam', loss=dice_coef_loss)
    model = tflearn.DNN(net, tensorboard_verbose=1,tensorboard_dir='/tmp/tflearn_logs/')

    return model


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model = get_net()
    model.fit(imgs_train, imgs_mask_train, n_epoch=20, batch_size=64) #,validation_set=0.15)
    model.save('tfmodel_v1.tflearn')

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
