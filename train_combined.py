from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import tflearn

from data import load_train_data, load_test_data, calc_dups
from data import img_rows, img_cols

def load_data():
    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    #target is existance of mask
    has_mask_train = np.zeros((imgs_mask_train.shape[0],2), dtype='int32')
    has_mask_sum = np.sum(imgs_mask_train, axis=(1,2,3))
    for i in range(imgs_mask_train.shape[0]):
        if (has_mask_sum[i] < 10.0):
            has_mask_train[i,0] = 1
        else:
            has_mask_train[i,1] = 1
    print('We have ',imgs_train.shape[0],' training samples')
    print('-'*30)
    print('Loading test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = imgs_test.astype('float32')
    print('We have ',imgs_test.shape[0],' test samples')
    return imgs_train, imgs_mask_train, imgs_test, imgs_id_test, has_mask_train

def remove_dups(imgs_train, imgs_mask_train, has_mask_train):
    #remove dups
    print('-'*30)
    print('Removing Duplicates...')
    print('-'*30)
    dups_ind = list(calc_dups(imgs_train))
    keep_ind = [x for x in range(imgs_train.shape[0]) if x not in dups_ind]
    imgs_train = imgs_train[keep_ind,]
    imgs_mask_train = imgs_mask_train[keep_ind,]
    has_mask_train = has_mask_train[keep_ind,]
    print('Removed ',len(dups_ind),' dups.')
    return imgs_train, imgs_mask_train, has_mask_train, dups_ind

def scale_data(imgs_train, imgs_test):
    #calc mean & std
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean
    imgs_train /= std
    imgs_test -= mean
    imgs_test /= std
    return imgs_train, imgs_test #check probaly dont need to return

def dice_coef(y_true, y_pred):
    smooth = 1.0
    #ave_coverage = 0.1
    shape = y_pred.get_shape()
    n_features = shape[1].value*shape[2].value
    #flatten da tensors a smidge
    y_true_f = tf.reshape(y_true, shape=(-1,n_features))
    y_pred_f = tf.reshape(y_pred, shape=(-1,n_features))
    #Calc dice
    y_true_s = tf.reduce_sum(y_true_f, reduction_indices=(1))
    y_pred_s = tf.reduce_sum(y_pred_f, reduction_indices=(1))
    intersection = tf.reduce_sum(tf.mul(y_true_f,y_pred_f), reduction_indices=(1))
    dice_coef = tf.reduce_mean(tf.div(tf.add(tf.mul(2.0,intersection),smooth),tf.add(tf.add(y_true_s,y_pred_s),smooth)), reduction_indices=(0))
    #dice_coef = tf.div(tf.add(tf.mul(2.0,intersection),smooth),tf.add(tf.add(y_true_s,y_pred_s),smooth))
    #Calc ratio
    #ytrue_cov = tf.add(tf.reduce_sum(tf.cast(y_true_f > 0.5,dtype='float32'), reduction_indices=1),smooth)
    #ypred_cov = tf.add(tf.reduce_sum(tf.cast(y_pred_f > 0.5,dtype='float32'), reduction_indices=1),smooth)
    #ratio_loss = tf.reduce_mean(tf.abs(tf.add(tf.div(ytrue_cov,ypred_cov),-1.0)))
    #return
    #print('Dice: ',tf.Print(dice_coef, [dice_coef],first_n=3),' RatioLoss: ',tf.Print(ratio_loss,[ratio_loss], first_n=3))
    return -dice_coef#-ratio_loss)

def create_unet():
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
    adam = tflearn.Adam(learning_rate=1e-6, epsilon=1e-7) #1=5-e7 #2=1-e7
    net = tflearn.regression(conv10, optimizer=adam,loss=dice_coef)
    model = tflearn.DNN(net, tensorboard_verbose=1,tensorboard_dir='/tmp/tflearn_logs/')
    return model

def create_yn_net():
    net = tflearn.input_data(shape=[None, img_rows, img_cols, 1]) #D = 256, 256
    net = tflearn.conv_2d(net,nb_filter=8,filter_size=3, activation='relu', name='conv0.1')
    net = tflearn.conv_2d(net,nb_filter=8,filter_size=3, activation='relu', name='conv0.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool0') #D = 128, 128
    net = tflearn.dropout(net,0.70,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=16,filter_size=3, activation='relu', name='conv1.1')
    net = tflearn.conv_2d(net,nb_filter=16,filter_size=3, activation='relu', name='conv1.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool1') #D = 64,  64
    net = tflearn.dropout(net,0.65,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv2.1')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv2.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool2') #D = 32 by 32
    net = tflearn.dropout(net,0.60,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv3.1')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv3.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool3') #D = 16 by 16
    net = tflearn.dropout(net,0.55,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=64,filter_size=3, activation='relu', name='conv4.1')
    net = tflearn.conv_2d(net,nb_filter=64,filter_size=3, activation='relu', name='conv4.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool4') #D = 8 by 8
    net = tflearn.dropout(net,0.50,name='dropout0')
    net = tflearn.fully_connected(net, n_units = 128, activation='relu', name='fc1')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.00001)#, loss='mean_square')
    model = tflearn.DNN(net, tensorboard_verbose=1,tensorboard_dir='/tmp/tflearn_logs/')
    return model

def create_net():
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
    up6   = tflearn.upsample_2d(conv5, kernel_size = [2,2]) #8 by 10
    conv6 = tflearn.conv_2d(up6, nb_filter = 256, filter_size = 3, activation='relu')
    conv6 = tflearn.conv_2d(conv6, nb_filter = 256, filter_size = 3, activation='relu')
    up7   = tflearn.upsample_2d(conv6, kernel_size = [2,2]) #16 by 20
    conv7 = tflearn.conv_2d(up7, nb_filter = 128, filter_size = 3, activation='relu')
    conv7 = tflearn.conv_2d(conv7, nb_filter = 128, filter_size = 3, activation='relu')
    up8   = tflearn.upsample_2d(conv7, kernel_size = [2,2]) #32 by 40
    conv8 = tflearn.conv_2d(up8, nb_filter = 64, filter_size = 3, activation='relu')
    conv8 = tflearn.conv_2d(conv8, nb_filter = 64, filter_size = 3, activation='relu')
    up9   = tflearn.upsample_2d(conv8, kernel_size = [2,2]) #64 by 80
    conv9 = tflearn.conv_2d(up9, nb_filter = 32, filter_size = 3, activation='relu')
    conv9 = tflearn.conv_2d(conv9, nb_filter = 32, filter_size = 3, activation='relu')
    conv10 = tflearn.conv_2d(conv9, nb_filter = 1, filter_size = 1, activation='sigmoid')
    adam = tflearn.Adam(learning_rate=1e-8, epsilon=1e-7) #1=5-e7 #2=1-e7
    net = tflearn.regression(conv10, optimizer=adam,loss=dice_coef)
    #net = tflearn.regression(conv10, learning_rate=1e-4,loss='dice_coef')
    model = tflearn.DNN(net, tensorboard_verbose=1,tensorboard_dir='/tmp/tflearn_logs/')
    return model

def train(imgs_train, has_mask_train):
    print('-'*30)
    print('Its Train time...')
    print('-'*30)
    #Fit or load
    model = create_yn_net()
    #model.load('tfmodel_yn_v1.tflearn')
    model.fit(imgs_train, has_mask_train, n_epoch=100, batch_size=200, show_metric=True, validation_set=0.15) #,validation_set=0.15)
    model.save('tfmodel_yn_v1.tflearn')
    return model

def predict(model, imgs_test):
    print('-'*30)
    print('Its Predict time...')
    print('-'*30)
    imgs_mask_test = np.empty([imgs_test.shape[0],img_rows, img_cols],dtype='float32')
    for i in range(imgs_test.shape[0]):
        if i % 100 == 0:
            print('Done:',i)
        imgs_mask_test[i]  = np.asarray(model.predict(imgs_test[i].reshape(1,img_rows, img_cols,1))[0]).reshape(img_rows, img_cols)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    return imgs_mask_test

def save_processed_data(imgs_train, imgs_mask_train, imgs_test, imgs_id_test, has_mask_train):
    np.save('processed_imgs_train.npy',imgs_train)
    np.save('processed_imgs_mask_train.npy',imgs_mask_train)
    np.save('processed_imgs_test.npy',imgs_test)
    np.save('processed_imgs_id_test.npy',imgs_id_test)
    np.save('processed_has_mask_train.npy',has_mask_train)

def load_processed_data():
    imgs_train = np.load('processed_imgs_train.npy')
    imgs_mask_train = np.load('processed_imgs_mask_train.npy')
    imgs_test = np.load('processed_imgs_test.npy')
    imgs_id_test = np.load('processed_imgs_id_test.npy')
    has_mask_train = np.load('processed_has_mask_train.npy')
    return imgs_train, imgs_mask_train, imgs_test, imgs_id_test, has_mask_train

def train_and_predict():
    #load the data
    #imgs_train, imgs_mask_train, imgs_test, imgs_id_test, has_mask_train = load_data()
    #remove dups
    #imgs_train, imgs_mask_train, has_mask_train, dups_ind = remove_dups(imgs_train, imgs_mask_train, has_mask_train)
    #scale data
    #imgs_train, imgs_test = scale_data(imgs_train, imgs_test)
    #save processed data
    #save_processed_data(imgs_train, imgs_mask_train, imgs_test, imgs_id_test, has_mask_train)
    ##load processed_data
    imgs_train, imgs_mask_train, imgs_test, imgs_id_test, has_mask_train = load_processed_data()
    #train
    model = train(imgs_train, has_mask_train)
    #predict
    #imgs_mask_test = predict(model, imgs_test)

if __name__ == '__main__':
    train_and_predict()
