from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf
import tflearn

from data import load_train_data, load_test_data, calc_dups
from data import img_rows, img_cols

def load_data():
    imgs_train, imgs_mask_train = load_train_data()
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    print('We have ',imgs_train.shape[0],' training samples')
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = imgs_test.astype('float32')
    print('We have ',imgs_test.shape[0],' test samples')
    return imgs_train, imgs_mask_train, imgs_test, imgs_id_test

def create_has_mask(imgs_mask_train):
    #target is existance of mask
    has_mask_train = np.zeros((imgs_mask_train.shape[0],2), dtype='int32')
    has_mask_sum = np.sum(imgs_mask_train, axis=(1,2,3))
    for i in range(imgs_mask_train.shape[0]):
        if (has_mask_sum[i] < 10.0):
            has_mask_train[i,0] = 1
        else:
            has_mask_train[i,1] = 1
    print(np.sum(has_mask_train[:,1]),' have a mask')
    return has_mask_train
    #has_mask_train = create_has_mask(imgs_mask_train)

def create_location(imgs_mask_train):
    imgs_mask_location = np.zeros((imgs_mask_train.shape[0],2), dtype='float32')
    for i in range(imgs_mask_train.shape[0]):
        a = imgs_mask_train[i]
        if (np.sum(a) != 0.0):
            rmin=np.min(np.where(np.sum(a,axis=(0,2))>0.1))
            rmax=np.max(np.where(np.sum(a,axis=(0,2))>0.1))
            rmid = (rmin+rmax)/2.0
            cmin=np.min(np.where(np.sum(a,axis=(1,2))>1.0))
            cmax=np.max(np.where(np.sum(a,axis=(1,2))>1.0))
            cmid = (cmin+cmax)/2.0
            imgs_mask_location[i] = [rmid / img_rows, cmid/img_cols]
    return imgs_mask_location
    #imgs_mask_location = create_location(imgs_mask_train)

def create_span(imgs_mask_train):
    imgs_mask_size = np.zeros((imgs_mask_train.shape[0],1), dtype='float32')
    for i in range(imgs_mask_train.shape[0]):
        a = imgs_mask_train[i]
        if (np.sum(a) != 0.0):
            rmin=np.min(np.where(np.sum(a,axis=(0,2))>0.1))
            rmax=np.max(np.where(np.sum(a,axis=(0,2))>0.1))
            rmid = (rmin+rmax)/2.0
            rspan = rmax-rmid
            cmin=np.min(np.where(np.sum(a,axis=(1,2))>1.0))
            cmax=np.max(np.where(np.sum(a,axis=(1,2))>1.0))
            cmid = (cmin+cmax)/2.0
            cspan = cmax-cmid
            imgs_mask_size[i] = (cspan+rspan)*(cspan+rspan)/(4.0*img_rows*img_cols)
    return imgs_mask_size
    #imgs_mask_size = create_span(imgs_mask_train)

def scale_data(imgs_train, imgs_test):
    #calc mean & std
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean
    imgs_train /= std
    imgs_test -= mean
    imgs_test /= std
    return imgs_train, imgs_test

def remove_dups(nparry, imgs_mask_train, has_mask_train):
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

def yn_net():
    net = tflearn.input_data(shape=[None, img_rows, img_cols, 1]) #D = 256, 256
    net = tflearn.conv_2d(net,nb_filter=8,filter_size=3, activation='relu', name='conv0.1')
    net = tflearn.conv_2d(net,nb_filter=8,filter_size=3, activation='relu', name='conv0.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool0') #D = 128, 128
    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=16,filter_size=3, activation='relu', name='conv1.1')
    net = tflearn.conv_2d(net,nb_filter=16,filter_size=3, activation='relu', name='conv1.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool1') #D = 64,  64
    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv2.1')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv2.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool2') #D = 32 by 32
    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv3.1')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv3.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool3') #D = 16 by 16
    net = tflearn.dropout(net,0.75,name='dropout0')
#    net = tflearn.conv_2d(net,nb_filter=64,filter_size=3, activation='relu', name='conv4.1')
#    net = tflearn.conv_2d(net,nb_filter=64,filter_size=3, activation='relu', name='conv4.2')
#    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool4') #D = 8 by 8
#    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.fully_connected(net, n_units = 128, activation='relu', name='fc1')
    net = tflearn.fully_connected(net, 2, activation='sigmoid')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001)
    model = tflearn.DNN(net, tensorboard_verbose=1,tensorboard_dir='/tmp/tflearn_logs/')
    return model

def loc_net():
    net = tflearn.input_data(shape=[None, img_rows, img_cols, 1]) #D = 256, 256
    net = tflearn.conv_2d(net,nb_filter=8,filter_size=3, activation='relu', name='conv0.1')
    net = tflearn.conv_2d(net,nb_filter=8,filter_size=3, activation='relu', name='conv0.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool0') #D = 128, 128
    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=16,filter_size=3, activation='relu', name='conv1.1')
    net = tflearn.conv_2d(net,nb_filter=16,filter_size=3, activation='relu', name='conv1.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool1') #D = 64,  64
    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv2.1')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv2.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool2') #D = 32 by 32
    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv3.1')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv3.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool3') #D = 16 by 16
    net = tflearn.dropout(net,0.75,name='dropout0')
#    net = tflearn.conv_2d(net,nb_filter=64,filter_size=3, activation='relu', name='conv4.1')
#    net = tflearn.conv_2d(net,nb_filter=64,filter_size=3, activation='relu', name='conv4.2')
#    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool4') #D = 8 by 8
#    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.fully_connected(net, n_units = 128, activation='relu', name='fc1')
    net = tflearn.fully_connected(net, n_units = 64, activation='relu', name='fc2')
    net = tflearn.fully_connected(net, 2, activation='linear')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='mean_square')
    model = tflearn.DNN(net, tensorboard_verbose=1,tensorboard_dir='/tmp/tflearn_logs/')
    return model

def span_net():
    net = tflearn.input_data(shape=[None, img_rows, img_cols, 1]) #D = 256, 256
    net = tflearn.conv_2d(net,nb_filter=8,filter_size=3, activation='relu', name='conv0.1')
    net = tflearn.conv_2d(net,nb_filter=8,filter_size=3, activation='relu', name='conv0.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool0') #D = 128, 128
    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=16,filter_size=3, activation='relu', name='conv1.1')
    net = tflearn.conv_2d(net,nb_filter=16,filter_size=3, activation='relu', name='conv1.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool1') #D = 64,  64
    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv2.1')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv2.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool2') #D = 32 by 32
    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv3.1')
    net = tflearn.conv_2d(net,nb_filter=32,filter_size=3, activation='relu', name='conv3.2')
    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool3') #D = 16 by 16
    net = tflearn.dropout(net,0.75,name='dropout0')
#    net = tflearn.conv_2d(net,nb_filter=64,filter_size=3, activation='relu', name='conv4.1')
#    net = tflearn.conv_2d(net,nb_filter=64,filter_size=3, activation='relu', name='conv4.2')
#    net = tflearn.max_pool_2d(net, kernel_size = [2,2], name='maxpool4') #D = 8 by 8
#    net = tflearn.dropout(net,0.75,name='dropout0')
    net = tflearn.fully_connected(net, n_units = 128, activation='relu', name='fc1')
    net = tflearn.fully_connected(net, n_units = 64, activation='relu', name='fc2')
    net = tflearn.fully_connected(net, 1, activation='linear')
    #net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='mean_square')
    net = tflearn.regression(net, learning_rate=0.00001,loss='binary_crossentropy')
    model = tflearn.DNN(net, tensorboard_verbose=1,tensorboard_dir='/tmp/tflearn_logs/')
    return model

def train_span(imgs_train, imgs_mask_size):
    print('-'*30)
    print('Training for Location...')
    print('-'*30)
    #Clean up Graphs
    tf.reset_default_graph()
    model = span_net()
    model.fit(imgs_train, imgs_mask_size, n_epoch=50, batch_size=512, show_metric=True, validation_set=0.15)
    model.save('model_size_v1.tflearn')
    return model


model_span = train_span(imgs_train, imgs_mask_size)
imgs_mask_size[:20]
model.span.predict(imgs_train[:20])

def predict_span(model, imgs_test):
    print('-'*30)
    print('Its Predict time...')
    print('-'*30)
    imgs_mask_size_test = np.empty([imgs_test.shape[0]],dtype='float32')
    for i in range(imgs_test.shape[0]):
        if i % 100 == 0:
            print('Done:',i)
        imgs_mask_size_test[i]  = np.asarray(model.predict(imgs_test[i].reshape(1,img_rows, img_cols,1))[0]).reshape(img_rows, img_cols)
    np.save('imgs_mask_test.npy', imgs_mask_test)
    return imgs_mask_size_test

def train_loc(imgs_train, imgs_mask_location):
    print('-'*30)
    print('Training for Location...')
    print('-'*30)
    #Clean up Graphs
    tf.reset_default_graph()
    model = loc_net()
    model.fit(imgs_train, imgs_mask_location, n_epoch=50, batch_size=400, show_metric=True, validation_set=0.15)
    model.save('model_loc_v1.tflearn')
    return model
    #model_loc = train_loc(imgs_train, imgs_mask_location)

def train_yn(imgs_train, has_mask_train):
    print('-'*30)
    print('Training for Y/N...')
    print('-'*30)
    #Clean up Graphs
    tf.reset_default_graph()
    model = yn_net()
    model.fit(imgs_train, has_mask_train, n_epoch=50, batch_size=400, show_metric=True, validation_set=0.15)
    model.save('model_yn_v1.tflearn')
    return model
    #model_yn = train_yn(imgs_train, has_mask_train)



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

def save_processed_data(imgs_train, imgs_mask_train, imgs_test, imgs_id_test, has_mask_train, imgs_mask_location, imgs_mask_size):
    np.save('processed_imgs_train.npy',imgs_train)
    np.save('processed_imgs_mask_train.npy',imgs_mask_train)
    np.save('processed_imgs_test.npy',imgs_test)
    np.save('processed_imgs_id_test.npy',imgs_id_test)
    np.save('processed_has_mask_train.npy',has_mask_train)
    np.save('processed_imgs_mask_location.npy',imgs_mask_location)
    np.save('processed_imgs_mask_size.npy',imgs_mask_size)

def load_processed_data():
    imgs_train = np.load('processed_imgs_train.npy')
    imgs_mask_train = np.load('processed_imgs_mask_train.npy')
    imgs_test = np.load('processed_imgs_test.npy')
    imgs_id_test = np.load('processed_imgs_id_test.npy')
    has_mask_train = np.load('processed_has_mask_train.npy')
    imgs_mask_location = np.load('processed_imgs_mask_location.npy')
    imgs_mask_size = np.load('processed_imgs_mask_size.npy')
    return imgs_train, imgs_mask_train, imgs_test, imgs_id_test, has_mask_train, imgs_mask_location, imgs_mask_size

def train_and_predict():
    #load the data
    imgs_train, imgs_mask_train, imgs_test, imgs_id_test = load_data()
    #remove dups
    #imgs_train, imgs_mask_train, has_mask_train, dups_ind = remove_dups(imgs_train, imgs_mask_train, has_mask_train)
    #scale data
    imgs_train, imgs_test = scale_data(imgs_train, imgs_test)
    #create y/n data
    has_mask_train = create_has_mask(imgs_mask_train)
    imgs_mask_location = create_location(imgs_mask_train)
    imgs_mask_size = create_span(imgs_mask_train)
    #save processed data
    save_processed_data(imgs_train, imgs_mask_train, imgs_test, imgs_id_test, has_mask_train, imgs_mask_location, imgs_mask_size)
    ##load processed_data
    imgs_train, imgs_mask_train, imgs_test, imgs_id_test, has_mask_train, imgs_mask_location, imgs_mask_size = load_processed_data()
    #train
    model = train(imgs_train, has_mask_train)
    #predict
    #imgs_mask_test = predict(model, imgs_test)

if __name__ == '__main__':
    train_and_predict()
