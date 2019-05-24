from keras.layers import Input, Dense, Convolution2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten
from keras.layers import concatenate, BatchNormalization
from keras.models import Model
import numpy as np

def smallNet(im_shape = (224,224,1),weights=None):
    inputLayer = Input(shape=im_shape, dtype='float32', name='inputimg')
    #input = Input(shape=(224, 224, 3))

    conv1 = Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1')(inputLayer)
    pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same',name='pool1')(conv1)
    norm1 = BatchNormalization(axis=3, name='norm1')(pool1)
    reduction2 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='reduction2')(norm1)
    conv2 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='conv2')(reduction2)
    norm2 = BatchNormalization(axis=3, name='norm2')(conv2)
    pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool2')(norm2)
    

    icp1_reduction1 = Convolution2D(96,1,1,border_mode='same',activation='relu',name='icp1_reduction1')(pool2)
    icp1_out1 = Convolution2D(128,3,3,border_mode='same',activation='relu',name='icp1_out1')(icp1_reduction1)
    icp1_reduction2 = Convolution2D(16,1,1,border_mode='same',activation='relu',name='icp1_reduction2')(pool2)
    icp1_out2 = Convolution2D(32,5,5,border_mode='same',activation='relu',name='icp1_out2')(icp1_reduction2)
    icp1_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp1_pool')(pool2)
    icp1_out3 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp1_out3')(icp1_pool)
    icp1_out0 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp1_out0')(pool2)
    icp2_in = concatenate([icp1_out0, icp1_out1, icp1_out2, icp1_out3],axis=3,name='icp2_in')

    icp2_reduction1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp2_reduction1')(icp2_in)
    icp2_out1 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='icp2_out1')(icp2_reduction1)
    icp2_reduction2 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp2_reduction2')(icp2_in)
    icp2_out2 = Convolution2D(96,5,5,border_mode='same',activation='relu',name='icp2_out2')(icp2_reduction2)
    icp2_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp2_pool')(icp2_in)
    icp2_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp2_out3')(icp2_pool)
    icp2_out0 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp2_out0')(icp2_in)
    icp2_out = concatenate([icp2_out0, icp2_out1, icp2_out2, icp2_out3],axis=3,name='icp2_out')

    icp3_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same',name='icp3_in')(icp2_out)
    icp3_reduction1 = Convolution2D(96,1,1,border_mode='same',activation='relu',name='icp3_reduction1')(icp3_in)
    icp3_out1 = Convolution2D(208,3,3,border_mode='same',activation='relu',name='icp3_out1')(icp3_reduction1)
    icp3_reduction2 = Convolution2D(16,1,1,border_mode='same',activation='relu',name='icp3_reduction2')(icp3_in)
    icp3_out2 = Convolution2D(48,5,5,border_mode='same',activation='relu',name='icp3_out2')(icp3_reduction2)
    icp3_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp3_pool')(icp3_in)
    icp3_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp3_out3')(icp3_pool)
    icp3_out0 = Convolution2D(192,1,1,border_mode='same',activation='relu',name='icp3_out0')(icp3_in)
    icp3_out = concatenate([icp3_out0, icp3_out1, icp3_out2, icp3_out3],axis=3,name='icp3_out')

    cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),border_mode='valid',name='cls1_pool')(icp3_out)
    cls1_reduction_pose = Convolution2D(128,1,1,border_mode='same',activation='relu',name='cls1_reduction_pose')(cls1_pool)
    cls1_fc1_flat = Flatten()(cls1_reduction_pose)
    cls1_fc1_pose = Dense(1024,activation='relu',name='cls1_fc1_pose')(cls1_fc1_flat)
    cls1_fc_pose_xyz = Dense(3,name='cls1_fc_pose_xyz')(cls1_fc1_pose)
    cls1_fc_pose_wpqr = Dense(4,name='cls1_fc_pose_wpqr')(cls1_fc1_pose)


    net = Model(input=inputLayer, output=[cls1_fc_pose_xyz, cls1_fc_pose_wpqr])

    if weights:
        if weights[-3:] == ".h5":
            net.load_weights(weights, by_name=True)
        else:
            weights_data = np.load(weights, allow_pickle=True, fix_imports=True, encoding='latin1').item()
            #weights_data = np.load(weights).item()
            for layer in net.layers:
                if layer.name in weights_data.keys():
                    layer_weights = weights_data[layer.name]
                    layer.set_weights((layer_weights['weights'], layer_weights['biases']))
        print("FINISHED SETTING THE WEIGHTS!")

    return net
