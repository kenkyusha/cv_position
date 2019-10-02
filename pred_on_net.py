#!/usr/bin/env python3
import tensorflow as tf
import tqdm
from scripts.nets import smallNet
from scripts.utils import *
from scripts.metrics import *
from scripts.loss import *
import keras
from keras import backend as K
from keras.models import load_model
import random
import argparse
import pdb

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument('--net', help="name of the model to load", required=False)
parser.add_argument('--wts', help="path to the trained neural network to test ", required=True)
#parser.add_argument('--fname', help="name of test", required=True)
parser.add_argument('--data', help="dataset for testing", required=True)

args = parser.parse_args()

if args.net == None:
    model = smallNet(args.wts)
    m_name = 'smallNet'
else:
    m_name = os.path.basename(args.net).split('.')[0]
    model = load_model(args.net, custom_objects={'euc_loss1x': euc_loss1x, 'euc_loss1q': euc_loss1q})
    model.load_weights(args.wts)

try:
    model = multi_gpu_model(model, gpus=None)
except:
    pass

opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt, loss={'cls1_fc_pose_xyz': euc_loss1x, 'cls1_fc_pose_wpqr': euc_loss1q})

fname = os.path.basename(args.data).split('.')[0]
model.summary()

imgs, true_label = gen_test_data(args.data, flag = False)
list_pred_pos = []
list_pred_wpgr = []
# TODO: replace for-looping with batch prediction
for i in tqdm(range(len(imgs))):
    testImg = imgs[i]
    testImg = np.expand_dims(testImg, axis=0)
    out = model.predict({'inputimg': testImg})
    pred_pos = out[0][0]
    pred_wpgr = out[1][0]
    list_pred_pos.append(pred_pos)
    list_pred_wpgr.append(pred_wpgr)

# METRICS
theta = rotation_error(list_pred_wpgr, true_label[1])
ER, MAE, CEP, CE95 = calc_metrics(true_label[0], list_pred_pos)
print('3D Metrics on testing data:')
print('MAE = {}, CEP = {}, CE95 = {}'.format(MAE, CEP, CE95))
ER, MAE, CEP, CE95 = calc_metrics_2D(true_label[0], list_pred_pos)
print('2D Metrics on testing data:')
print('MAE = {}, CEP = {}, CE95 = {}'.format(MAE, CEP, CE95))
print('Error in degrees = {}'.format(np.median(theta)))
fname = 'imgs/' + m_name + '_pred_'+ fname + '.png'
plot_results(true_label[0], list_pred_pos, fname)

