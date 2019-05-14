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
import os, sys
import cv2

parser = argparse.ArgumentParser()
#parser.add_argument('--fname', help="name of test", required=True)
parser.add_argument('--test_data', help="dataset for testing", required=True)
parser.add_argument('--train_data', help="dataset for testing", required=True)
parser.add_argument('--net', help="name of the model to load", required=False)
parser.add_argument('--wts', help="path to the trained neural network to test ", required=False)

args = parser.parse_args()

# pred mode included
if args.net != None and args.wts != None:
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

img_list, test_pos, test_wpgr = load_labels(args.test_data)

# returns the list with image pairs
sens = 0.5
idx_pairs, diff = compare_label(args.train_data, test_pos, sensitivity = sens)
print('***************************************')
print('Found {} matching pairs with sensitivity of {}'.format(len(idx_pairs), sens))
#pdb.set_trace()

# load the train images
f = open(args.train_data)
lines = f.readlines()  
# display the image pairs side by side
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,470)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

make_sure_path_exists('res')

for i in range(len(idx_pairs)):
    fname = img_list[idx_pairs[i,0]]
    test_img = cv2.imread(fname[:1]+'/'+os.path.dirname(args.test_data)+fname[1:])
    if args.net != None and args.wts != None:
        testImg = preprocess(test_img, 224, 224)
        testImg = np.expand_dims(testImg, axis=0)
        out = model.predict({'inputimg': testImg})
        pred_pos = out[0][0]
        pred_wpgr = out[1][0]
        theta = rotation_error(pred_wpgr, test_wpgr[idx_pairs[i,0]])
        err = np.linalg.norm(test_pos[idx_pairs[i,0]] - pred_pos)
        string = 'Test, error = {}, deg = {}'.format(np.round(err,2), np.round(abs(np.median(theta)),2))
    else:
        string = 'Test'
    cv2.putText(test_img, string, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    ### train img
    fname, p0, p1, p2, p3, p4, p5, p6 = lines[idx_pairs[i,1]].split()
    train_img = cv2.imread(fname[:1]+'/'+os.path.dirname(args.train_data)+fname[1:])
    string = 'Train img, difference = {}'.format(np.round(diff[i]),2)
    cv2.putText(train_img, string, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    frame1 = np.concatenate((test_img, train_img), axis = 1)
    cv2.imwrite('res/res_sens_' + str(sens)+ '_'+ str(i) + ".jpg", frame1)
    if cv2.waitKey(1) == 0x1b: # ESC
        print('ESC pressed. Exiting ...')
        break
    '''
    cv2.imshow('img',frame1)
    if cv2.waitKey(0) & 0xFF == ord ('q'):
        print('exiting')
        sys.exit(0)
    if cv2.waitKey(0) & 0xFF == ord ('n'):
        continue
    '''



