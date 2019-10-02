#start
import tensorflow as tf
import tqdm
from scripts.nets import smallNet
from scripts.utils import *
from scripts.metrics import *
from scripts.loss import *
import argparse
import pdb
import keras
import random

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='dataset that is used for training')
parser.add_argument('--test_data', help='dataset that is used for testing during training')
parser.add_argument('--model', help='name of the model to load', required=True)
parser.add_argument('--restore', help='restore training from certain point')
parser.add_argument('--channels', help='num of channels')
args = parser.parse_args()

if args.channels == '3':
	img_shape = (224,224,3)
	ch_flag = True
else:
	img_shape = (224,224,1)
	ch_flag = False

# insert here all possible models from scripts.nets:
if args.model == 'smallNet':
	model = smallNet(img_shape)
	m_name = 'smallNet'
else:
	print('Enter a valid model, exiting... ')
	exit(0)

try:
	# automatically choose maximum number of GPUs
	model = tf.keras.multi_gpu_model(model, gpus=None)
except:
	pass

# optimizer
opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999)
# compile a model
model.compile(optimizer=opt, loss={'cls1_fc_pose_xyz': euc_loss1x, 'cls1_fc_pose_wpqr': euc_loss1q})
model.summary()

# data preparation:
test_data = gen_test_data(args.test_data, flag = True, normal = ch_flag)

count = 0
count = len(open(args.data).readlines())
print('Size of training set = {}'.format(count))
print('Size of testing set = {}'.format(len(test_data[0])))

batch_size = 32
num_epochs = 200
train_it = int(count/batch_size)
train_idx = list(range(count))
#shuffle the indexes
random.shuffle(train_idx)

lines = []
for line in tqdm(open(args.data)):
	if line.isspace():
		continue
	else:
		lines.append(line)

make_sure_path_exists('summary')
make_sure_path_exists('imgs')
make_sure_path_exists('wts')

f = open('summary/err_metrics2D_{}.txt'.format(m_name), 'w')
fo = open('summary/err_metrics3D_{}.txt'.format(m_name), 'w')

# Training loop:
for e in tqdm(range(num_epochs)):
	print('Training progress {}/{} epochs'.format(e+1,num_epochs))
	for i in tqdm(range(train_it)):
		print('Training progress {}/{} iterations'.format(i+1,train_it))
		input_batch = None
		label_batch = None
		input_batch, label_batch = gen_train_batch(train_idx, batch_size, i, lines, normal = ch_flag)
		y_pos = label_batch[0]
		y_wpgr = label_batch[1] 
		# fit the model
		history = model.fit(input_batch, [y_pos, y_wpgr], epochs=1, batch_size=batch_size, verbose=0)
	if e % 5 == 0: # 5
		# test the network
		imgs, true_pos, true_wpgr = grab_test(test_data)
		list_pred_pos = []
		list_pred_wpgr = []
		for i in tqdm(range(len(imgs))):
			testImg = imgs[i]
			testImg = np.expand_dims(testImg, axis=0)
			inputsImg = np.zeros([1, 224, 224, 1])
			inputsImg[0,:] = imgs[i]
			out = model.predict({'inputimg': testImg})
			pred_pos = out[0][0]
			pred_wpgr = out[1][0]
			list_pred_pos.append(pred_pos)
			list_pred_wpgr.append(pred_wpgr)
		
		# METRIXS
		theta = rotation_error(list_pred_wpgr, true_wpgr)
		ER, MAE, CEP, CE95 = calc_metrics(true_pos, list_pred_pos)
		print('3D Metrics on testing data:')
		print('MAE = {}, CEP = {}, CE95 = {}'.format(MAE, CEP, CE95))
		string = str(e)+ ',' + str(MAE) + ',' + str(CEP) + ',' + str(CE95) + ',' + str(np.median(theta)) + '\n'
		fo.write(string)
		ER, MAE, CEP, CE95 = calc_metrics_2D(true_pos, list_pred_pos)
		print('2D Metrics on testing data:')
		print('MAE = {}, CEP = {}, CE95 = {}'.format(MAE, CEP, CE95))
		print('Error in degrees = {}'.format(np.median(theta)))
		string = str(e)+ ',' + str(MAE) + ',' + str(CEP) + ',' + str(CE95) + ',' + str(np.median(theta)) + '\n'
		f.write(string)
		fname = 'imgs/test_' + m_name+ '_' + str(e) + '.png'
		plot_results(true_pos, list_pred_pos, fname)
		# SAVE MODEL
		wts_path = 'wts_' + m_name + '_epoch_' + str(e) + '.h5'
		m_path = 'net_arch'  + m_name + '_' + '_epoch_' + str(e) + '.h5'
		model.save_weights('wts/' + wts_path)

		if e == 0:
			# only necessary to save once! if at all.
			model.save('wts/' + m_path)
