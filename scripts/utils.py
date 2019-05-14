from tqdm import tqdm
import cv2
import os, errno
import numpy as np
import matplotlib.pyplot as plt
import pdb

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def preprocess(img, w, h):
	img = cv2.resize(img, (w, h))
	img = cv2.Canny(img, 50, 150)
	img = img.reshape(w,h,1)
	return img

def gen_test_data(filename, flag = False, w=224, h=224):
	images_batch = []
	xyz = []
	wpgr = []
	with open(filename) as f:
		print('preparing data for testing ' + filename)
		i = 0
		for line in tqdm(f):
			if line.isspace():
				continue
			i = i + 1 
			if flag:
			# for loading only the partial set
				if i % 95 == 0 or i % 96 == 0 or i % 97 == 0 or i % 98 == 0 or i % 99 == 0 or i % 100 == 0:
					fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
					img_name = fname[:1]+'/'+os.path.dirname(filename)+fname[1:]
					img = cv2.imread(img_name)
					# apply preprocessing
					img = preprocess(img, w, h)
					images_batch.append(img)
					xyz.append(np.array((np.float(p0), np.float(p1), np.float(p2))))
					wpgr.append(np.array((np.float(p3), np.float(p4), np.float(p5), np.float(p6))))
			else:
				fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
				img_name = fname[:1]+'/'+os.path.dirname(filename)+fname[1:]
				img = cv2.imread(img_name)
				# apply preprocessing
				img = preprocess(img, w, h)
				images_batch.append(img)
				xyz.append(np.array((np.float(p0), np.float(p1), np.float(p2))))
				wpgr.append(np.array((np.float(p3), np.float(p4), np.float(p5), np.float(p6))))

	return (images_batch, [np.asarray(xyz), np.asarray(wpgr)])

def gen_train_batch(indexes, batch, iteration, lines, w=224, h=224):
	images_batch = []
	xyz = []
	wpgr = []
	idx_offset = batch * iteration
	for idx in range(idx_offset, idx_offset+batch):
		rnd_idx = indexes[idx]
		fname, p0, p1, p2, p3, p4, p5, p6 = lines[rnd_idx].split()

		img_name = fname[:1]+'/data'+fname[1:]
		img = cv2.imread(img_name)
		# apply preprocessing
		img = preprocess(img, w, h)
		images_batch.append(img)
		xyz.append(np.array((np.float(p0), np.float(p1), np.float(p2))))
		wpgr.append(np.array((np.float(p3), np.float(p4), np.float(p5), np.float(p6))))
	return np.asarray(images_batch), [np.asarray(xyz), np.asarray(wpgr)]

def grab_test(source):
	imgs = []
	pos = []
	rot = [] 
	for i in range(len(source[0])):
		imgs.append(source[0][i])
		pos.append(source[1][0][i])
		rot.append(source[1][1][i])
	return imgs, pos, rot


def plot_results(true_pos, pred_pos, fname):
	if type(true_pos) and type(pred_pos) is list:
		true_pos = np.array(true_pos)
		pred_pos = np.array(pred_pos)
	plt.clf()
	plt.xlim(min(true_pos[:,0])-1,max(true_pos[:,0])+1)
	plt.ylim(min(true_pos[:,1])-1,max(true_pos[:,1])+1)
	
	ax = plt.gca()
	ax.set_xlabel('x [m]', fontsize=15)
	ax.set_ylabel('y [m]', fontsize=15)
	fpredx= []
	fpredy= []
	col = []
	errthresh = 2 #2
	for i in range(0, len(true_pos), 1):
		diffx = true_pos[i,0] - pred_pos[i,0]
		diffy = true_pos[i,1] - pred_pos[i,1]
		maxerr = min(errthresh,max(abs(diffx), abs(diffy)))
		col.append((maxerr/errthresh, (errthresh-maxerr)/errthresh, 0)) #rgb
		fpredx.append(pred_pos[i,0])
		fpredy.append(pred_pos[i,1])
	
	#### draw error shift lines 
	for i in range(0, len(true_pos), 1):
		x2, y2 = [pred_pos[i,0],true_pos[i,0]] , [pred_pos[i,1],true_pos[i,1]]
		plt.plot(x2, y2, color=(0.8,0.8,0.8), alpha=0.5, linestyle='dashed', zorder=2)
	
	# draw predictions and reference  
	plt.scatter(true_pos[:,0], true_pos[:,1], color='b', s=1, zorder=3)
	plt.scatter(fpredx, fpredy, color=col, s=2, zorder=4)
	
	plt.savefig(fname, bbox_inches='tight')

def load_labels(filename):
	xyz = []
	wpgr = []
	img_list = []
	with open(filename) as f:
		i = 0
		for line in tqdm(f):
			if line.isspace():
				continue
			i = i + 1 
			fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
			img_name = fname[:1]+'/'+os.path.dirname(filename)+fname[1:]
			xyz.append(np.array((np.float(p0), np.float(p1), np.float(p2))))
			wpgr.append(np.array((np.float(p3), np.float(p4), np.float(p5), np.float(p6))))
			img_list.append(fname)

	return img_list, xyz, wpgr

def compare_label(filename, compare_list, sensitivity = 0.1):
	idx_pairs = []
	diff_list = []

	with open(filename) as f:
		i = 0
		for line in tqdm(f):
			if line.isspace():
				continue
			fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
			xyz = np.array((np.float(p0), np.float(p1), np.float(p2)))
			# find difference between our test list and current train img, take the abs as we sum
			diff = abs(compare_list - xyz)
			# sum the 3 coords together for easier comparison
			diff_sum = np.sum(diff, axis=1)
			diff_abs = np.abs(diff_sum)
			# find samples with respect to our sensitivity range
			idx = np.where(diff_abs <= sensitivity)
			# store first the idx of test image and training image idx
			if not all(idx[0]) == True:
				for j in range(len(idx[0])):
					print('idx[0] = ', idx)
					print('abs diff = ', diff_abs[idx[0][j]])
					print('train img = ', i)
					#pdb.set_trace()
					idx_pairs.append([idx[0][j], i])
					diff_list.append(diff_abs[idx[0][j]])

			i = i + 1 

	return np.array(idx_pairs), diff_list