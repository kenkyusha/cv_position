import numpy as np
import pdb

def calc_metrics(truth, pred, verbose = 0):
	if type(truth) and type(pred) is list:
		truth = np.array(truth)
		pred = np.array(pred)

	ER = np.sqrt(np.sum(np.square(pred - truth), axis = 1))
	MAE = np.mean(ER)
	norm_ER = ER/np.max(ER)
	sort_ER = np.sort(ER)
	CEP_ind = int(sort_ER.shape[0]*0.5)
	CEP95_ind = int(sort_ER.shape[0]*0.95)
	if verbose == 1:
		print('MAE= ', MAE)
		print('CEP= ', sort_ER[CEP_ind])
		print('CEP95= ', sort_ER[CEP95_ind])
	return ER, MAE, sort_ER[CEP_ind], sort_ER[CEP95_ind]

def calc_metrics_2D(truth, pred, verbose = 0):
	if type(truth) and type(pred) is list:
		truth = np.array(truth)
		pred = np.array(pred)

	ER = np.sqrt(np.sum(np.square(pred[:,:2] - truth[:,:2]), axis = 1))
	MAE = np.mean(ER)
	norm_ER = ER/np.max(ER)
	sort_ER = np.sort(ER)
	CEP_ind = int(sort_ER.shape[0]*0.5)
	CEP95_ind = int(sort_ER.shape[0]*0.95)
	if verbose == 1:
		print('MAE= ', MAE)
		print('CEP= ', sort_ER[CEP_ind])
		print('CEP95= ', sort_ER[CEP95_ind])
	return ER, MAE, sort_ER[CEP_ind], sort_ER[CEP95_ind]

def quaternion_to_rot(w, x, y, z):
	sinr = 2 * (w * x + y * z)
	cosr = 1 - 2 * (x * x + y * y)
	roll = np.arctan2(sinr, cosr)

	sinp = 2* (w * y - z * x)
	roll_list = []
	yaw_list = []
	pitch_list = []
	if type(sinp) == np.ndarray:
		for i in range(len(sinp)):
			if abs(sinp[i]) >= 1:
				pitch = np.copysign(np.pi / 2, sinp[i])
			else:
				pitch = np.arcsin(sinp[i])

			siny = 2 * (w * z + x * y)
			cosy = 1 - 2 * (y * y + z * z)
			yaw = np.arctan2(siny, cosy)

			roll_list.append(np.rad2deg(roll))
			pitch_list.append(np.rad2deg(pitch))
			yaw_list.append(np.rad2deg(yaw))
	else:
		if abs(sinp) >= 1:
			pitch = np.copysign(np.pi / 2, sinp)
		else:
			pitch = np.arcsin(sinp)

		siny = 2 * (w * z + x * y)
		cosy = 1 - 2 * (y * y + z * z)
		yaw = np.arctan2(siny, cosy)

		roll_list.append(np.rad2deg(roll))
		pitch_list.append(np.rad2deg(pitch))
		yaw_list.append(np.rad2deg(yaw))

	return roll_list, pitch_list, yaw_list

def rotation_error(pred_wpgr, true_wpgr):
	if type(true_wpgr) and type(pred_wpgr) is list:
		true_wpgr = np.array(true_wpgr)
		pred_wpgr = np.array(pred_wpgr)
	if len(pred_wpgr.shape) > 1:
		q1 = quaternion_to_rot(pred_wpgr[:,0], pred_wpgr[:,1], pred_wpgr[:,2], pred_wpgr[:,3])
		q2 = quaternion_to_rot(true_wpgr[:,0], true_wpgr[:,1], true_wpgr[:,2], true_wpgr[:,3])
	else:
		q1 = quaternion_to_rot(pred_wpgr[0], pred_wpgr[1], pred_wpgr[2], pred_wpgr[3])
		q2 = quaternion_to_rot(true_wpgr[0], true_wpgr[1], true_wpgr[2], true_wpgr[3])
	theta = abs(np.array(q1[2])) - abs(np.array(q2[2]))
	return theta
