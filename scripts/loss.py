import keras
import keras.backend as K

def euc_loss1x(y_true, y_pred):
	lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
	return (1 * lx)

def euc_loss1q(y_true, y_pred):
	lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
	return (100 * lq)