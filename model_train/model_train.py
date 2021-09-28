#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'zhonghaolin'
__mtime__ = '2020/12/7'
__function__ = 'model train fea: esm+hmm+ss+rsa!'
"""


from __future__ import print_function
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

seq_length = 700


def sensitivity(y_true, y_pred):
	true_label = K.argmax(y_true, axis=-1)
	pred_label = K.argmax(y_pred, axis=-1)
	INTERESTING_CLASS_ID = 2
	sample_mask = K.cast(K.not_equal(true_label, INTERESTING_CLASS_ID), 'int32')

	TP_tmp1 = K.cast(K.equal(true_label, 0), 'int32') * sample_mask
	TP_tmp2 = K.cast(K.equal(pred_label, 0), 'int32') * sample_mask
	TP = K.sum(TP_tmp1 * TP_tmp2)

	FN_tmp1 = K.cast(K.equal(true_label, 0), 'int32') * sample_mask
	FN_tmp2 = K.cast(K.not_equal(pred_label, 0), 'int32') * sample_mask   
	FN = K.sum(FN_tmp1 * FN_tmp2)

	epsilon = 0.000000001
	return K.cast(TP, 'float') / (K.cast(TP, 'float') + K.cast(FN, 'float') + epsilon)


def precision(y_true, y_pred):
	true_label = K.argmax(y_true, axis=-1)
	pred_label = K.argmax(y_pred, axis=-1)
	INTERESTING_CLASS_ID = 2
	sample_mask = K.cast(K.not_equal(true_label, INTERESTING_CLASS_ID), 'int32')

	TP_tmp1 = K.cast(K.equal(true_label, 0), 'int32') * sample_mask
	TP_tmp2 = K.cast(K.equal(pred_label, 0), 'int32') * sample_mask
	TP = K.sum(TP_tmp1 * TP_tmp2)

	FP_tmp1 = K.cast(K.not_equal(true_label, 0), 'int32') * sample_mask
	FP_tmp2 = K.cast(K.equal(pred_label, 0), 'int32') * sample_mask
	FP = K.sum(FP_tmp1 * FP_tmp2)

	epsilon = 0.000000001
	return K.cast(TP, 'float') / (K.cast(TP, 'float') + K.cast(FP, 'float') + epsilon)


def f1_score(y_true, y_pred):
	pre = precision(y_true, y_pred)
	sen = sensitivity(y_true, y_pred)
	epsilon = 0.000000001
	f1 = 2 * pre * sen / (pre + sen + epsilon)
	return f1


def draw_history_curve(history, name):
	import matplotlib.pyplot as plt
	# plot loss curve
	val_acc = history.history['val_acc']
	val_loss = history.history['val_loss']
	val_f1_score = history.history['val_f1_score']
	acc = history.history['acc']
	loss = history.history['loss']
	f1_score = history.history['f1_score']
	epochs = range(1, len(acc) + 1)
	# png1 Train/Val acc
	plt.title('Accuracy')
	plt.plot(epochs, acc, 'red', label='Train acc')
	plt.plot(epochs, val_acc, 'blue', label='Validation acc')
	plt.legend()
	plt.savefig("png_dir/{}_acc.png".format(name))
	plt.close()
	# png2 Train loss
	plt.title('Loss')
	plt.plot(epochs, loss, 'red', label='Train loss')
	plt.plot(epochs, val_loss, 'blue', label='Validation loss')
	plt.legend()
	plt.savefig("png_dir/{}_loss.png".format(name))
	plt.close()
	# png3 Train f1_score
	plt.title('f1_score')
	plt.plot(epochs, f1_score, 'red', label='Train f1_score')
	plt.plot(epochs, val_f1_score, 'blue', label='Validation f1_score')
	plt.legend()
	plt.savefig("png_dir/{}_f1score.png".format(name))
	plt.close()


def parse_history(history_dict):
	epoch = len(history_dict["val_loss"])
	min_loss_ind = history_dict["val_loss"].index(min(history_dict["val_loss"]))
	ind = min_loss_ind
	min_val_loss = (history_dict["val_loss"][ind], history_dict['val_precision'][ind], history_dict['val_sensitivity'][ind], history_dict['val_f1_score'][ind])

	max_f1_ind = history_dict['val_f1_score'].index(max(history_dict['val_f1_score']))
	ind = max_f1_ind
	max_val_fi_score = (history_dict["val_loss"][ind], history_dict['val_precision'][ind], history_dict['val_sensitivity'][ind], history_dict['val_f1_score'][ind])
	print("训练了{}轮.".format(epoch))
	print("最低val_loss, {}epoch 结果: val_loss:{}, val_precision:{}, val_sensitivity:{}, val_f1:{}".format(epoch-5,
																							  min_val_loss[0],
																							  min_val_loss[1],
																							  min_val_loss[2],
																							  min_val_loss[3]))
	print("最低val_f1, {} epoch 结果: val_loss:{}, val_precision:{}, val_sensitivity:{}, val_f1:{}".format(max_f1_ind+1,
																							  max_val_fi_score[0],
																							  max_val_fi_score[1],
																							  max_val_fi_score[2],
																							  max_val_fi_score[3]))


def data(data_path):
	x1_train = np.load(os.path.join(data_path, "x1_all.npy"))
	y_train = np.load(os.path.join(data_path, "y_all.npy"))
	print("x1_train: ", x1_train.shape)
	print("y_train: ", y_train.shape)
	print("load data done!")
	return x1_train, y_train


from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization
from keras.layers import Activation, Dense, Dropout
from keras.layers import Dropout, GRU, LSTM, TimeDistributed
from keras.layers import Embedding
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import concatenate, add
from keras import regularizers

class ResNet:

	@staticmethod
	def residual_module(x, filters=64, kernel_size=3, reg=1e-4):
		"""
		param: x: The input to the residual module.
			   filters: The number of the filters that will be learned by the final CONV in the bottlenecks.最终卷积层的输出
			   kernel_size: 64
		return:
			   x: Return the output of the residual module
		"""
		shortcut = x

		# The first block of the ResNet module
		conv1 = Conv1D(filters, kernel_size, padding="same", use_bias=False,
					   kernel_regularizer=regularizers.l1_l2(reg))(x)
		bn1 = BatchNormalization(scale=False, center=True)(conv1)
		act1 = Activation("relu")(bn1)

		# The second block of the ResNet module
		conv2 = Conv1D(filters, kernel_size, padding="same", use_bias=False,
					   kernel_regularizer=regularizers.l1_l2(reg))(act1)
		bn2 = BatchNormalization(scale=False, center=True)(conv2)
		act2 = Activation("relu")(bn2)

		print("res_block", x.shape[-1], act2.shape[-1])
		if x.shape[-1] == act2.shape[-1]:
			shortcut = x
		else:
			shortcut = Conv1D(filters, kernel_size, padding="same", use_bias=False,
							   kernel_regularizer=regularizers.l1_l2(reg))(x)
			shortcut = BatchNormalization(scale=False, center=True)(shortcut)

		x = add([shortcut, conv2])
		x = Activation("relu")(x)
		return x

	@staticmethod
	def build():

		input = Input(shape=(seq_length, 1315), dtype='float32')
		stages = [2, 2, 2, 1]
		filters_ls = [64, 128, 256, 512]
		kernal_size_ls = [3, 3, 3, 3]
		reg = 0.001
		x = BatchNormalization()(input)
		for i in range(0, len(stages)):
			for j in range(0, stages[i]):
				# Apply a residual module.
				x = ResNet.residual_module(x, filters_ls[i], kernal_size_ls[i], reg)

		hidden_units = 512
		lstm_1 = Bidirectional(LSTM(hidden_units, return_sequences=True), merge_mode="sum")(x)
		lstm_1 = BatchNormalization()(lstm_1)
		dense_1 = TimeDistributed(Dense(512, kernel_regularizer=regularizers.l2(0.001)))(lstm_1)
		dense_1 = BatchNormalization()(dense_1)
		dense_2 = TimeDistributed(Dense(256, kernel_regularizer=regularizers.l2(0.001)))(dense_1)
		dense_2 = BatchNormalization()(dense_2)
		dense_3 = TimeDistributed(Dense(128, kernel_regularizer=regularizers.l2(0.001)))(dense_2)
		dense_3 = BatchNormalization()(dense_3)
		output = TimeDistributed(Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.001)))(dense_3)

		model = Model(inputs=input, outputs=output)
		print(model.summary())
		return model


def train_model(X1_train, Y_train):
	"""
	train Res-Dom
	"""
	from keras.optimizers import SGD, RMSprop, Adam
	from keras.utils import multi_gpu_model
	from keras.callbacks import EarlyStopping, ModelCheckpoint

	model = ResNet.build()
	savepath = '/path/to/res-dom-all-{epoch:02d}-{val_loss:.2f}.hdf5'
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.002, mode='min', patience=50)
	saveBestModel = ModelCheckpoint(savepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True,
										mode='min')
	cb = [early_stopping, saveBestModel]
	batch_size = 256
	model = multi_gpu_model(model, gpus=2)
	model.compile(loss='categorical_crossentropy',
					optimizer=Adam(lr=0.0001),
					metrics=['accuracy', precision, sensitivity, f1_score])
	history = model.fit(X1_train, Y_train, batch_size=256, epochs=200,
						validation_split=0.2, shuffle = True, callbacks=cb,
						verbose=1)

	# plot loss curve
	parse_history(history.history)
	png_name = "Res-Dom_model_train"
	draw_history_curve(history, png_name)

	print("trainning is end.")


if __name__ == "__main__":
	data_path = "/path/to/"
	X_train, Y_train = data(data_path)
	# print(X_train.shape, X_train2.shape, Y_train.shape)
	train_model(X_train, Y_train)
