import tensorflow as tf
import numpy as np

import CNN_input

FLAGS = CNN_input.FLAGS

# Global Constants
IMAGE_SIZE = CNN_input.IMAGE_SIZE
IMAGE_CHANNEL_DEPTH = CNN_input.IMAGE_CHANNEL_DEPTH
NUM_EXAMPLES_PER_EMPOCH_FOR_TRAIN = CNN_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EMPOCH_FOR_EVAL = CNN_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
NUM_CLASSES = CNN_input.NUM_CLASSES

# Model Constants

# Convolution Layer 1
CONV_1_FILTERS = 64
CONV_1_FILTER_SZ = 5
CONV_1_STRIDE = 2

# Convolution Layer 2
CONV_2_FILTERS = 64
CONV_2_FILTER_SZ = 5
CONV_2_STRIDE = 1

# Max Pool 1
MP1_KSIZE = 2
MP1_STRIDE = 2

# Convolution Layer 3
CONV_3_FILTERS = 96
CONV_3_FILTER_SZ = 3
CONV_3_STRIDE = 1

# Convolution Layer 4
CONV_4_FILTERS = 128
CONV_4_FILTER_SZ = 3
CONV_4_STRIDE = 1

# Max Pool 2
MP2_KSIZE = 2
MP2_STRIDE = 2

# Fully Connected Layer 1
FC1_W_STDDEV = 0.04
FC1_W_DECAY = 0.004
FC1_BIAS_INIT = 0.1
FC1_DROPOUT = 0.5

# Fully Connected Layer 2
FC2_SIZE = 384
FC2_W_STDDEV = 0.04
FC2_W_DECAY = 0.004
FC2_BIAS_INIT = 0.1
FC2_DROPOUT = 0.5

# Fully Connected Softmax Layer 3
FC3_SIZE = 192
FC3_W_STDDEV = 1/FC3_SIZE
FC3_W_DECAY = 0.004
FC3_BIAS_INIT = 0.0

# Training Constants
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 10.0
LEARNING_RATE_DECAY_FACTOR = 0.995
INITIAL_LEARNING_RATE = 0.15
NUM_BATCHES_PER_EPOCH = NUM_EXAMPLES_PER_EMPOCH_FOR_TRAIN / FLAGS.batch_size
DECAY_STEPS = int(NUM_BATCHES_PER_EPOCH * NUM_EPOCHS_PER_DECAY)


def _weight_variable_with_decay(name, shape, stddev, weight_decay=None):
	initial = tf.truncated_normal(name=name, shape=shape, stddev=stddev)

	tf_var = tf.Variable(initial)

	if (weight_decay is not None):
		wd = tf.multiply(tf.nn.l2_loss(tf_var), weight_decay, name='weight_loss')
		tf.add_to_collection('losses', wd)

	return tf_var


def _bias_variable(init, shape):
	initial = tf.constant(init, shape=shape, name='biases')
	return tf.Variable(initial)


def _activation_summary(x):
	tensor_name = x.op.name
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _build_convolution_layer(name, source_layer, filter_size, filter_depth, num_filters, stride, padding):
	with tf.variable_scope(name) as scope:
		kernel = _weight_variable_with_decay('weights',
											 shape=[filter_size, filter_size, filter_depth, num_filters],
											 stddev=5e-2,
											 weight_decay=0.004)
		conv = tf.nn.conv2d(source_layer, kernel, stride, padding=padding)
		biases = _bias_variable(0.1, [num_filters])
		pre_activation = tf.nn.bias_add(conv, biases)
		layer = tf.nn.elu(pre_activation, name=name)
		_activation_summary(layer)

	return layer


def _build_fully_connected_layer(name, source_layer, input_size, output_size, weight_stddev, weight_decay, bias_init, dropout):
	with tf.variable_scope(name) as scope:
		weights = _weight_variable_with_decay('weights',
											  shape=[input_size, output_size],
											  stddev=weight_stddev,
											  weight_decay=weight_decay)
		biases = _bias_variable(bias_init, [output_size])
		fully_con = tf.nn.relu(tf.matmul(source_layer, weights) + biases, name=name)

		if dropout is not None:
			fully_con = tf.nn.dropout(fully_con, 1 - dropout, name=name)

		_activation_summary(fully_con)

		return fully_con

def _print_tensor_shapes(tensor_list):
	if (FLAGS.report_layer_shapes):
		max_len = 0
		for tensor in tensor_list:
			name_len = len(tensor.name.split(':')[0].split('/')[0])
			if (name_len > max_len):
				max_len = name_len

		for tensor in tensor_list:
			name = tensor.name.split(':')[0].split('/')[0]
			print(name + " shape: " + (" "*(max_len - len(name))) + str(tensor.get_shape()))
	return

def inference(is_training, images):
	# Convolution Layer 1
	conv1 = _build_convolution_layer('conv1', images, CONV_1_FILTER_SZ, IMAGE_CHANNEL_DEPTH, CONV_1_FILTERS, [1, CONV_1_STRIDE, CONV_1_STRIDE, 1], 'SAME')

	# Convolution Layer 2
	conv2 = _build_convolution_layer('conv2', conv1, CONV_2_FILTER_SZ, CONV_1_FILTERS, CONV_2_FILTERS, [1, CONV_2_STRIDE, CONV_2_STRIDE, 1], 'SAME')

	# Max Pool 1
	max_pool1 = tf.nn.max_pool(name='pool1', value=conv2, ksize=[1, MP1_KSIZE, MP1_KSIZE, 1], strides=[1, MP1_STRIDE, MP1_STRIDE, 1], padding='SAME')

	# Convolution Layer 3
	conv3 = _build_convolution_layer('conv3', max_pool1, CONV_3_FILTER_SZ, CONV_2_FILTERS, CONV_3_FILTERS, [1, CONV_3_STRIDE, CONV_3_STRIDE, 1], 'SAME')

	# Convolution Layer 4
	conv4 = _build_convolution_layer('conv4', conv3, CONV_4_FILTER_SZ, CONV_3_FILTERS, CONV_4_FILTERS, [1, CONV_4_STRIDE, CONV_4_STRIDE, 1], 'SAME')

	# Max Pool 2
	max_pool2 = tf.nn.max_pool(name='pool2', value=conv4, ksize=[1, MP2_KSIZE, MP2_KSIZE, 1], strides=[1, MP2_STRIDE, MP2_STRIDE, 1], padding='SAME')
	max_pool2_shape = max_pool2.get_shape()

	# Average Pool
	avg_pool = tf.nn.avg_pool(name='avg_pool', value=max_pool2, ksize=[1, max_pool2_shape[1], max_pool2_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')

	# Reshape Into Depth
	reshape = tf.reshape(avg_pool, [FLAGS.batch_size, -1])

	#Dropouts
	if (is_training):
		fc1_dropout = FC1_DROPOUT
		fc2_dropout = FC2_DROPOUT
	else:
		fc1_dropout, fc2_dropout = None

	# Fully Connected
	fc1 = _build_fully_connected_layer('fc1', reshape, CONV_4_FILTERS, FC2_SIZE, weight_stddev=FC1_W_STDDEV, weight_decay=FC1_W_DECAY, bias_init=FC1_BIAS_INIT, dropout=FC1_DROPOUT)

	# Fully Connected
	fc2 = _build_fully_connected_layer('fc2', fc1, FC2_SIZE, FC3_SIZE, weight_stddev=FC2_W_STDDEV, weight_decay=FC2_W_DECAY, bias_init=FC2_BIAS_INIT, dropout=FC2_DROPOUT)

	# Softmax
	with tf.variable_scope('softmax_linear') as scope:
		weights = _weight_variable_with_decay('weights',
											  [FC3_SIZE, NUM_CLASSES],
											  stddev=FC3_W_STDDEV,
											  weight_decay=FC3_W_DECAY)
		biases = _bias_variable(FC3_BIAS_INIT, [NUM_CLASSES])
		softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)
		_activation_summary(softmax_linear)

	_print_tensor_shapes([images, conv1, conv2, max_pool1, conv3, conv4, max_pool2, avg_pool, reshape, fc1, fc2, softmax_linear])

	return softmax_linear


def loss(logits, labels):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	correct_prediction = tf.equal(labels, tf.argmax(logits, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summeries(total_loss):
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	for loss in losses + [total_loss]:
		tf.summary.scalar(loss.op.name + ' (raw)', loss)
		tf.summary.scalar(loss.op.name, loss_averages.average(loss))

	return loss_averages_op


def train(total_loss, global_step):
	learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
											   global_step,
											   DECAY_STEPS,
											   LEARNING_RATE_DECAY_FACTOR,
											   staircase=True)
	tf.summary.scalar('learning_rate', learning_rate)

	loss_averages_op = _add_loss_summeries(total_loss)

	with tf.control_dependencies([loss_averages_op]):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		gradients = optimizer.compute_gradients(total_loss)

	apply_gradient_op = optimizer.apply_gradients(gradients, global_step=global_step)

	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	for grad, var in gradients:
		if (grad is not None):
			tf.summary.histogram(var.op.name + '/gradients', grad)

	variables_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variables_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return train_op
