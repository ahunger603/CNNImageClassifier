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

#Model Constants
CONV_1_FILTERS = 32
CONV_1_FILTER_SZ = 5

CONV_2_FILTERS = 48
CONV_2_FILTER_SZ = 5

CONV_3_FILTERS = 64
CONV_3_FILTER_SZ = 3

CONV_4_FILTERS = 64
CONV_4_FILTER_SZ = 3

FC2_SIZE = 384
FC3_SIZE = 192

# Training Constants
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1


def _weight_variable_with_decay(name, shape, stddev, weight_decay=None):
	initial = tf.truncated_normal(name=name, shape=shape, stddev=stddev)

	tf_var = tf.Variable(initial)

	if (weight_decay is not None):
		wd = tf.multiply(tf.nn.l2_loss(tf_var), weight_decay, name='weight_loss')

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
											 weight_decay=0.0)
		conv = tf.nn.conv2d(source_layer, kernel, stride, padding=padding)
		biases = _bias_variable(0.1, [num_filters])
		pre_activation = tf.nn.bias_add(conv, biases)
		layer = tf.nn.elu(pre_activation, name=name)
		_activation_summary(layer)

	return layer


def _build_fully_connected_layer(name, source_layer, input_size, output_size, weight_stddev, weight_decay, bias_init):
	with tf.variable_scope(name) as scope:
		weights = _weight_variable_with_decay('weights',
											  shape=[input_size, output_size],
											  stddev=weight_stddev,
											  weight_decay=weight_decay)
		biases = _bias_variable(bias_init, [output_size])
		fully_con = tf.nn.relu(tf.matmul(source_layer, weights) + biases, name=name)
		_activation_summary(fully_con)

		return fully_con


def inference(images):
	print("IMG SHAPE: " + str(images.get_shape()))

	# Convolution Layer 1
	conv1 = _build_convolution_layer('conv1', images, CONV_1_FILTER_SZ, IMAGE_CHANNEL_DEPTH, CONV_1_FILTERS, [1, 1, 1, 1], 'SAME')

	print("CL1 SHAPE: " + str(conv1.get_shape()))

	# Convolution Layer 2
	conv2 = _build_convolution_layer('conv2', conv1, CONV_2_FILTER_SZ, CONV_1_FILTERS, CONV_2_FILTERS, [1, 1, 1, 1], 'SAME')

	print("CL2 SHAPE: " + str(conv2.get_shape()))

	# Max Pool 1
	max_pool1 = tf.nn.max_pool(name='pool1', value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	print("MP1 SHAPE: " + str(max_pool1.get_shape()))

	# Convolution Layer 3
	conv3 = _build_convolution_layer('conv3', max_pool1, CONV_3_FILTER_SZ, CONV_2_FILTERS, CONV_3_FILTERS, [1, 1, 1, 1], 'SAME')

	print("CL3 SHAPE: " + str(conv3.get_shape()))

	# Convolution Layer 4
	conv4 = _build_convolution_layer('conv4', conv3, CONV_4_FILTER_SZ, CONV_3_FILTERS, CONV_4_FILTERS, [1, 1, 1, 1], 'SAME')

	print("CL4 SHAPE: " + str(conv4.get_shape()))

	# Max Pool 2
	max_pool2 = tf.nn.max_pool(name='pool2', value=conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
	max_pool_shape = max_pool2.get_shape()

	print("MP2 SHAPE: " + str(max_pool_shape))

	# Average Pool
	avg_pool = tf.nn.avg_pool(name='avg_pool', value=max_pool2, ksize=[1, max_pool_shape[1], max_pool_shape[2], 1], strides=[1, 1, 1, 1], padding='VALID')

	print("AP SHAPE: " + str(avg_pool.get_shape()))

	# Reshape Into Depth
	reshape = tf.reshape(avg_pool, [FLAGS.batch_size, -1])

	print("RESHAPED: " + str(reshape.get_shape()))

	# Fully Connected
	fc1 = _build_fully_connected_layer('fc1', reshape, CONV_4_FILTERS, FC2_SIZE, weight_stddev=0.04, weight_decay=0.004, bias_init=0.1)

	print("FC1 SHAPE: " + str(fc1.get_shape()))

	# Fully Connected
	fc2 = _build_fully_connected_layer('fc2', fc1, FC2_SIZE, FC3_SIZE, weight_stddev=0.04, weight_decay=0.004, bias_init=0.1)

	print("FC2 SHAPE: " + str(fc2.get_shape()))

	# Softmax
	with tf.variable_scope('softmax_linear') as scope:
		weights = _weight_variable_with_decay('weights',
											  [FC3_SIZE, NUM_CLASSES],
											  stddev=1/FC3_SIZE,
											  weight_decay=0.0)
		biases = _bias_variable(0.0, [NUM_CLASSES])
		softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)
		_activation_summary(softmax_linear)

	print("SOFTMAX SHAPE: " + str(softmax_linear.get_shape()))

	return softmax_linear


def loss(logits, labels):
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	tf.add_to_collection('losses', cross_entropy_mean)

	return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss, global_step):
	num_batches_per_epoch = NUM_EXAMPLES_PER_EMPOCH_FOR_TRAIN / FLAGS.batch_size
	decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
											   global_step,
											   decay_steps,
											   LEARNING_RATE_DECAY_FACTOR,
											   staircase=True)
	tf.contrib.depreciated.scalar_summary('learning_rate', learning_rate)



	return 0
