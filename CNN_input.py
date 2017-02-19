import os
import tarfile

import tensorflow as tf

DIRECTORY = os.path.dirname(__file__)

FLAGS = tf.app.flags.FLAGS

# Basic model parameters
tf.app.flags.DEFINE_string('data_dir', os.path.join(DIRECTORY, "data", "cifar10_data"),
						   """Path to CIFAR-10 data directory""")

tf.app.flags.DEFINE_string('train_dir', os.path.join(DIRECTORY, "data", "cifar10_train"),
							"""Directory where to write event logs """
							"""and checkpoint.""")

tf.app.flags.DEFINE_string('eval_dir', os.path.join(DIRECTORY, "data", "cifar10_eval"),
						   """Directory where to write event logs.""")

tf.app.flags.DEFINE_integer('batch_size', 128,
							"""Number of images to process in a batch.""")

tf.app.flags.DEFINE_integer('max_steps', 80000,
							"""Number of batches to run.""")

tf.app.flags.DEFINE_boolean('log_device_placement', False,
							"""Whether to log device placement.""")

tf.app.flags.DEFINE_boolean('report_layer_shapes', True,
							"""Whether to print shapes of network layers""")

tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")

tf.app.flags.DEFINE_integer('eval_interval_secs', 3,
                            """How often to run the eval.""")

tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")

tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

#Global Constants
LABEL_BYTES = 1
IMAGE_SIZE = 32
IMAGE_CHANNEL_DEPTH = 3
IMAGE_BYTES = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNEL_DEPTH
RECORD_BYTES = LABEL_BYTES + IMAGE_BYTES

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 3000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 300
MIN_FRACTION_EXAMPLES_QUEUED = 0.4

NUM_CLASSES = 10

FILE_NAME = 'cifar-10-binary.tar.gz'
BATCHES_BIN_FOLDER = 'cifar-10-batches-bin'
BATCH_FILE_FORMAT = 'data_batch_%d.bin'


# Downloads CIFAR-10 Image Data
def download_extract():
	dest_directory = FLAGS.data_dir

	if (os.path.exists(os.path.join(dest_directory, "cifar-10-batches-bin"))):
		return

	if (not os.path.exists(dest_directory)):
		raise ValueError('Data Directory "' + dest_directory + '" doest not exist!')


	filepath = os.path.join(dest_directory, FILE_NAME)

	if (os.path.exists(filepath)):
		tarfile.open(filepath, 'r:gz').extractall(dest_directory)
	else:
		raise ValueError(FILE_NAME + " does not exist!")


# Reads file from CIFAR-10 filename queue
def read_file(filename_queue):
	reader = tf.FixedLengthRecordReader(record_bytes=RECORD_BYTES)
	key, value = reader.read(filename_queue)

	record_bytes = tf.decode_raw(value, tf.uint8)

	label = tf.cast(tf.strided_slice(record_bytes, [0], [LABEL_BYTES], strides=[1]), tf.int32)

	depth_major = tf.reshape(tf.strided_slice(record_bytes, [LABEL_BYTES], [RECORD_BYTES], strides=[1]),
							 [IMAGE_CHANNEL_DEPTH, IMAGE_SIZE, IMAGE_SIZE])

	uint8image = tf.transpose(depth_major, [1, 2, 0])

	return uint8image, label


# Generates Batch
def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
	num_preprocess_threads = 16
	if (shuffle):
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size)

	tf.summary.image('images', images)

	return images, tf.reshape(label_batch, [batch_size])


def construct_inputs(is_evaluation_inputs, shuffle):
	if (is_evaluation_inputs):
		filenames = [os.path.join(FLAGS.data_dir, BATCHES_BIN_FOLDER, 'test_batch.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
	else:
		filenames = [os.path.join(FLAGS.data_dir, BATCHES_BIN_FOLDER, (BATCH_FILE_FORMAT % i)) for i in range(1, 6)]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

	for file in filenames:
		if (not tf.gfile.Exists(file)):
			raise ValueError('Failed to find file: ' + file)

	filename_queue = tf.train.string_input_producer(filenames)

	uint8image, label = read_file(filename_queue)

	float_image = tf.cast(uint8image, tf.float32)

	# Random distortions to reduce training overfitting
	if (not is_evaluation_inputs):
		float_image = tf.image.random_flip_left_right(float_image)
		float_image = tf.image.random_brightness(float_image, max_delta=63)
		float_image = tf.image.random_contrast(float_image, lower=0.2, upper=1.8)

	finalized_image = tf.image.per_image_standardization(float_image)
	finalized_image.set_shape([IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL_DEPTH])
	label.set_shape([1])

	min_queue_examples = int(num_examples_per_epoch * MIN_FRACTION_EXAMPLES_QUEUED)

	return _generate_image_and_label_batch(finalized_image,
										   label,
										   min_queue_examples,
										   FLAGS.batch_size,
										   shuffle=shuffle)
