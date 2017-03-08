import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox, TextArea)

import CNN_input
import CNN_model

FLAGS = CNN_input.FLAGS

LABELS = ["Air Plane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
DISPLAY_TOP_K = 3


def reformat_image(image):

	for i in range(CNN_input.IMAGE_SIZE):
		for j in range(CNN_input.IMAGE_SIZE):
			image[i][j][0] /= 255
			image[i][j][1] /= 255
			image[i][j][2] /= 255

	return image


def get_top_k(prediction, k):
	prediction_cpy = []
	for i in range(len(prediction)):
		prediction_cpy.append(prediction[i])

	top_k = []
	for i in range(k):
		top_k_i_val = 0
		top_k_i_ind = 0
		for j in range(CNN_input.NUM_CLASSES):
			if (prediction_cpy[j] > top_k_i_val):
				top_k_i_val = prediction_cpy[j]
				top_k_i_ind = j
		prediction_cpy[top_k_i_ind] = -top_k_i_ind
		top_k.append(top_k_i_ind)

	return top_k



def display_result(image, label, prediction):
	top_k = get_top_k(prediction, DISPLAY_TOP_K)

	top_k_values = []
	top_k_labels = []
	for i in range(DISPLAY_TOP_K):
		top_k_values.append(math.ceil(prediction[top_k[i]]*100))
		top_k_labels.append(LABELS[top_k[i]])

	fig, ax = plt.subplots()
	bar_ind = np.arange(DISPLAY_TOP_K)
	width = 0.2
	bars = ax.bar(bar_ind, tuple(top_k_values), width, color='r')

	for i in range(DISPLAY_TOP_K):
		if (top_k[i] == label):
			bars[i].set_color('g')

	ax.set_title('Image Top ' + str(DISPLAY_TOP_K) + ' Results')
	ax.set_ylabel('Confidence')

	ax.set_xticks(bar_ind)
	ax.set_xticklabels(tuple(top_k_labels))

	imagebox = OffsetImage(image, zoom=2.5, interpolation='bicubic')
	imagebox.image.axes = ax

	ab = AnnotationBbox(imagebox, [0, 0], xybox=(.75, .75), xycoords='data', boxcoords='axes fraction')

	ax.add_artist(ab)

	offsetbox = TextArea(LABELS[label], minimumdescent=False)

	ab = AnnotationBbox(offsetbox, [0,0], xybox=(.75, .50), xycoords='data', boxcoords='axes fraction')

	ax.add_artist(ab)

	plt.show(bars)


def main(argv=None):
	CNN_input.extract()

	images, display_images, labels = CNN_input.construct_inputs(True, True)
	logits = CNN_model.inference(False, images)

	variable_averages = tf.train.ExponentialMovingAverage(CNN_model.MOVING_AVERAGE_DECAY)
	variables_to_restore = variable_averages.variables_to_restore()
	saver = tf.train.Saver(variables_to_restore)

	while True:
		with tf.Session() as sess:
			check_point = tf.train.get_checkpoint_state(FLAGS.train_dir)
			if check_point and check_point.model_checkpoint_path:
				saver.restore(sess, check_point.model_checkpoint_path.replace("E:\\", "D:\\"))
			else:
				raise IOError("No Checkpoint file found!")

			coord = tf.train.Coordinator()
			threads = []
			#try:
			for queue_runner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(queue_runner.create_threads(sess, coord=coord, daemon=True, start=True))

			session_result = sess.run([tf.nn.softmax(logits), labels, display_images])
			for i in range(0, FLAGS.batch_size):
				display_result(reformat_image(session_result[2][i]), session_result[1][i], session_result[0][i])

			coord.request_stop()
			coord.join(threads, stop_grace_period_secs=10)
			#except Exception as e:
			#	print(e.__traceback__)
			#	coord.request_stop()


if __name__ == '__main__':
	tf.app.run()