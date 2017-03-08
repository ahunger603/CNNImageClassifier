import tensorflow as tf
import numpy as np
import math
import time
from datetime import datetime

import CNN_input
import CNN_model

FLAGS = CNN_input.FLAGS

def eval_once(saver, summary_writer, top_k_op, summary_op):
	with tf.Session() as sess:
		check_point = tf.train.get_checkpoint_state(FLAGS.train_dir)
		if check_point and check_point.model_checkpoint_path:
			saver.restore(sess, check_point.model_checkpoint_path)
			global_step = check_point.model_checkpoint_path.split('/')[-1].split('-')[-1]
		else:
			raise ValueError("No checkpoint file found")

		coord = tf.train.Coordinator()
		threads = []
		try:
			for queue_runner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
				threads.extend(queue_runner.create_threads(sess, coord=coord, daemon=True, start=True))

			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			true_count = 0
			total_sample_count = num_iter * FLAGS.batch_size
			step = 0
			while step < num_iter:
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				step += 1
				if step % 10 == 0:
					print("%.2f%%" % ((step / num_iter) * 100))

			print(str(true_count) + ' ' + str(total_sample_count) + ' ' + str(step))
			precision = true_count / total_sample_count
			print('%s: precision @ 3 = %.3f' % (datetime.now(), precision))

			summary = tf.Summary()
			summary.ParseFromString(sess.run(summary_op))
			summary.value.add(tag='Precision @ 3', simple_value=precision)
			summary_writer.add_summary(summary, global_step)

			coord.request_stop()
			coord.join(threads, stop_grace_period_secs=10)
		except Exception as e:
			print(e)
			coord.request_stop(e)

def evaluate(run_once):
	with tf.Graph().as_default() as graph:
		images, display_images, labels = CNN_input.construct_inputs(True, False)

		logits = CNN_model.inference(False, images)

		top_k_op = tf.nn.in_top_k(logits, labels, 3)

		variable_averages = tf.train.ExponentialMovingAverage(CNN_model.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph)

		while True:
			eval_once(saver, summary_writer, top_k_op, summary_op)
			if run_once:
				break
			time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
	CNN_input.extract()
	if (tf.gfile.Exists(FLAGS.eval_dir)):
		tf.gfile.DeleteRecursively(FLAGS.eval_dir)
	tf.gfile.MakeDirs(FLAGS.eval_dir)
	evaluate(False)

if __name__ == '__main__':
	tf.app.run()
