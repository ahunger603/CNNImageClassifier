from datetime import datetime
import time

import numpy as np
import tensorflow as tf

import CNN_input
import CNN_model

FLAGS = CNN_input.FLAGS


def train():
	with (tf.Graph().as_default()):
		global_step = tf.contrib.framework.get_or_create_global_step()

		images, labels = CNN_input.construct_inputs(False, True)

		logits = CNN_model.inference(images)

		loss = CNN_model.loss(logits, labels)

		train_op = CNN_model.train(loss, global_step)

		# Logs loss and runtime
		class _LoggerHook(tf.train.SessionRunHook):

			def begin(self):
				self._step = -1

			def before_run(self, run_context):
				self._step += 1
				self._start_time = time.time()
				return tf.train.SessionRunArgs(loss)

			def after_run(self, run_context, run_values):
				duration = time.time() - self._start_time
				loss_value = run_values.results
				if (self._step % 10 == 0):
					return ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)' %
							(datetime.now(), self._step, loss_value, FLAGS.batch_size / duration, float(duration)))

		with tf.train.MonitoredTrainingSession(
			checkpoint_dir=FLAGS.train_dir,
			hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
				   tf.train.NanTensorHook(loss),
				   _LoggerHook()],
			config=tf.ConfigProto(
				log_device_placement=FLAGS.log_device_placement)) as mon_sess:
			while (not mon_sess.should_stop()):
				mon_sess.run(train_op)


def main(argv=None):
	CNN_input.download_extract()

	if (tf.gfile.Exists(FLAGS.train_dir)):
		tf.gfile.DeleteRecursively(FLAGS.train_dir)
	tf.gfile.MakeDirs(FLAGS.train_dir)

	train()


if __name__ == "__main__":
	tf.app.run()
