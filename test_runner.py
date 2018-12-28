from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import tensorflow as tf
from test_graph import TestGraph
slim = tf.contrib.slim
from gender_data import GenderTestData 
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="2"

class TestRunner(object):
	def __init__(self, dataset, 
		train_dir="./output/tfmodel",
		width=256,
		height=256,
		batch_size=32,
		per_process_gpu_memory_fraction=0.5):

		self._train_dir = train_dir
		self._session_config = tf.ConfigProto()
		self._session_config.gpu_options.allow_growth = True
		self._session_config.gpu_options.per_process_gpu_memory_fraction = \
			 			per_process_gpu_memory_fraction 

		self._file_names, self._probs, self._predictions, self._variables_to_restore =\
			TestGraph(dataset, 256, 256, batch_size).get_test_ops()	
	#enddef

	def start(self):
		if self._probs is None or self._predictions is None or self._variables_to_restore is None:
			raise ValueError('results of inference cannot be None.')

		graph = tf.get_default_graph()
		with graph.as_default():
			with tf.name_scope('init_ops'):
				init_op = tf.global_variables_initializer()
				
				#1-D tensor: names of uninitialized variables.
				#ready_op = tf.report_uninitialized_variables()

				local_init_op = tf.group(
					tf.local_variables_initializer(), #ops: initialize all local variables. 
					tf.tables_initializer()	#ops:initialize all tables. NoOp if nonexists.
				)

			saver = tf.train.Saver(self._variables_to_restore)
			with tf.Session() as sess:
				sess.run(tf.group([init_op, local_init_op]))
				saver.restore(sess, tf.train.latest_checkpoint(self._train_dir))

				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(sess, coord)
				try:
					while not coord.should_stop():
						np_file_names, np_probs, np_predictions = sess.run([
									self._file_names,
									self._probs,
									self._predictions])
						print(np.hstack((np.expand_dims(np_file_names, -1),
								 np_probs,
								 np.expand_dims(np_predictions, -1))))					
				except tf.errors.OutOfRangeError as e:
					tf.logging.info('Caught OutOfRangeError. Stopping Training. %s', e)
				finally :
					coord.request_stop()
					tf.logging.info('request to stop all threads!')
				coord.join(threads)
				tf.logging.info('all threads are stopped!')
	#enddef
#endclass

if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	g_test_data = GenderTestData(num_classes=2, label_info={'male':0, 'female':1})
	g_test_data.load_data("../../data/jd_image/test", "*.jpg")
	TestRunner(g_test_data).start()
