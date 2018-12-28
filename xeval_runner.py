from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import pickle
import tensorflow as tf
from xeval_graph import XEvalGraph
slim = tf.contrib.slim
from gender_data import GenderEvalData
from visualize_model import *
 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

class XEvalRunner(object):
	def __init__(self, dataset, 
		saved_model_list=None,
		width=256,
		height=256,
		batch_size=64,
		per_process_gpu_memory_fraction=1.0):

		self._saved_model_list = saved_model_list 
		self._eval_results = {}
	
		self._session_config = tf.ConfigProto()
		self._session_config.gpu_options.allow_growth = True
		self._session_config.gpu_options.per_process_gpu_memory_fraction = \
			 			per_process_gpu_memory_fraction 

		self._filenames, self._labels, self._probs, self._predictions, self._variables_to_restore =\
			XEvalGraph(dataset, 256, 256, batch_size).get_eval_ops()	
	#enddef

	def run(self, saver, init_op, ckpt_path):
		file_names = []
		true_labels = []
		pred_labels = []
		pred_scores = []

		with tf.Session() as sess:
			sess.run(init_op)
			saver.restore(sess, ckpt_path)
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess, coord)
			try:
				while not coord.should_stop():
					np_filenames, np_labels, np_probs, np_predictions =\
								 sess.run([
									self._filenames,
									self._labels,
									self._probs,
									self._predictions])

					file_names.extend(np_filenames.tolist())
					true_labels.extend(np_labels.tolist())
					pred_labels.extend(np_predictions.tolist())
					pred_scores.extend(list(zip(*(np_probs.tolist()))[1]))

					#print(np.hstack(( np.expand_dims(np_filenames, -1),
					#		 np.expand_dims(np_labels, -1),
					#		 #np_probs,
					#		 np.expand_dims(np_predictions, -1))))					
					#print(pred_scores)

			except tf.errors.OutOfRangeError as e:
				pass
				#tf.logging.info('Caught OutOfRangeError. Stopping Training. %s', e)
			finally :
				coord.request_stop()
				tf.logging.info('request to stop all threads!')
			coord.join(threads)
			tf.logging.info('all threads are stopped!')
		return zip(file_names, true_labels, pred_labels, pred_scores)
	#enddef

	def start(self):
		if self._probs is None or self._predictions is None or self._variables_to_restore is None:
			raise ValueError('results of inference cannot be None.')
		
		graph = tf.get_default_graph()
		with graph.as_default():
			with tf.name_scope('init_ops'):
				global_init_op = tf.global_variables_initializer()
				
				#1-D tensor: names of uninitialized variables.
				#ready_op = tf.report_uninitialized_variables()

				local_init_op = tf.group(
					tf.local_variables_initializer(), #ops: initialize all local variables. 
					tf.tables_initializer()	#ops:initialize all tables. NoOp if nonexists.
				)

			saver = tf.train.Saver(self._variables_to_restore)
			init_op = tf.group([global_init_op, local_init_op])
			for ckpt_path in self._saved_model_list:
				self._eval_results[ckpt_path.split("/")[-1]] = self.run(saver, init_op, ckpt_path)
		return self._eval_results	
	#enddef
#endclass

if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	g_eval_data = GenderEvalData(num_classes=2, label_info={'male':0, 'female':1})
	g_eval_data.load_data("../../data/jd_image/data/1315_gender_model_eval_samples.dat", 
			"../../data/jd_image/eval",
			third_cate_list=None)
	model_list = [
			'./output/tfmodel/model.ckpt-77624'
		]
	results = XEvalRunner(g_eval_data, model_list).start()
	with open('./output/pkl/eval_results.pkl', 'wb') as fw:
		pickle.dump(results, fw)
