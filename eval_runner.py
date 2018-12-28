from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import math
import tensorflow as tf
from eval_graph import EvalGraph 
slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

class EvalRunner(object):
	def __init__(self, dataset, 
		train_dir="./output/tfmodel",
		eval_dir="./output/eval",
		width=256,
		height=256,
		batch_size=64,
		eval_interval_secs = 0,
		max_num_of_evaluations = None, 
		per_process_gpu_memory_fraction=1.0):

		self._train_dir = train_dir
		self._eval_dir = eval_dir
		self._num_samples = dataset.get_num_samples()
		self._batch_size = batch_size
		self._eval_interval_secs = eval_interval_secs 
		self._max_num_of_evaluations = max_num_of_evaluations

		self._session_config = tf.ConfigProto()
		self._session_config.gpu_options.allow_growth = True
		self._session_config.gpu_options.per_process_gpu_memory_fraction = \
			 			per_process_gpu_memory_fraction 

		self._eval_op, self._final_op, self._variables_to_restore =\
				 EvalGraph(dataset, 256, 256, self._batch_size).get_eval_ops() 
	#enddef

	def start(self):
		if self._eval_op is None or self._final_op is None or self._variables_to_restore is None:
			raise ValueError('eval_op or final_op or variables_to_restore cannot be None.')

		num_batches = math.ceil(self._num_samples/float(self._batch_size))-1
		print("will run %d batches."%num_batches)

		graph = tf.get_default_graph()
		with graph.as_default():
			confusion_matrix = slim.evaluation.evaluate_once(
        				master="",
        				checkpoint_path = tf.train.latest_checkpoint(self._train_dir),
        				logdir=self._eval_dir,
        				num_evals=num_batches,
					session_config = self._session_config,
					final_op = self._final_op,
        				eval_op=self._eval_op,
        				variables_to_restore=self._variables_to_restore)
			"""
			confusion_matrix = slim.evaluation.evaluation_loop(
        				master="",
        				checkpoint_dir = self._train_dir,
        				logdir=self._eval_dir,
        				num_evals=num_batches,
					session_config = self._session_config,
					final_op = self._final_op,
					max_number_of_evaluations = self._max_num_of_evaluations,
        				eval_op=self._eval_op,
        				eval_interval_secs=self._eval_interval_secs,
        				variables_to_restore=self._variables_to_restore)
			"""
			print(confusion_matrix)
	#enddef
#endclass

if __name__ == "__main__":
	tf.logging.set_verbosity(tf.logging.INFO)
	from gender_data import GenderEvalData 
	g_eval_data = GenderEvalData(num_classes=2, label_info={'male':0, 'female':1})
	g_eval_data.load_data("../../data/jd_image/data/1315_gender_model_train_samples.dat", 
			"../../data/jd_image/train")
	EvalRunner(g_eval_data).start()
