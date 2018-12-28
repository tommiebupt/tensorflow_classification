from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
from nets import nets_factory
slim = tf.contrib.slim

class InferGraph(object):
	def __init__(self, 
		inputs, 
		num_classes, 
		labels_offset = 0,
		model_name = "resnet_v1_50", 
		moving_average_decay = None): #smooth variable.

		self._input = inputs #tf.placeholder(tf.float32, [None, None, None])
		self._model_name = model_name
		self._num_classes = num_classes
		self._labels_offset = labels_offset

		#DEFAULT
		self._moving_average_decay = moving_average_decay

		#RETURN
		self._probs = None
		self._predictions = None
		self._variables_to_restore = None

		self._build_network()
	#enddef
	
	def infer_ops(self):
		return self._probs, self._predictions
	#enddef
	
	def get_variables_to_restore(self):
		#print(self._variables_to_restore)
		return self._variables_to_restore
	#enddef

	def _build_network(self):
		#with tf.Graph().as_default():
			#################
			#  stage 1
			#################

			#Build network.
			self._network_fn = nets_factory.get_network_fn(self._model_name,
					self._num_classes-self._labels_offset,
					is_training = False)
			#Forward network.
			self._logits, self._end_points = self._network_fn(self._input)
			self._predictions = tf.argmax(self._logits, 1, name="predictions")
			self._probs = tf.nn.softmax(self._logits, name="probabilities")
			
			for variable in slim.get_model_variables():
				tf.summary.histogram(variable.op.name, variable)
			for end_point in self._end_points:
				x = self._end_points[end_point]
				tf.summary.histogram('activations/'+end_point,x)
				tf.summary.scalar('sparsity/'+end_point, tf.nn.zero_fraction(x))

			#################
			# stage 2
			#################

			self._global_step = tf.train.create_global_step()
			self._get_variables_to_restore()

	#enddef
	
	def _get_variables_to_restore(self):
		if self._moving_average_decay:
			variable_averages = tf.train.ExponentialMovingAverage(self._moving_average_decay, 
						self._global_step)
			self._variables_to_restore  = variable_averages.variables_to_restore(
					slim.get_model_variables())
			self._variables_to_restore[self._global_step.op.name]=self._global_step
		else:
			self._variables_to_restore = slim.get_variables_to_restore()
	#enddef
#endclass


if __name__ == "__main__":
	labels = tf.cast(tf.convert_to_tensor(np.random.randint(2, size=10)),tf.int32)
	images = tf.cast(tf.convert_to_tensor(np.random.random([10, 256, 256, 3])),tf.float32)
	probs, predictions = InferGraph(images, labels, 2).infer_ops()

	with tf.Session() as sess:
		init_op = tf.group([
					tf.global_variables_initializer(), 
					tf.local_variables_initializer(), 
			])
		sess.run(init_op)
		np_probs, np_predictions = sess.run([probs, predictions])
		print(np.hstack((np.expand_dims(np_predictions,-1), np_probs)))
