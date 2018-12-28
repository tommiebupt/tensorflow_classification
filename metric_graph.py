from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
from nets import nets_factory
slim = tf.contrib.slim
from eval_metric import streamingConfusionMatrix

class MetricGraph(object):
	def __init__(self, 
		inputs, 
		labels, 
		num_classes, 
		model_name = "resnet_v1_50", 
		labels_offset = 0,
		moving_average_decay = None): #smooth variable.

		self._input = inputs #tf.placeholder(tf.float32, [None, None, None])
		self._labels = labels #tf.placeholder(tf.float32, [None, None])
		self._labels = tf.squeeze(self._labels) #tf.placeholder(tf.float32, [None, None])
		self._model_name = model_name
		self._num_classes = num_classes
		self._labels_offset = labels_offset

		#DEFAULT
		self._moving_average_decay = moving_average_decay

		self._all_update_ops = []

		#RETURN
		self._eval_op = None
		self._final_op = None
		self._variables_to_restore = None

		self._build_network()
	#enddef
	
	def eval_ops(self):
		return self._eval_op, self._final_op
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
			self._predictions = tf.argmax(self._logits, 1)
			

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
			self._eval_metrics()

			self._merge_all_update_ops()
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
	
	def _eval_metrics(self):
		names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        		'Accuracy': tf.metrics.accuracy(self._labels, self._predictions),
        		'Recall_2': slim.metrics.streaming_sparse_recall_at_k(self._logits, self._labels, 2),
			'Confusion_matrix': streamingConfusionMatrix(self._labels, 
							self._predictions, num_classes=self._num_classes),
    		})	
		
		self._all_update_ops.extend(list(names_to_updates.values()))
		self._final_op = names_to_values['Confusion_matrix']

		# Print the summaries to screen.
    		for name, value in names_to_values.items():
			if name <> "Confusion_matrix":
      				summary_name = 'eval/%s' % name
      				op = tf.summary.scalar(summary_name, value, collections=[])
      				op = tf.Print(op, [value], summary_name)
      				tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
			else:
				indices = [(l,l) for l in range(self._num_classes)] 
				tp = tf.gather_nd(value, indices=indices) 
  				positive = tf.reduce_sum(value, -1)
  				prediction = tf.reduce_sum(value, 0)

				recall = tf.truediv(tp+1,positive+1)
				averge_recall = tf.reduce_mean(recall)

				precision = tf.truediv(tp+1, prediction+1)
				averge_precision = tf.reduce_mean(precision)

				summary_name = 'eval/Averge_recall'
      				op = tf.summary.scalar(summary_name, averge_recall, collections=[])
      				op = tf.Print(op, [averge_recall], summary_name)
      				tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
				 
				summary_name = 'eval/Averge_precision'
      				op = tf.summary.scalar(summary_name, averge_precision, collections=[])
      				op = tf.Print(op, [averge_precision], summary_name)
      				tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
	#enddef

	def _merge_all_update_ops(self):
		self._eval_op = tf.group(*self._all_update_ops)
	#enddef

#endclass


if __name__ == "__main__":
	labels = tf.cast(tf.convert_to_tensor(np.random.randint(2, size=100)),tf.int32)
	images = tf.cast(tf.convert_to_tensor(np.random.random([100, 256, 256, 3])),tf.float32)
	
	train_op, summary_op = OptimizeGraph(images, labels, 2, 100, 100).train_ops()

	with tf.Session() as sess:
		init_op = tf.group([tf.global_variables_initializer(),
				tf.local_variables_initializer(), tf.tables_initializer()])
		sess.run(init_op)
		total_loss, _ =sess.run([train_op,summary_op])
		print(total_loss)
